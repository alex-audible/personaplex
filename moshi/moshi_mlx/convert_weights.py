# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Convert PersonaPlex PyTorch safetensors weights to MLX format.

This script:
1. Downloads PersonaPlex weights from nvidia/personaplex-7b-v1 on HuggingFace
2. Applies dep_q expansion patches (self_attn doubling + codebook 0..7 -> 8..15 copy)
3. Maps PyTorch key names to Kyutai MLX key names
4. Saves as MLX safetensors (float16 by default)

Usage:
    python -m moshi_mlx.convert_weights [--output PATH] [--dtype float16|float32|bfloat16]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

# PersonaPlex architecture constants
_NUM_SLICES = 8           # inference-time depformer slices
_DEP_Q_EXPANDED = 16      # checkpoint dep_q (expanded for weight loading)
_DEPFORMER_NUM_LAYERS = 6
_AUDIO_CODEBOOKS = 16
_DEPFORMER_CHUNKS = 16    # number of chunks in the expanded self_attn weights


def _download_weights(repo_id: str = "nvidia/personaplex-7b-v1",
                      filename: str = "model.safetensors") -> str:
    """Download PersonaPlex weights from HuggingFace, returning the local path."""
    from huggingface_hub import hf_hub_download
    logger.info("Downloading %s/%s ...", repo_id, filename)
    path = hf_hub_download(repo_id, filename)
    logger.info("Weights cached at %s", path)
    return path


def _apply_dep_q_patches(state_dict: dict, model_keys: set[str]) -> dict:
    """Apply PersonaPlex dep_q expansion patches to a PyTorch state_dict.

    Patch 1: Expand depformer self_attn weights by concatenating tensor
             with itself (doubling along dim 0) when checkpoint shape is
             smaller than the expected dep_q=16 shape.

    Patch 2: Copy codebook indices 0..7 to 8..15 for gating, linears,
             depformer_in, and depformer_emb keys that are missing from
             the checkpoint.

    Args:
        state_dict: The raw checkpoint state_dict (PyTorch tensors).
        model_keys: The full set of expected model state_dict keys
                    (from a dep_q=16 model).

    Returns:
        The patched state_dict.
    """
    import torch

    # Patch 1: expand depformer self_attn weights if needed
    for name in list(state_dict.keys()):
        if "depformer" not in name or "self_attn" not in name:
            continue
        tensor = state_dict[name]
        if name not in model_keys:
            continue
        # Expected shapes for dep_q=16:
        #   in_proj_weight: [16 * 3 * d_model, d_model] = [49152, 1024]
        #   out_proj.weight: [16 * d_model, d_model] = [16384, 1024]
        # If checkpoint has half that (dep_q=8), double it.
        if "in_proj_weight" in name:
            expected_dim0 = _DEP_Q_EXPANDED * 3 * 1024  # 49152
        elif "out_proj.weight" in name:
            expected_dim0 = _DEP_Q_EXPANDED * 1024  # 16384
        else:
            continue

        if tensor.shape[0] < expected_dim0:
            logger.info("Patch 1: expanding %s from %s (doubling dim 0)", name, tensor.shape)
            state_dict[name] = torch.cat([tensor, tensor], dim=0)
        else:
            logger.debug("Patch 1: %s already has expected shape %s", name, tensor.shape)

    # Patch 2: fill missing keys by copying 0..7 -> 8..15
    to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
    for name in sorted(model_keys):
        if name in state_dict:
            continue
        replaced = False
        for old_idx, new_idx in zip(range(8), range(8, 16)):
            for rep in to_replace:
                needle = f"{rep}.{new_idx}."
                if needle in name:
                    src = name.replace(needle, f"{rep}.{old_idx}.")
                    if src in state_dict:
                        logger.info("Patch 2: copying %s <- %s", name, src)
                        state_dict[name] = state_dict[src]
                        replaced = True
                    break
            if replaced:
                break
        if not replaced and name not in state_dict:
            logger.warning("Missing key after patches: %s", name)

    return state_dict


def _compute_expected_model_keys() -> set[str]:
    """Compute the expected PyTorch state_dict keys for a dep_q=16 PersonaPlex model.

    Rather than instantiating the full PyTorch model (which would need ~14GB),
    we generate the expected key names programmatically from the known architecture.
    """
    keys = set()

    # Top-level keys
    keys.add("out_norm.alpha")
    keys.add("text_emb.weight")
    keys.add("text_linear.weight")
    keys.add("depformer_text_emb.weight")

    # Audio embeddings: emb.{0..15}
    for i in range(_AUDIO_CODEBOOKS):
        keys.add(f"emb.{i}.weight")

    # Transformer layers (32 layers)
    for layer_idx in range(32):
        prefix = f"transformer.layers.{layer_idx}"
        keys.add(f"{prefix}.norm1.alpha")
        keys.add(f"{prefix}.norm2.alpha")
        keys.add(f"{prefix}.self_attn.in_proj_weight")
        keys.add(f"{prefix}.self_attn.out_proj.weight")
        keys.add(f"{prefix}.gating.linear_in.weight")
        keys.add(f"{prefix}.gating.linear_out.weight")

    # Depformer per-step weights (dep_q=16 expanded)
    for step_idx in range(_DEP_Q_EXPANDED):
        keys.add(f"depformer_in.{step_idx}.weight")
        keys.add(f"linears.{step_idx}.weight")

    # depformer_emb: 0..14 (slice 0 uses depformer_text_emb instead)
    for i in range(_DEP_Q_EXPANDED - 1):
        keys.add(f"depformer_emb.{i}.weight")

    # Depformer layers (6 layers, shared across slices but with per-step gating)
    for layer_idx in range(_DEPFORMER_NUM_LAYERS):
        prefix = f"depformer.layers.{layer_idx}"
        keys.add(f"{prefix}.norm1.alpha")
        keys.add(f"{prefix}.norm2.alpha")
        keys.add(f"{prefix}.self_attn.in_proj_weight")
        keys.add(f"{prefix}.self_attn.out_proj.weight")
        for step_idx in range(_DEP_Q_EXPANDED):
            keys.add(f"{prefix}.gating.{step_idx}.linear_in.weight")
            keys.add(f"{prefix}.gating.{step_idx}.linear_out.weight")

    return keys


def _map_pytorch_to_mlx(pth_state_dict: dict) -> dict[str, mx.array]:
    """Map patched PyTorch state_dict keys to Kyutai MLX key names.

    This replicates the logic from moshi_mlx.models.lm.Lm.load_pytorch_weights()
    but works on numpy/mlx arrays directly (no need for a live MLX model).

    The mapping handles:
    - out_norm.alpha[1,1,D] -> out_norm.weight[D]
    - emb.{N} -> audio_embs.{N}
    - transformer.*.alpha -> transformer.*.weight
    - transformer.*.in_proj_weight -> transformer.*.in_proj.weight
    - depformer per-slice remapping with self_attn splitting

    Args:
        pth_state_dict: Fully patched PyTorch state_dict (dep_q=16).

    Returns:
        Dictionary of MLX key names -> mx.array values.
    """
    import numpy as np

    def to_mx(tensor) -> mx.array:
        """Convert a PyTorch tensor to an MLX array via numpy."""
        return mx.array(tensor.float().numpy())

    mlx_t: dict[str, mx.array] = {}

    # out_norm: alpha[1,1,D] -> weight[D]
    mlx_t["out_norm.weight"] = to_mx(pth_state_dict["out_norm.alpha"])[0, 0]

    # Text embeddings and linear
    for name in ["text_emb.out1.weight", "text_emb.out2.weight",
                 "text_emb.weight", "text_linear.weight"]:
        if name in pth_state_dict:
            mlx_t[name] = to_mx(pth_state_dict[name])

    # Audio embeddings: emb.{N} -> audio_embs.{N}
    for cb_idx in range(_AUDIO_CODEBOOKS):
        mlx_t[f"audio_embs.{cb_idx}.weight"] = to_mx(
            pth_state_dict[f"emb.{cb_idx}.weight"]
        )

    # Transformer layers
    for k in sorted(pth_state_dict.keys()):
        if k.startswith("transformer"):
            v = to_mx(pth_state_dict[k])
            if k.endswith(".alpha"):
                v = v[0, 0]
            k = k.replace(".alpha", ".weight")
            k = k.replace(".in_proj_weight", ".in_proj.weight")
            mlx_t[k] = v

    # Condition provider and extra heads (PersonaPlex doesn't use these, but
    # include for completeness)
    for k in sorted(pth_state_dict.keys()):
        if k.startswith("condition_provider.") or k.startswith("extra_heads."):
            mlx_t[k] = to_mx(pth_state_dict[k])

    # DepFormer slices
    # PersonaPlex: num_slices=8 for inference, but checkpoint has dep_q=16.
    # We split self_attn by _DEPFORMER_CHUNKS=16 and take indices 0..7.
    # pth_idx = slice_idx (no weights_per_step_schedule).
    for slice_idx in range(_NUM_SLICES):
        pth_idx = slice_idx
        slice_p = f"depformer.slices.{slice_idx}"

        # linear_in: depformer_in.{pth_idx} -> depformer.slices.{slice_idx}.linear_in
        mlx_t[f"{slice_p}.linear_in.weight"] = to_mx(
            pth_state_dict[f"depformer_in.{pth_idx}.weight"]
        )

        # linear_out: linears.{slice_idx} -> depformer.slices.{slice_idx}.linear_out
        mlx_t[f"{slice_p}.linear_out.weight"] = to_mx(
            pth_state_dict[f"linears.{slice_idx}.weight"]
        )

        # embeddings
        if slice_idx == 0:
            mlx_t[f"{slice_p}.emb.weight"] = to_mx(
                pth_state_dict["depformer_text_emb.weight"]
            )
            # Check for low_rank, out1, out2 variants
            for _n in ["low_rank", "out1", "out2"]:
                src_key = f"depformer_text_emb.{_n}.weight"
                if src_key in pth_state_dict:
                    mlx_t[f"{slice_p}.emb.{_n}.weight"] = to_mx(
                        pth_state_dict[src_key]
                    )
        else:
            mlx_t[f"{slice_p}.emb.weight"] = to_mx(
                pth_state_dict[f"depformer_emb.{slice_idx - 1}.weight"]
            )
            lr_key = f"depformer_emb.{slice_idx - 1}.low_rank.weight"
            if lr_key in pth_state_dict:
                mlx_t[f"{slice_p}.emb.low_rank.weight"] = to_mx(
                    pth_state_dict[lr_key]
                )

        # DepFormer transformer layers
        for layer_idx in range(_DEPFORMER_NUM_LAYERS):
            p = f"{slice_p}.transformer.layers.{layer_idx}"

            # Norms: alpha[1,1,D] -> weight[D]
            mlx_t[f"{p}.norm1.weight"] = to_mx(
                pth_state_dict[f"depformer.layers.{layer_idx}.norm1.alpha"]
            )[0, 0]
            mlx_t[f"{p}.norm2.weight"] = to_mx(
                pth_state_dict[f"depformer.layers.{layer_idx}.norm2.alpha"]
            )[0, 0]

            # Gating: per-step gating.{pth_idx}
            mlx_t[f"{p}.gating.linear_in.weight"] = to_mx(
                pth_state_dict[
                    f"depformer.layers.{layer_idx}.gating.{pth_idx}.linear_in.weight"
                ]
            )
            mlx_t[f"{p}.gating.linear_out.weight"] = to_mx(
                pth_state_dict[
                    f"depformer.layers.{layer_idx}.gating.{pth_idx}.linear_out.weight"
                ]
            )

            # Self-attention: split by _DEPFORMER_CHUNKS and take pth_idx
            in_proj_full = to_mx(
                pth_state_dict[
                    f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"
                ]
            )
            out_proj_full = to_mx(
                pth_state_dict[
                    f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"
                ]
            )
            mlx_t[f"{p}.self_attn.in_proj.weight"] = mx.split(
                in_proj_full, _DEPFORMER_CHUNKS
            )[pth_idx]
            mlx_t[f"{p}.self_attn.out_proj.weight"] = mx.split(
                out_proj_full, _DEPFORMER_CHUNKS
            )[pth_idx]

    return mlx_t


def convert(
    output_path: str | Path,
    dtype_str: str = "float16",
    repo_id: str = "nvidia/personaplex-7b-v1",
    filename: str = "model.safetensors",
    input_path: str | None = None,
) -> Path:
    """Download, patch, remap, and save PersonaPlex weights in MLX format.

    Args:
        output_path: Where to save the MLX safetensors file.
        dtype_str: Target dtype ('float16', 'float32', or 'bfloat16').
        repo_id: HuggingFace repository ID.
        filename: Weight file name in the repository.
        input_path: If provided, use this local file instead of downloading.

    Returns:
        Path to the saved MLX weights file.
    """
    from safetensors.torch import load_file as torch_load_file

    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Use one of {list(dtype_map)}")
    target_dtype = dtype_map[dtype_str]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load PyTorch weights
    if input_path:
        src_path = input_path
    else:
        src_path = _download_weights(repo_id, filename)
    logger.info("Loading PyTorch weights from %s ...", src_path)
    state_dict = torch_load_file(src_path, device="cpu")
    logger.info("Loaded %d keys from checkpoint", len(state_dict))

    # Step 2: Compute expected keys and apply dep_q patches
    expected_keys = _compute_expected_model_keys()
    state_dict = _apply_dep_q_patches(state_dict, expected_keys)

    # Step 3: Map PyTorch keys to MLX keys
    logger.info("Mapping PyTorch keys to MLX format ...")
    mlx_weights = _map_pytorch_to_mlx(state_dict)
    logger.info("Mapped %d MLX keys", len(mlx_weights))

    # Step 4: Cast to target dtype
    logger.info("Casting to %s ...", dtype_str)
    for k in mlx_weights:
        mlx_weights[k] = mlx_weights[k].astype(target_dtype)

    # Step 5: Save
    logger.info("Saving MLX weights to %s ...", output_path)
    mx.save_safetensors(str(output_path), mlx_weights)

    # Verify saved file size
    file_size_gb = output_path.stat().st_size / (1024**3)
    logger.info("Saved %.2f GB to %s", file_size_gb, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PersonaPlex PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--output", "-o",
        default="personaplex_mlx.safetensors",
        help="Output path for MLX safetensors file (default: personaplex_mlx.safetensors)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Target dtype (default: float16)",
    )
    parser.add_argument(
        "--repo-id",
        default="nvidia/personaplex-7b-v1",
        help="HuggingFace repo ID (default: nvidia/personaplex-7b-v1)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to local PyTorch safetensors file (skip download)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    convert(
        output_path=args.output,
        dtype_str=args.dtype,
        repo_id=args.repo_id,
        input_path=args.input,
    )


if __name__ == "__main__":
    main()
