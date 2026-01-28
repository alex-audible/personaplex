# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""MLX model loader for PersonaPlex.

Provides:
- personaplex_config() -> LmConfig matching PersonaPlex's architecture
- get_personaplex_lm() to instantiate the Kyutai Lm with PersonaPlex weights

Usage:
    from moshi_mlx.loaders_mlx import get_personaplex_lm
    model = get_personaplex_lm("personaplex_mlx.safetensors")
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from moshi_mlx.models.lm import (
    DepFormerConfig,
    Lm,
    LmConfig,
)
from moshi_mlx.modules.transformer import TransformerConfig

logger = logging.getLogger(__name__)


def personaplex_config() -> LmConfig:
    """Return an LmConfig that matches the PersonaPlex 7B architecture.

    Key differences from Kyutai's config_v0_1:
    - dim_feedforward = 16896 (hidden_scale=4.125 * 4096), NOT 16384 (4 * 4096)
    - depformer dim_feedforward = 4224 (4.125 * 1024), NOT 4096 (4 * 1024)
    - audio_delays pattern differs: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    - text_in_vocab_size = 32001, text_out_vocab_size = 32000
    """
    transformer = TransformerConfig(
        d_model=4096,
        num_heads=32,
        num_layers=32,
        dim_feedforward=16896,  # 4.125 * 4096 = 16896 (NOT 4 * 4096 = 16384)
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=3000,
        max_period=10000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=4224,  # 4.125 * 1024 = 4224 (NOT 4 * 1024 = 4096)
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=8,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=8,  # 8 depformer slices for inference
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,       # card + 1 = 2048 + 1
        text_in_vocab_size=32001,    # text_card + 1 = 32000 + 1
        text_out_vocab_size=32000,   # text_card
        audio_codebooks=16,          # n_q = 16
        # delays from _lm_kwargs: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        # audio_delays = delays[1:] (skip the first text delay)
        audio_delays=[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        conditioners={},
    )


def get_personaplex_lm(
    weights_path: str | Path,
    dtype: mx.Dtype = mx.float16,
) -> Lm:
    """Create and load a PersonaPlex Lm model with converted MLX weights.

    Args:
        weights_path: Path to the converted MLX safetensors file
                      (output of convert_weights.py).
        dtype: Target dtype for the model weights (default: float16).

    Returns:
        A fully loaded Lm model ready for inference.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"MLX weights not found at {weights_path}. "
            f"Run convert_weights.py first to create them."
        )

    cfg = personaplex_config()
    logger.info("Creating PersonaPlex MLX model (num_slices=%d) ...", cfg.depformer.num_slices)
    model = Lm(cfg)

    logger.info("Loading MLX weights from %s ...", weights_path)
    weights = mx.load(str(weights_path))
    logger.info("Loaded %d weight tensors", len(weights))

    # Cast weights to target dtype if needed
    if dtype != mx.float16:
        logger.info("Casting weights to %s ...", dtype)
        weights = {k: v.astype(dtype) for k, v in weights.items()}

    # Collect model's expected leaf keys for diagnostics
    model_keys = set()

    def _collect_keys(params, prefix=""):
        if isinstance(params, mx.array):
            model_keys.add(prefix)
        elif isinstance(params, dict):
            for k, v in params.items():
                _collect_keys(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(params, (list, tuple)):
            for i, v in enumerate(params):
                _collect_keys(v, f"{prefix}.{i}")

    _collect_keys(model.parameters())

    weight_keys = set(weights.keys())
    missing = model_keys - weight_keys
    unexpected = weight_keys - model_keys

    if missing:
        logger.warning("Missing keys (%d): %s", len(missing), sorted(missing))
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), sorted(unexpected))

    # Load weights into the model (strict=False to allow partial loading)
    model.load_weights(list(weights.items()), strict=False)

    # Force evaluation of all parameters to ensure weights are loaded
    mx.eval(model.parameters())

    logger.info("PersonaPlex MLX model loaded successfully")
    return model


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Default weights path
    default_path = "personaplex_mlx.safetensors"
    weights_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    print(f"Loading PersonaPlex MLX model from: {weights_path}")
    model = get_personaplex_lm(weights_path)

    # Print model summary
    print("\n=== PersonaPlex MLX Model Summary ===")
    print(f"Config: {model.cfg}")
    print(f"  audio_codebooks (n_q): {model.n_q}")
    print(f"  depformer slices (dep_q): {model.dep_q}")
    print(f"  audio_delays: {model.delays}")

    # Count parameters
    def count_params(params):
        """Recursively count total parameters."""
        total = 0
        if isinstance(params, mx.array):
            return params.size
        elif isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, (list, tuple)):
            for v in params:
                total += count_params(v)
        return total

    params = model.parameters()
    total = count_params(params)
    print(f"\nTotal parameters: {total:,} ({total / 1e9:.2f}B)")

    # Print leaf parameter shapes
    print("\n=== Leaf Parameter Shapes ===")

    def print_params(params, prefix=""):
        """Recursively print leaf parameter shapes."""
        if isinstance(params, mx.array):
            print(f"  {prefix}: {params.shape} ({params.dtype})")
        elif isinstance(params, dict):
            for k in sorted(params.keys()):
                print_params(params[k], f"{prefix}.{k}" if prefix else k)
        elif isinstance(params, (list, tuple)):
            for i, v in enumerate(params):
                print_params(v, f"{prefix}.{i}")

    print_params(params)

    # Verify no missing weights by checking for zero-initialized parameters
    print("\n=== Weight Loading Verification ===")
    counts = {"leaf": 0, "zero": 0}

    def check_zeros(params, prefix=""):
        if isinstance(params, mx.array):
            counts["leaf"] += 1
            if mx.all(params == 0).item():
                counts["zero"] += 1
                print(f"  WARNING: all-zero parameter: {prefix} {params.shape}")
        elif isinstance(params, dict):
            for k in sorted(params.keys()):
                check_zeros(params[k], f"{prefix}.{k}" if prefix else k)
        elif isinstance(params, (list, tuple)):
            for i, v in enumerate(params):
                check_zeros(v, f"{prefix}.{i}")

    check_zeros(params)
    print(f"  Checked {counts['leaf']} leaf parameters, {counts['zero']} are all-zero")
    if counts["zero"] == 0:
        print("  All parameters have non-zero values - weights loaded correctly!")
    else:
        print("  WARNING: Some parameters are all-zero, check weight mapping!")
