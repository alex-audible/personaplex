# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Offline inference entry point for PersonaPlex using Apple MLX + rustymimi.

This mirrors ``moshi.moshi.offline`` but replaces:
- PyTorch LMModel -> MLX Lm (via loaders_mlx)
- PyTorch Mimi -> rustymimi Tokenizer for encode/decode
- PyTorch LMGen -> Kyutai MLX LmGen (wrapped by PersonaPlexLmGen)
- torch samplers -> moshi_mlx.utils.Sampler

High-level flow:
  1. Load PersonaPlex MLX model via loaders_mlx.get_personaplex_lm()
  2. Load rustymimi Tokenizer for Mimi encode/decode
  3. Load sentencepiece tokenizer for text
  4. Create PersonaPlexLmGen with the right samplers
  5. Warmup the model
  6. (Optional) Voice prompt injection (stub for Stage 3)
  7. Read input WAV via sphn
  8. For each frame of user audio:
     a. Encode user audio with rustymimi.encode_step()
     b. Feed to PersonaPlexLmGen.step()
     c. If tokens returned, decode agent audio with rustymimi.decode_step()
     d. Collect PCM frames
  9. Concatenate frames, write output WAV via rustymimi.write_wav()
 10. Write text tokens to JSON

Usage::

    python -m moshi_mlx.offline_mlx \\
        --input-wav assets/test/input_assistant.wav \\
        --output-wav /tmp/claude/output_mlx.wav \\
        --output-text /tmp/claude/output_mlx.json \\
        --weights personaplex_mlx.safetensors \\
        --greedy --seed 42424242
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

from moshi_mlx.loaders_mlx import get_personaplex_lm
from moshi_mlx.lm_gen_mlx import PersonaPlexLmGen
from moshi_mlx.perf_logger import PerfLogger
from moshi_mlx.utils.sampling import Sampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching PersonaPlex loaders.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples per frame

DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"

# Codebook counts
NUM_GENERATED_CODEBOOKS = 8  # agent audio (depformer)
NUM_OTHER_CODEBOOKS = 8      # user audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(level: str, msg: str) -> None:
    """Coloured log matching Kyutai's client_utils.make_log style."""
    try:
        from moshi_mlx.client_utils import make_log
        print(make_log(level, msg))
    except ImportError:
        print(f"[{level.upper()}] {msg}")


def _seed_all(seed: int) -> None:
    """Seed MLX, numpy, and Python RNG for reproducibility."""
    mx.random.seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    _log("info", f"seeded all RNGs with {seed}")


def _wrap_system_tags(text: str) -> str:
    """Add PersonaPlex system tags if missing."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup(
    gen: PersonaPlexLmGen,
    audio_tokenizer: rustymimi.Tokenizer,
    num_steps: int = 4,
) -> None:
    """Run a short warmup loop to prime MLX compilation and caches.

    Feeds zero-PCM through the audio tokenizer and LmGen for a few steps.
    """
    _log("info", "warming up the model")
    for _ in range(num_steps):
        pcm_zeros = np.zeros((1, 1, FRAME_SIZE), dtype=np.float32)
        codes = audio_tokenizer.encode_step(pcm_zeros)
        # codes shape: [1, 1, num_codebooks] from rustymimi
        # Transpose to [B, num_codebooks, 1] then take [:, :other, :]
        codes_mx = mx.array(codes).transpose(0, 2, 1)[:, :NUM_OTHER_CODEBOOKS, :]
        out = gen.step(codes_mx)
        if out is not None:
            # Force evaluation to trigger compilation
            mx.eval(out)
            audio_tokens = out[:, 1:, :]  # [B, dep_q, 1]
            audio_np = np.array(audio_tokens).astype(np.uint32)
            _ = audio_tokenizer.decode_step(audio_np)
    mx.eval(gen.lm_gen.gen_sequence)
    _log("info", "warmup complete")


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    input_wav: str,
    output_wav: str,
    output_text: str,
    weights_path: str,
    mimi_weight: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    hf_repo: str = DEFAULT_HF_REPO,
    text_prompt: str = "",
    seed: Optional[int] = None,
    temp_audio: float = 0.8,
    temp_text: float = 0.7,
    topk_audio: int = 250,
    topk_text: int = 25,
    greedy: bool = False,
    perf_logger: Optional[PerfLogger] = None,
) -> None:
    """Run offline inference on an input WAV using PersonaPlex MLX.

    Args:
        input_wav: Path to input WAV file (user audio).
        output_wav: Path to write generated agent audio WAV.
        output_text: Path to write generated text tokens as JSON.
        weights_path: Path to PersonaPlex MLX safetensors weights.
        mimi_weight: Path to Mimi tokenizer weights. If None, downloaded from HF.
        tokenizer_path: Path to sentencepiece text tokenizer. If None, downloaded.
        hf_repo: HuggingFace repository for asset downloads.
        text_prompt: System text prompt for the model.
        seed: Random seed for reproducibility. None or -1 disables.
        temp_audio: Audio sampling temperature.
        temp_text: Text sampling temperature.
        topk_audio: Top-k for audio sampling.
        topk_text: Top-k for text sampling.
        greedy: If True, use greedy decoding (temp=0).
        perf_logger: Optional PerfLogger for per-frame timing.
    """
    if seed is not None and seed != -1:
        _seed_all(seed)

    # ----------------------------------------------------------------
    # 1) Load PersonaPlex MLX model
    # ----------------------------------------------------------------
    _log("info", f"loading PersonaPlex MLX model from {weights_path}")
    model = get_personaplex_lm(weights_path, dtype=mx.float16)
    _log("info", "model loaded")

    # ----------------------------------------------------------------
    # 2) Load rustymimi audio tokenizer
    # ----------------------------------------------------------------
    if mimi_weight is None:
        _log("info", f"downloading Mimi weights from {hf_repo}")
        mimi_weight = hf_hub_download(hf_repo, MIMI_NAME)
    _log("info", f"loading audio tokenizer from {mimi_weight}")
    mimi_codebooks = max(NUM_GENERATED_CODEBOOKS, NUM_OTHER_CODEBOOKS)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weight, num_codebooks=mimi_codebooks)
    _log("info", "audio tokenizer loaded")

    # ----------------------------------------------------------------
    # 3) Load sentencepiece text tokenizer
    # ----------------------------------------------------------------
    if tokenizer_path is None:
        _log("info", f"downloading text tokenizer from {hf_repo}")
        tokenizer_path = hf_hub_download(hf_repo, TEXT_TOKENIZER_NAME)
    _log("info", f"loading text tokenizer from {tokenizer_path}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    _log("info", "text tokenizer loaded")

    # ----------------------------------------------------------------
    # 4) Load input audio
    # ----------------------------------------------------------------
    _log("info", f"loading input audio from {input_wav}")
    in_pcms, in_sr = sphn.read(input_wav, sample_rate=SAMPLE_RATE)
    total_samples = in_pcms.shape[-1]
    steps = total_samples // FRAME_SIZE
    _log("info", f"input audio: {total_samples} samples, {steps} frames "
         f"({total_samples / SAMPLE_RATE:.2f}s)")

    # ----------------------------------------------------------------
    # 5) Create samplers and PersonaPlexLmGen
    # ----------------------------------------------------------------
    if greedy:
        text_sampler = Sampler(temp=0.0)
        audio_sampler = Sampler(temp=0.0)
        _log("info", "using greedy decoding (temp=0)")
    else:
        text_sampler = Sampler(top_k=topk_text, temp=temp_text)
        audio_sampler = Sampler(top_k=topk_audio, temp=temp_audio)
        _log("info", f"sampling: text temp={temp_text} top_k={topk_text}, "
             f"audio temp={temp_audio} top_k={topk_audio}")

    # Estimate text prompt length for max_steps calculation
    prompt_steps = 0
    if text_prompt:
        tagged = _wrap_system_tags(text_prompt)
        prompt_steps = len(text_tokenizer.encode(tagged))

    # Add extra steps for text prompt injection, delay compensation,
    # and safety margin
    max_steps = steps + prompt_steps + 20
    gen = PersonaPlexLmGen(
        model=model,
        max_steps=max_steps,
        text_sampler=text_sampler,
        audio_sampler=audio_sampler,
        batch_size=1,
    )

    # ----------------------------------------------------------------
    # 6) Warmup
    # ----------------------------------------------------------------
    warmup(gen, audio_tokenizer, num_steps=4)

    # Reset caches after warmup by re-creating the LmGen
    # (warmup consumed steps in the gen_sequence)
    gen = PersonaPlexLmGen(
        model=model,
        max_steps=max_steps,
        text_sampler=text_sampler,
        audio_sampler=audio_sampler,
        batch_size=1,
    )
    # Reset transformer and depformer KV caches
    for c in model.transformer_cache:
        c.reset()
    for c in model.depformer_cache:
        c.reset()

    # ----------------------------------------------------------------
    # 7) Text prompt injection (optional)
    # ----------------------------------------------------------------
    if text_prompt:
        tagged = _wrap_system_tags(text_prompt)
        prompt_ids = text_tokenizer.encode(tagged)
        _log("info", f"injecting text prompt: {len(prompt_ids)} tokens")
        gen.step_text_prompt(prompt_ids)
        mx.eval(gen.lm_gen.gen_sequence)
        _log("info", "text prompt injection complete")

    # ----------------------------------------------------------------
    # 8) Run inference frame-by-frame
    # ----------------------------------------------------------------
    generated_frames: List[np.ndarray] = []
    generated_text_tokens: List[str] = []

    _log("info", f"running inference: {steps} steps")
    start_time = time.time()

    for idx in range(steps):
        if perf_logger:
            perf_logger.frame_start()

        # 8a. Slice one frame of user PCM
        pcm_data = in_pcms[:, idx * FRAME_SIZE : (idx + 1) * FRAME_SIZE]
        # pcm_data: [channels, FRAME_SIZE] -> need [1, 1, FRAME_SIZE]
        pcm_input = pcm_data[0:1][np.newaxis, :, :]  # [1, 1, FRAME_SIZE]

        # 8b. Encode with rustymimi
        codes = audio_tokenizer.encode_step(pcm_input)
        # codes: [1, 1, num_codebooks] -> transpose to [1, num_codebooks, 1]
        codes_mx = mx.array(codes).transpose(0, 2, 1)[:, :NUM_OTHER_CODEBOOKS, :]

        if perf_logger:
            mx.eval(codes_mx)
            perf_logger.mark("mimi_encode")

        # 8c. Feed to PersonaPlexLmGen
        tokens = gen.step(codes_mx)

        if perf_logger and tokens is not None:
            mx.eval(tokens)
            perf_logger.mark("lm_forward")

        if tokens is None:
            if perf_logger:
                perf_logger.mark("lm_forward")
                perf_logger.frame_end(missed=False)
            continue

        # 8d. Decode agent audio with rustymimi
        # tokens: [B, dep_q+1, 1] -> agent audio is tokens[:, 1:, :]
        text_token_val = tokens[0, 0, 0].item()
        agent_audio = tokens[:, 1:, :]  # [1, dep_q, 1]

        agent_audio_np = np.array(agent_audio).astype(np.uint32)
        out_pcm = audio_tokenizer.decode_step(agent_audio_np)

        if perf_logger:
            perf_logger.mark("mimi_decode")

        # out_pcm: [1, 1, FRAME_SIZE]
        generated_frames.append(out_pcm[0, 0])

        # 8e. Decode text token
        if text_token_val not in (0, 3):
            try:
                _text = text_tokenizer.id_to_piece(int(text_token_val))
                _text = _text.replace("\u2581", " ")
                generated_text_tokens.append(_text)
            except Exception:
                generated_text_tokens.append(f"<unk:{text_token_val}>")
        else:
            token_map = ["EPAD", "BOS", "EOS", "PAD"]
            generated_text_tokens.append(token_map[int(text_token_val)])

        if perf_logger:
            perf_logger.frame_end()

        # Progress logging every 100 steps
        if (idx + 1) % 100 == 0 or idx == steps - 1:
            elapsed = time.time() - start_time
            tok_per_sec = (idx + 1) / elapsed if elapsed > 0 else 0
            _log("info", f"step {idx + 1}/{steps} "
                 f"({tok_per_sec:.1f} tok/s, "
                 f"{elapsed:.1f}s elapsed)")

    elapsed = time.time() - start_time
    _log("info", f"inference complete: {steps} steps in {elapsed:.1f}s "
         f"({steps / elapsed:.1f} tok/s)")

    # ----------------------------------------------------------------
    # 9) Write output WAV
    # ----------------------------------------------------------------
    if len(generated_frames) == 0:
        _log("error", "no audio frames generated — check input file and config")
        return

    output_pcm = np.concatenate(generated_frames, axis=-1)

    # Trim or pad to match input duration
    if output_pcm.shape[-1] > total_samples:
        output_pcm = output_pcm[:total_samples]
    elif output_pcm.shape[-1] < total_samples:
        pad_len = total_samples - output_pcm.shape[-1]
        output_pcm = np.concatenate(
            [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
        )

    # Ensure output directory exists
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    rustymimi.write_wav(output_wav, output_pcm, sample_rate=SAMPLE_RATE)
    _log("info", f"wrote output audio to {output_wav}")

    # ----------------------------------------------------------------
    # 10) Write text tokens to JSON
    # ----------------------------------------------------------------
    Path(output_text).parent.mkdir(parents=True, exist_ok=True)
    with open(output_text, "w") as f:
        json.dump(generated_text_tokens, f, ensure_ascii=False)
    _log("info", f"wrote output text to {output_text}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run offline inference."""
    parser = argparse.ArgumentParser(
        description="Offline PersonaPlex inference using MLX + rustymimi."
    )
    parser.add_argument(
        "--input-wav", required=True, type=str,
        help="Path to input WAV file (user audio).",
    )
    parser.add_argument(
        "--output-wav", required=True, type=str,
        help="Path to output WAV file (agent audio).",
    )
    parser.add_argument(
        "--output-text", required=True, type=str,
        help="Path to output JSON file (agent text tokens).",
    )
    parser.add_argument(
        "--weights", required=True, type=str,
        help="Path to PersonaPlex MLX safetensors weights.",
    )

    # Optional model assets
    parser.add_argument(
        "--mimi-weight", type=str, default=None,
        help="Path to Mimi tokenizer weights. Downloaded from HF if omitted.",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Path to sentencepiece text tokenizer. Downloaded from HF if omitted.",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=DEFAULT_HF_REPO,
        help=f"HuggingFace repo for asset downloads (default: {DEFAULT_HF_REPO}).",
    )

    # Prompt
    parser.add_argument(
        "--text-prompt", type=str,
        default="You are a wise and friendly teacher. Answer questions or "
                "provide advice in a clear and engaging way.",
        help="System text prompt for the model.",
    )

    # Sampling
    parser.add_argument(
        "--temp-audio", type=float, default=0.8,
        help="Audio sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--temp-text", type=float, default=0.7,
        help="Text sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--topk-audio", type=int, default=250,
        help="Audio top-k sampling (default: 250).",
    )
    parser.add_argument(
        "--topk-text", type=int, default=25,
        help="Text top-k sampling (default: 25).",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy decoding (temp=0).",
    )

    # Runtime
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="Random seed for reproducibility (-1 disables).",
    )
    parser.add_argument(
        "--perf-output", type=str, default=None,
        help="Path to write performance profiling JSON.",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Optional performance logging
    perf_logger = None
    if args.perf_output:
        perf_logger = PerfLogger(
            enabled=True,
            backend="mlx",
            device_name=None,  # auto-detect
        )

    run_inference(
        input_wav=args.input_wav,
        output_wav=args.output_wav,
        output_text=args.output_text,
        weights_path=args.weights,
        mimi_weight=args.mimi_weight,
        tokenizer_path=args.tokenizer,
        hf_repo=args.hf_repo,
        text_prompt=args.text_prompt,
        seed=args.seed if args.seed != -1 else None,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        topk_audio=args.topk_audio,
        topk_text=args.topk_text,
        greedy=args.greedy,
        perf_logger=perf_logger,
    )

    if perf_logger:
        perf_logger.export_json(args.perf_output)
        _log("info", f"performance data written to {args.perf_output}")


if __name__ == "__main__":
    main()
