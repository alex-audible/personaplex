# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Benchmark CLI for PersonaPlex inference performance profiling.

Wraps the offline inference pipeline with PerfLogger instrumentation to
capture per-frame timing baselines. Supports two modes:

1. **Run mode** -- execute instrumented inference and export perf JSON::

    python -m moshi_mlx.benchmark \\
        --voice-prompt "NATF2.pt" \\
        --input-wav "assets/test/input_assistant.wav" \\
        --seed 42424242 \\
        --device mps \\
        --output-wav "baseline_assistant_mps.wav" \\
        --output-text "baseline_assistant_mps.json" \\
        --perf-output "baseline_assistant_mps_perf.json"

2. **Compare mode** -- load two perf JSONs and print a side-by-side table::

    python -m moshi_mlx.benchmark --compare baseline_mps.json mlx_stage2.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch

from moshi_mlx.perf_logger import PerfLogger, compare_summaries, load_perf_json


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Mirrors the arguments of ``moshi.offline.main`` with the addition of
    ``--perf-output`` and ``--compare``.
    """
    parser = argparse.ArgumentParser(
        prog="moshi_mlx.benchmark",
        description=(
            "PersonaPlex inference benchmark. Run instrumented offline "
            "inference with per-frame timing, or compare two perf JSON files."
        ),
    )

    # --- Compare mode (exclusive) ---
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help=(
            "Compare mode: provide two perf JSON files and print a "
            "side-by-side summary table. All other arguments are ignored."
        ),
    )

    # --- Benchmark output ---
    parser.add_argument(
        "--perf-output",
        type=str,
        required=False,
        help="Path to write performance profiling JSON (required in run mode).",
    )

    # --- Inference arguments (same as moshi.offline) ---
    parser.add_argument(
        "--input-wav", type=str, help="Path to input WAV file (user audio)"
    )
    parser.add_argument(
        "--output-wav", type=str, help="Path to output WAV file of agent audio to write"
    )
    parser.add_argument(
        "--output-text", type=str, help="Path to output JSON file of agent text to write"
    )
    parser.add_argument(
        "--text-prompt",
        default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
        type=str,
        help="Text prompt",
    )
    parser.add_argument(
        "--voice-prompt", type=str,
        help="Voice prompt filename (basename) inside --voice-prompt-dir (e.g. 'NATM1.pt').",
    )
    parser.add_argument(
        "--voice-prompt-dir", type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
        ),
    )

    # Model assets
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")

    # Import DEFAULT_REPO lazily to avoid pulling in torch at parse time for --compare mode
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HF repo to look into (defaults to pre-trained model repo)",
    )

    # Runtime / sampling controls
    parser.add_argument("--temp-audio", type=float, default=0.8, help="Audio sampling temperature (default: 0.8)")
    parser.add_argument("--temp-text", type=float, default=0.7, help="Text sampling temperature (default: 0.7)")
    parser.add_argument("--topk-audio", type=int, default=250, help="Audio top-k sampling (default: 250)")
    parser.add_argument("--topk-text", type=int, default=25, help="Text top-k sampling (default: 25)")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling (greedy decoding)")
    parser.add_argument("--device", type=str, default="mps", help="Device on which to run, defaults to 'mps'.")
    parser.add_argument(
        "--cpu-offload", action="store_true",
        help="Offload LM model layers to CPU when GPU memory is insufficient.",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Seed for reproducibility (-1 disables)")

    return parser


def _run_compare(file_a: str, file_b: str) -> None:
    """Load two perf JSON files and print a comparison table to stderr."""
    data_a = load_perf_json(file_a)
    data_b = load_perf_json(file_b)

    summary_a = data_a.get("summary", data_a)
    summary_b = data_b.get("summary", data_b)

    label_a = os.path.basename(file_a)
    label_b = os.path.basename(file_b)

    table = compare_summaries(summary_a, summary_b, label_a, label_b)
    print(table, file=sys.stderr)


def _run_benchmark(args: argparse.Namespace) -> None:
    """Execute instrumented offline inference and export performance data."""
    # Validate required args for run mode
    missing = []
    for required in ("input_wav", "output_wav", "output_text", "voice_prompt", "perf_output"):
        if getattr(args, required, None) is None:
            missing.append(f"--{required.replace('_', '-')}")
    if missing:
        print(
            f"Error: the following arguments are required in run mode: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Lazy imports to keep --compare fast
    from moshi.models import loaders
    from moshi.offline import run_inference, _get_voice_prompt_dir, log

    hf_repo = args.hf_repo or loaders.DEFAULT_REPO

    # Resolve voice prompt path
    voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, hf_repo)
    if not os.path.exists(voice_prompt_dir):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir}")

    voice_prompt_path = os.path.join(voice_prompt_dir, args.voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(
            f"Voice prompt '{args.voice_prompt}' not found in "
            f"'{voice_prompt_dir}' (resolved: {voice_prompt_path})"
        )

    greedy = bool(args.greedy)

    # Create PerfLogger
    perf_logger = PerfLogger(
        enabled=True,
        backend=args.device,
        device_name=None,  # auto-detect
    )

    print(f"[benchmark] Starting instrumented inference on device={args.device}", file=sys.stderr)
    print(f"[benchmark] Input: {args.input_wav}", file=sys.stderr)
    print(f"[benchmark] Perf output: {args.perf_output}", file=sys.stderr)

    with torch.no_grad():
        run_inference(
            input_wav=args.input_wav,
            output_wav=args.output_wav,
            output_text=args.output_text,
            text_prompt=args.text_prompt,
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            hf_repo=hf_repo,
            device=args.device,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=greedy,
            save_voice_prompt_embeddings=False,
            cpu_offload=args.cpu_offload,
            perf_logger=perf_logger,
        )

    # Export full perf data
    perf_logger.export_json(args.perf_output)

    # Print summary to stderr
    summary = perf_logger.summary()
    print("\n" + "=" * 60, file=sys.stderr)
    print("  PERFORMANCE SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Device:          {summary.get('device', 'unknown')}", file=sys.stderr)
    print(f"  Backend:         {summary.get('backend', 'unknown')}", file=sys.stderr)
    print(f"  Total frames:    {summary.get('total_frames', 0)}", file=sys.stderr)
    print(f"  Missed frames:   {summary.get('missed_frames', 0)} ({summary.get('missed_pct', 0)}%)", file=sys.stderr)
    print(f"  Avg frame:       {summary.get('avg_frame_ms', 0):.1f} ms", file=sys.stderr)
    print(f"  P50 frame:       {summary.get('p50_frame_ms', 0):.1f} ms", file=sys.stderr)
    print(f"  P95 frame:       {summary.get('p95_frame_ms', 0):.1f} ms", file=sys.stderr)
    print(f"  P99 frame:       {summary.get('p99_frame_ms', 0):.1f} ms", file=sys.stderr)
    print(f"  Max frame:       {summary.get('max_frame_ms', 0):.1f} ms", file=sys.stderr)
    print(f"  Min frame:       {summary.get('min_frame_ms', 0):.1f} ms", file=sys.stderr)

    component_keys = ["avg_mimi_encode_ms", "avg_lm_forward_ms", "avg_depformer_ms", "avg_mimi_decode_ms"]
    for key in component_keys:
        if key in summary:
            label = key.replace("avg_", "").replace("_ms", "")
            print(f"  Avg {label:<14s} {summary[key]:.1f} ms", file=sys.stderr)

    if "peak_memory_mb" in summary:
        print(f"  Peak memory:     {summary['peak_memory_mb']:.0f} MB", file=sys.stderr)

    print("=" * 60, file=sys.stderr)
    print(f"\n[benchmark] Full perf data written to {args.perf_output}", file=sys.stderr)


def main() -> None:
    """Entry point for ``python -m moshi_mlx.benchmark``."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.compare:
        _run_compare(args.compare[0], args.compare[1])
    else:
        _run_benchmark(args)


if __name__ == "__main__":
    main()
