#!/usr/bin/env python3
"""Quick MLX transformer forward pass benchmark.

Measures per-step latency of the PersonaPlex LM on MLX,
comparable to the ~266ms lm_forward measured on PyTorch MPS.

Usage:
    python -m moshi_mlx.bench_forward --weights personaplex_mlx.safetensors
"""
import argparse
import time
import sys

import mlx.core as mx

from moshi_mlx.models.lm import Lm
from moshi_mlx.utils.sampling import Sampler
from moshi_mlx.loaders_mlx import personaplex_config, get_personaplex_lm


def bench_forward(weights_path: str, n_warmup: int = 5, n_steps: int = 50):
    print(f"Loading PersonaPlex MLX model from {weights_path}...")
    t0 = time.time()
    model = get_personaplex_lm(weights_path)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    cfg = model.cfg
    other_codebooks = cfg.other_codebooks  # 16 - 8 = 8
    print(f"Config: d_model={cfg.transformer.d_model}, layers={cfg.transformer.num_layers}, "
          f"heads={cfg.transformer.num_heads}, depformer_slices={cfg.depformer.num_slices}, "
          f"other_codebooks={other_codebooks}")

    text_sampler = Sampler()
    audio_sampler = Sampler()

    # Dummy inputs matching streaming inference (batch=1, seq=1)
    text_token = mx.array([[cfg.text_out_vocab_size]])  # [1, 1]
    audio_tokens = [mx.array([[0]])] * other_codebooks  # list of 8 x [1, 1]

    # --- Warmup ---
    print(f"\nWarming up ({n_warmup} steps)...")
    for i in range(n_warmup):
        text_out, audio_out = model.sample(
            text_token, audio_tokens, text_sampler, audio_sampler
        )
        mx.eval(text_out)
        if audio_out is not None:
            mx.eval(audio_out)
    # Reset KV caches after warmup
    for c in model.transformer_cache:
        c.reset()

    # --- Benchmark: Full sample (transformer + depformer) ---
    print(f"\nBenchmarking full sample (transformer + depformer), {n_steps} steps...")
    times_full = []
    for i in range(n_steps):
        t_start = time.perf_counter()
        text_out, audio_out = model.sample(
            text_token, audio_tokens, text_sampler, audio_sampler
        )
        mx.eval(text_out)
        if audio_out is not None:
            mx.eval(audio_out)
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000
        times_full.append(elapsed_ms)

    # Reset caches
    for c in model.transformer_cache:
        c.reset()

    # --- Benchmark: Transformer only (no depformer) ---
    print(f"Benchmarking transformer only, {n_steps} steps...")
    times_transformer = []
    for i in range(n_steps):
        t_start = time.perf_counter()
        text_logits = model(text_token)
        mx.eval(text_logits)
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000
        times_transformer.append(elapsed_ms)

    # Reset caches
    for c in model.transformer_cache:
        c.reset()

    # --- Results ---
    def stats(times):
        times_sorted = sorted(times)
        n = len(times_sorted)
        return {
            "mean": sum(times) / n,
            "p50": times_sorted[n // 2],
            "p95": times_sorted[int(n * 0.95)],
            "p99": times_sorted[int(n * 0.99)],
            "min": times_sorted[0],
            "max": times_sorted[-1],
        }

    full_stats = stats(times_full)
    trans_stats = stats(times_transformer)

    print("\n" + "=" * 60)
    print("  MLX FORWARD PASS BENCHMARK")
    print("=" * 60)
    print(f"  Model:           PersonaPlex 7B (MLX)")
    print(f"  Steps:           {n_steps}")
    print(f"  Frame budget:    80.0 ms")
    print(f"  MPS baseline:    266.0 ms (lm_forward only)")
    print(f"                   310.2 ms (full frame)")
    print("-" * 60)
    print(f"  TRANSFORMER ONLY (comparable to MPS lm_forward=266ms):")
    print(f"    Mean:          {trans_stats['mean']:.1f} ms")
    print(f"    P50:           {trans_stats['p50']:.1f} ms")
    print(f"    P95:           {trans_stats['p95']:.1f} ms")
    print(f"    Min:           {trans_stats['min']:.1f} ms")
    print(f"    Max:           {trans_stats['max']:.1f} ms")
    speedup_trans = 266.0 / trans_stats['mean']
    print(f"    Speedup vs MPS: {speedup_trans:.1f}x")
    print("-" * 60)
    print(f"  FULL SAMPLE (transformer + depformer + sampling):")
    print(f"    Mean:          {full_stats['mean']:.1f} ms")
    print(f"    P50:           {full_stats['p50']:.1f} ms")
    print(f"    P95:           {full_stats['p95']:.1f} ms")
    print(f"    Min:           {full_stats['min']:.1f} ms")
    print(f"    Max:           {full_stats['max']:.1f} ms")
    speedup_full = 310.2 / full_stats['mean']
    print(f"    Speedup vs MPS: {speedup_full:.1f}x (vs full frame 310ms)")
    within_budget = full_stats['mean'] < 80.0
    print(f"    Within 80ms:   {'YES' if within_budget else 'NO'}")
    print("=" * 60)

    return full_stats, trans_stats


def main():
    parser = argparse.ArgumentParser(description="MLX forward pass benchmark")
    parser.add_argument("--weights", default="personaplex_mlx.safetensors",
                        help="Path to converted MLX weights")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    bench_forward(args.weights, args.warmup, args.steps)


if __name__ == "__main__":
    main()
