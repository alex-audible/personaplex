#!/usr/bin/env python3
"""Micro-benchmark: per-frame server pipeline timing breakdown.

Measures each component of the real-time inference pipeline to identify
bottlenecks and optimization targets.

Usage:
    python assets/test/bench_server_frame.py
"""

import time
import numpy as np
import mlx.core as mx
import rustymimi

from moshi_mlx.utils.sampling import Sampler
from moshi.moshi_mlx.loaders_mlx import get_personaplex_lm
from moshi.moshi_mlx.lm_gen_mlx import PersonaPlexLmGen

SAMPLE_RATE = 24000
FRAME_SIZE = 1920
N_WARMUP = 10
N_BENCH = 100


def bench():
    print("=== Per-Frame Server Pipeline Benchmark ===\n")

    # --- Load model ---
    print("Loading model...")
    model = get_personaplex_lm("./personaplex_mlx.safetensors", dtype=mx.float16)

    text_sampler = Sampler(top_k=25, temp=0.7)
    audio_sampler = Sampler(top_k=250, temp=0.8)

    lm_gen = PersonaPlexLmGen(
        model=model,
        max_steps=3000,
        text_sampler=text_sampler,
        audio_sampler=audio_sampler,
        batch_size=1,
    )

    # --- Load mimi ---
    from huggingface_hub import hf_hub_download
    mimi_weight = hf_hub_download("nvidia/personaplex-7b-v1",
                                  "tokenizer-e351c8d8-checkpoint125.safetensors")
    mimi = rustymimi.Tokenizer(mimi_weight, num_codebooks=8)

    # --- Run system prompts (minimal) ---
    print("Running system prompts...")
    lm_gen.step_audio_silence(n_frames=6)
    mx.eval(lm_gen.lm_gen.gen_sequence)
    print(f"System prompts done. step_idx={lm_gen.step_idx}\n")

    # --- Warmup ---
    print(f"Warmup ({N_WARMUP} frames)...")
    mimi.reset()
    for _ in range(N_WARMUP):
        chunk = np.zeros(FRAME_SIZE, dtype=np.float32)
        chunk_input = chunk[np.newaxis, np.newaxis, :]
        codes = mimi.encode_step(chunk_input)
        codes_np = np.array(codes)
        codes_mx = mx.array(codes_np).transpose(0, 2, 1)[:, :8, :]
        tokens = lm_gen.step(codes_mx)
        if tokens is not None:
            mx.eval(tokens)
            agent_audio_np = np.array(tokens[:, 1:, :]).astype(np.uint32)
            _ = mimi.decode_step(agent_audio_np)
    print("Warmup done.\n")

    # --- Benchmark with fine-grained timing ---
    print(f"Benchmarking ({N_BENCH} frames)...\n")

    times = {
        "total": [],
        "mimi_encode": [],
        "array_convert": [],
        "lm_step": [],
        "mx_eval": [],
        "mimi_decode": [],
        "np_convert": [],
    }

    for i in range(N_BENCH):
        chunk = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.01

        t0 = time.perf_counter()

        # 1. Mimi encode
        chunk_input = chunk[np.newaxis, np.newaxis, :]
        codes = mimi.encode_step(chunk_input)
        t1 = time.perf_counter()

        # 2. Array conversion
        codes_np = np.array(codes)
        codes_mx = mx.array(codes_np).transpose(0, 2, 1)[:, :8, :]
        t2 = time.perf_counter()

        # 3. LM step (lazy - builds compute graph)
        tokens = lm_gen.step(codes_mx)
        t3 = time.perf_counter()

        if tokens is None:
            continue

        # 4. mx.eval (forces GPU compute)
        mx.eval(tokens)
        t4 = time.perf_counter()

        # 5. numpy conversion
        tokens_np = np.array(tokens)
        agent_audio = tokens[:, 1:, :]
        agent_audio_np = np.array(agent_audio).astype(np.uint32)
        t5 = time.perf_counter()

        # 6. Mimi decode
        main_pcm = mimi.decode_step(agent_audio_np)
        t6 = time.perf_counter()

        times["mimi_encode"].append((t1 - t0) * 1000)
        times["array_convert"].append((t2 - t1) * 1000)
        times["lm_step"].append((t3 - t2) * 1000)
        times["mx_eval"].append((t4 - t3) * 1000)
        times["np_convert"].append((t5 - t4) * 1000)
        times["mimi_decode"].append((t6 - t5) * 1000)
        times["total"].append((t6 - t0) * 1000)

    # --- Print results ---
    print(f"{'Component':<20} {'Mean (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'Min (ms)':>10}")
    print("-" * 65)
    for name in ["mimi_encode", "array_convert", "lm_step", "mx_eval",
                  "np_convert", "mimi_decode", "total"]:
        vals = np.array(times[name])
        if len(vals) == 0:
            print(f"{name:<20} {'N/A':>10}")
            continue
        print(f"{name:<20} {vals.mean():>10.2f} {np.percentile(vals, 50):>10.2f} "
              f"{np.percentile(vals, 95):>10.2f} {vals.min():>10.2f}")

    total = np.array(times["total"])
    print(f"\n{'Frame budget':>20}: 80.00ms")
    print(f"{'Actual mean':>20}: {total.mean():.2f}ms")
    print(f"{'Over budget by':>20}: {total.mean() - 80:.2f}ms")
    print(f"{'Effective rate':>20}: {1000 / total.mean():.1f} frames/sec (need 12.5)")

    # --- Breakdown as percentage ---
    print("\n--- Percentage Breakdown ---")
    for name in ["mimi_encode", "array_convert", "lm_step", "mx_eval",
                  "np_convert", "mimi_decode"]:
        vals = np.array(times[name])
        if len(vals) == 0:
            continue
        pct = vals.mean() / total.mean() * 100
        print(f"{name:<20} {pct:>6.1f}%  ({vals.mean():.2f}ms)")

    # --- Check: is lm_step actually lazy? ---
    print("\n--- Analysis ---")
    lm_mean = np.array(times["lm_step"]).mean()
    eval_mean = np.array(times["mx_eval"]).mean()
    if lm_mean > 5:
        print(f"WARNING: lm_step takes {lm_mean:.1f}ms — NOT fully lazy!")
        print("  This suggests eager evaluation inside PersonaPlexLmGen.step()")
    else:
        print(f"lm_step is lazy ({lm_mean:.1f}ms) — GPU work is in mx_eval ({eval_mean:.1f}ms)")

    if eval_mean > 70:
        print(f"GPU compute dominates ({eval_mean:.1f}ms). Optimizations needed:")
        print("  1. mx.compile on transformer forward pass")
        print("  2. Parallelize depformer slices")
        print("  3. Pre-allocate KV cache to avoid dynamic resizing")
    elif total.mean() > 80:
        overhead = total.mean() - eval_mean
        print(f"CPU overhead is {overhead:.1f}ms. Optimizations:")
        print("  1. Reduce mimi encode/decode overhead")
        print("  2. Reduce array conversion overhead")
        print("  3. Pre-allocate numpy buffers")


if __name__ == "__main__":
    bench()
