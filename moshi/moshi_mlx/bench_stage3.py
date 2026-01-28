#!/usr/bin/env python3
"""Stage 3 frame benchmark: voice prompt system overhead + per-frame latency.

Measures:
1. Voice prompt embedding replay overhead (one-time)
2. System prompt overhead (one-time)
3. Per-frame inference latency after system prompts (should match Stage 1: ~60ms)

This verifies that voice prompts don't add per-frame overhead during ongoing inference.

Usage:
    python -m moshi_mlx.bench_stage3 --weights personaplex_mlx.safetensors --voice-prompt NATF2.pt
"""
import argparse
import time
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download

from moshi_mlx.loaders_mlx import get_personaplex_lm
from moshi_mlx.lm_gen_mlx import PersonaPlexLmGen, SINE_TOKENS
from moshi_mlx.utils.sampling import Sampler


def _get_voice_prompt_dir():
    """Download voices.tgz from HF hub if not cached, return voices/ directory."""
    try:
        voices_archive = hf_hub_download(
            repo_id="nvidia/personaplex-7b-v1",
            filename="voices.tgz",
            repo_type="model",
        )
        # HF caches to: ~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/<hash>/voices.tgz
        # Extract to: ~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/<hash>/voices/
        voices_dir = Path(voices_archive).parent / "voices"
        if not voices_dir.exists():
            import tarfile
            with tarfile.open(voices_archive, "r:gz") as tar:
                tar.extractall(path=voices_dir.parent)
        return voices_dir
    except Exception as e:
        print(f"[WARNING] Failed to download voices.tgz: {e}", file=sys.stderr)
        return None


def bench_stage3(
    weights_path: str,
    voice_prompt_path: str,
    n_warmup: int = 5,
    n_steps: int = 50,
):
    print(f"Loading PersonaPlex MLX model from {weights_path}...")
    t0 = time.time()
    model = get_personaplex_lm(weights_path)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    cfg = model.cfg
    other_codebooks = cfg.other_codebooks
    print(f"Config: d_model={cfg.transformer.d_model}, layers={cfg.transformer.num_layers}, "
          f"heads={cfg.transformer.num_heads}, depformer_slices={cfg.depformer.num_slices}, "
          f"other_codebooks={other_codebooks}")

    # Create LmGen wrapper
    print(f"\nCreating PersonaPlexLmGen wrapper...")
    text_sampler = Sampler(temp=0.0)
    audio_sampler = Sampler(temp=0.0)
    lm_gen = PersonaPlexLmGen(
        model=model,
        max_steps=1024,  # Large enough for benchmark
        text_sampler=text_sampler,
        audio_sampler=audio_sampler,
    )

    # Resolve voice prompt path
    if not Path(voice_prompt_path).exists():
        # Try HF voices directory
        voices_dir = _get_voice_prompt_dir()
        if voices_dir is not None:
            candidate = voices_dir / voice_prompt_path
            if candidate.exists():
                voice_prompt_path = str(candidate)
            else:
                print(f"[ERROR] Voice prompt not found: {voice_prompt_path}", file=sys.stderr)
                sys.exit(1)

    print(f"Voice prompt: {voice_prompt_path}")

    # Load voice prompt embeddings
    print(f"\nLoading voice prompt embeddings...")
    t_vp_start = time.perf_counter()
    lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    t_vp_end = time.perf_counter()
    vp_load_time = (t_vp_end - t_vp_start) * 1000
    print(f"Voice prompt loaded in {vp_load_time:.1f}ms")

    # Run system prompts
    text_prompt = "You are a helpful assistant."
    text_token_ids = [1, 2, 3, 4, 5]  # Dummy text tokens

    print(f"\nRunning system prompt pipeline (voice + silence + text + silence)...")
    t_sysprompt_start = time.perf_counter()
    lm_gen.step_system_prompts(
        audio_tokenizer=None,  # No audio tokenizer needed for embedding-based prompts
        text_token_ids=text_token_ids,
    )
    t_sysprompt_end = time.perf_counter()
    sysprompt_time = (t_sysprompt_end - t_sysprompt_start) * 1000
    print(f"System prompts complete in {sysprompt_time:.1f}ms (step_idx={lm_gen.step_idx})")

    # Create dummy user audio tokens (SINE_TOKENS) for inference benchmarking
    # Shape: [B, 8] where B=1, 8 codebooks
    user_audio = mx.array([SINE_TOKENS])  # [1, 8]

    # Warmup inference frames
    print(f"\nWarming up inference frames ({n_warmup} steps)...")
    for i in range(n_warmup):
        lm_gen.step(user_audio)

    # Benchmark inference frames
    print(f"\nBenchmarking inference frames ({n_steps} steps)...")
    times_frame = []
    for i in range(n_steps):
        t_start = time.perf_counter()
        lm_gen.step(user_audio)
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000
        times_frame.append(elapsed_ms)

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

    frame_stats = stats(times_frame)

    print("\n" + "=" * 60)
    print("  STAGE 3 VOICE PROMPT BENCHMARK")
    print("=" * 60)
    print(f"  Model:           PersonaPlex 7B (MLX)")
    print(f"  Voice prompt:    {Path(voice_prompt_path).name}")
    print(f"  Steps:           {n_steps}")
    print(f"  Frame budget:    80.0 ms")
    print(f"  Stage 1 baseline: 59.7 ms (full sample)")
    print("-" * 60)
    print(f"  ONE-TIME OVERHEAD:")
    print(f"    Voice prompt load:    {vp_load_time:.1f} ms")
    print(f"    System prompts:       {sysprompt_time:.1f} ms")
    print(f"    Total setup:          {vp_load_time + sysprompt_time:.1f} ms")
    print("-" * 60)
    print(f"  PER-FRAME INFERENCE (after system prompts):")
    print(f"    Mean:          {frame_stats['mean']:.1f} ms")
    print(f"    P50:           {frame_stats['p50']:.1f} ms")
    print(f"    P95:           {frame_stats['p95']:.1f} ms")
    print(f"    Min:           {frame_stats['min']:.1f} ms")
    print(f"    Max:           {frame_stats['max']:.1f} ms")
    within_budget = frame_stats['mean'] < 80.0
    print(f"    Within 80ms:   {'YES' if within_budget else 'NO'}")

    # Compare to Stage 1 baseline
    stage1_baseline = 59.7
    delta = frame_stats['mean'] - stage1_baseline
    delta_pct = (delta / stage1_baseline) * 100
    print(f"    vs Stage 1:    {delta:+.1f}ms ({delta_pct:+.1f}%)")
    print("=" * 60)

    return {
        "vp_load_time": vp_load_time,
        "sysprompt_time": sysprompt_time,
        "frame_stats": frame_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 3 voice prompt benchmark")
    parser.add_argument("--weights", default="personaplex_mlx.safetensors",
                        help="Path to converted MLX weights")
    parser.add_argument("--voice-prompt", default="NATF2.pt",
                        help="Voice prompt .pt file (name or path)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    bench_stage3(args.weights, args.voice_prompt, args.warmup, args.steps)


if __name__ == "__main__":
    main()
