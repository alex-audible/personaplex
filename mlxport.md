# PersonaPlex MLX Port: Comprehensive Migration Plan

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Existing MLX Moshi Port Analysis](#3-existing-mlx-moshi-port-analysis)
4. [Staged Port Approach](#4-staged-port-approach)
5. [Performance Logging Framework](#5-performance-logging-framework)
6. [MLX Architecture Mapping](#6-mlx-architecture-mapping)
7. [Weight Conversion](#7-weight-conversion)
8. [Streaming State Management](#8-streaming-state-management)
9. [Server Integration](#9-server-integration)
10. [Risk Assessment](#10-risk-assessment)
11. [File-by-File Port Reference](#11-file-by-file-port-reference)
12. [Timeline and Milestones](#12-timeline-and-milestones)

---

## 1. Executive Summary

### Problem

PersonaPlex is a 7B-parameter real-time speech-to-speech model designed for NVIDIA CUDA.
On Apple Silicon MPS (M3, 96GB unified memory), it runs approximately 5x slower than
real-time: each 80ms audio frame takes ~400ms+ to process, causing 80% audio dropout
and 10-second latency. This is caused by 12 compounding bottlenecks documented in
`optimisation_ultrathink.md`, primarily: CUDA Graphs disabled, `torch.compile` disabled,
no Flash Attention, bfloat16 emulation overhead, and CPU synchronization stalls.

### Solution

Port the inference pipeline from PyTorch/MPS to Apple's MLX framework, which provides:
- Native Metal GPU acceleration (no PyTorch MPS translation layer)
- Unified memory architecture awareness (zero-copy CPU/GPU)
- Lazy evaluation with automatic operation fusion (replaces CUDA Graphs)
- `mx.compile` for graph-level optimization (replaces `torch.compile`)
- `mx.fast.scaled_dot_product_attention` with causal mask string (replaces Flash Attention)
- Native float16 and bfloat16 support on Apple Silicon
- Built-in `nn.RoPE` for rotary embeddings

### Key Advantage: Existing Kyutai MLX Port

Kyutai already ships `moshi_mlx` (v0.3.0 on PyPI), a complete MLX implementation of the
base Moshi model at `github.com/kyutai-labs/moshi/tree/main/moshi_mlx`. This includes:
- `moshi_mlx/models/lm.py` -- LM with full DepFormer support
- `moshi_mlx/models/generate.py` -- LMGen streaming generation
- `moshi_mlx/modules/transformer.py` -- Streaming transformer with KV cache
- `moshi_mlx/modules/conv.py` -- Streaming causal convolutions
- `moshi_mlx/modules/seanet.py` -- SEANet encoder/decoder
- `moshi_mlx/modules/kv_cache.py` -- Standard and RotatingKVCache
- `moshi_mlx/modules/quantization.py` -- Vector quantization

The strategy is NOT to write a port from scratch, but to adapt the existing `moshi_mlx`
codebase to support PersonaPlex-specific features, then integrate it with the PersonaPlex
server and streaming infrastructure.

### Target Performance

- Per-frame latency: <80ms (12.5 Hz frame rate)
- End-to-end latency: <200ms
- Zero missed audio frames during 15-second conversations
- Memory: <20GB for full model in float16 (fits in 96GB M3 with ample headroom)

---

## 2. Architecture Overview

### PersonaPlex Model Components

```
Audio In (24kHz) --> [Opus Decode] --> [Mimi Encode] --> [LM (7B)] --> [Mimi Decode] --> [Opus Encode] --> Audio Out
                         CPU              GPU               GPU              GPU              CPU
```

**Main LM (7B parameters):**
- `LMModel` in `moshi/moshi/models/lm.py`
- Dimension: 4096, Heads: 32, Layers: 32, Hidden scale: 4.125
- RoPE positional embeddings, SiLU gating, RMS norm (float32 promoted)
- Context window: 3000 tokens
- Input: 17 codebook streams (1 text + 8 moshi audio + 8 user audio)
- Delays: `[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]`

**DepFormer (depth transformer):**
- Dimension: 1024, Heads: 16, Layers: 6
- SiLU gating, no positional embedding, weights-per-step
- `depformer_multi_linear=True` -- one linear per codebook
- Context: 8 tokens
- Generates 8 audio codebook tokens sequentially per frame
- dep_q configured as 16 for weight loading, 8 used during inference

**Mimi Codec (x2 instances):**
- SEANet encoder: causal Conv1d layers, ratios [8,6,5,4], kernel_size=7
- 8-layer transformer encoder (d=512, 8 heads, context=250)
- Residual VQ: 8 codebooks, 2048 bins, dim=256
- SEANet decoder: ConvTranspose1d layers (mirror of encoder)
- 8-layer transformer decoder
- Sample rate: 24kHz, Frame rate: 12.5Hz, Frame size: 1920 samples

**Streaming Pipeline (`server.py` opus_loop):**
```
For each 1920-sample audio frame (80ms):
  1. mimi.encode(chunk)           -- Mimi encode (user audio)
  2. other_mimi.encode(chunk)     -- Second Mimi encode (discarded)
  3. For each code step:
     a. lm_gen.step(codes)        -- Main 7B LM forward + DepFormer 8-step
  4. mimi.decode(tokens)          -- Mimi decode (agent audio)
  5. other_mimi.decode(tokens)    -- Second Mimi decode (discarded)
```

### PersonaPlex-Specific Features (vs. base Moshi)

1. **Voice prompting system**: `LMGen.load_voice_prompt()`, `step_embeddings()`,
   `_step_voice_prompt_core()` -- pre-computes voice prompt embeddings and caches them
2. **Text prompting**: `text_prompt_tokens` injected during system prompt phase
3. **Dual Mimi instances**: `mimi` + `other_mimi` for parallel audio stream state
4. **Weight copying (dep_q 8->16)**: `loaders.py` patches weights by copying codebooks
   0..7 to 8..15 for depformer_in, depformer_emb, linears, gating
5. **DepFormer self_attn expansion**: Weights are expanded by concatenation if shapes
   differ between checkpoint and model
6. **Loss reporting**: Optional per-step loss/rank computation for debugging
7. **Save voice prompt embeddings**: Optional `.pt` cache of voice prompt embeddings

---

## 3. Existing MLX Moshi Port Analysis

### What `moshi_mlx` Already Provides

The Kyutai `moshi_mlx` package (v0.3.0, tested on M3) includes:

| Component | MLX File | Status |
|-----------|----------|--------|
| Transformer with KV cache | `modules/transformer.py` | Complete |
| Attention (SDPA + RoPE) | `modules/transformer.py` | Complete, uses `mx.fast.scaled_dot_product_attention` |
| KV Cache (standard + rotating) | `modules/kv_cache.py` | Complete |
| Streaming Conv1d | `modules/conv.py` | Complete, handles causal padding + streaming state |
| SEANet encoder/decoder | `modules/seanet.py` | Complete |
| Vector quantization | `modules/quantization.py` | Complete |
| LM model with DepFormer | `models/lm.py` | Complete |
| LMGen streaming wrapper | `models/generate.py` | Complete |
| Mimi codec | `models/mimi.py` | Complete, uses `rustymimi` for Rust backend |
| Local CLI | `local.py` | Complete |
| Web server | `local_web.py` | Complete |

### What `moshi_mlx` Does NOT Have (PersonaPlex-specific)

1. **Voice prompt system** -- The voice prompt loading, encoding, caching, and embedding
   injection pipeline from PersonaPlex's `LMGen`
2. **Dual Mimi instances** -- The `other_mimi` pattern
3. **Weight patching (dep_q expansion)** -- The copy 0..7->8..15 logic from `loaders.py`
4. **PersonaPlex weight format** -- The specific `nvidia/personaplex-7b-v1` safetensors
5. **Text prompt injection** -- The system prompt tag wrapping and token injection
6. **Loss reporting** -- The optional `report_loss` mode
7. **Specific delays configuration** -- PersonaPlex uses 17-element delay vector
8. **aiohttp WebSocket server** -- `moshi_mlx` uses its own simpler server

### Architecture Differences to Reconcile

| Feature | Kyutai moshi_mlx | PersonaPlex PyTorch |
|---------|------------------|---------------------|
| Audio codec | `rustymimi` (Rust) | Pure PyTorch Mimi |
| LMGen state | MLX arrays, step_idx counter | PyTorch tensors, offset counter, CUDAGraphed |
| Streaming | `step()` with delay management | `step()` with cache/provided tensor pattern |
| Server | Simpler local server | aiohttp WebSocket with opus I/O |
| dep_q | 8 (standard Moshi) | 16 weight-init, 8 inference |
| n_q | 8 | 16 (8 user + 8 agent) |
| Voice prompts | Not supported | Full voice prompt pipeline |
| Sampling | MLX sampling module | PyTorch `sample_token()` |

### Critical Decision: Use rustymimi vs Pure MLX Mimi

The `moshi_mlx` port uses `rustymimi` -- a Rust implementation of the Mimi codec with
Python bindings. This is highly optimized and handles streaming natively. However,
PersonaPlex has customizations in the pure-PyTorch Mimi (dual instances, specific
streaming state management for voice prompts).

**Recommendation**: Start with `rustymimi` for the Mimi codec, as it is the most
battle-tested path. Only port to pure MLX Mimi if `rustymimi` proves incompatible
with PersonaPlex voice prompting requirements. The `rustymimi` approach also avoids
porting the entire SEANet + transformer codec to MLX, saving significant effort.

---

## 4. Staged Port Approach

### Stage 0: Performance Baseline (PREREQUISITE -- before any MLX work)

**Goal**: Build the PerfLogger, instrument the existing PyTorch offline inference
pipeline, and capture reproducible MPS baseline metrics using the repo's test assets.
This baseline is the reference point for all subsequent MLX work.

**Rationale**: Without precise per-component timings on the current MPS backend, we
cannot measure the impact of the MLX port. The baseline must be captured first, using
deterministic offline mode (fixed seed), so every future stage can be compared
apples-to-apples.

**Tasks**:

1. Create directory `moshi/moshi_mlx/`
2. Build `moshi/moshi_mlx/perf_logger.py` -- the performance logging framework
   (see Section 5 for full API design)
3. Build `moshi/moshi_mlx/benchmark.py` -- a harness that:
   a. Wraps `moshi.offline.run_inference()` with PerfLogger instrumentation
   b. Injects `perf.frame_start()` / `perf.mark()` / `perf.frame_end()` around
      each pipeline stage (Mimi encode, LM forward, depformer, Mimi decode)
   c. Uses `torch.mps.synchronize()` before each timing point to ensure accurate
      measurement on MPS (prevents measuring queued-but-not-executed work)
   d. Exports results as JSON for later comparison
4. Run the **assistant baseline** on MPS using the repo's existing test assets:
   ```bash
   python -m moshi.offline \
     --voice-prompt "NATF2.pt" \
     --input-wav "assets/test/input_assistant.wav" \
     --seed 42424242 \
     --output-wav "baseline_assistant_mps.wav" \
     --output-text "baseline_assistant_mps.json"
   ```
   With PerfLogger enabled, capturing per-frame timings to
   `baseline_assistant_mps_perf.json`.
5. Run the **service baseline** on MPS:
   ```bash
   python -m moshi.offline \
     --voice-prompt "NATM1.pt" \
     --text-prompt "$(cat assets/test/prompt_service.txt)" \
     --input-wav "assets/test/input_service.wav" \
     --seed 42424242 \
     --output-wav "baseline_service_mps.wav" \
     --output-text "baseline_service_mps.json"
   ```
   With PerfLogger enabled, capturing to `baseline_service_mps_perf.json`.
6. Verify baseline outputs are deterministic (run twice with same seed, compare)
7. Save baseline output WAVs and text JSONs as reference artifacts for numerical
   parity testing in Stage 2
8. Install MLX dependencies: `pip install mlx mlx-lm rustymimi moshi_mlx`
9. Verify that `moshi_mlx` can load and run the base Moshi model on M3

**Baseline metrics to capture** (per test case):

| Metric | Description |
|--------|-------------|
| `avg_frame_ms` | Mean per-frame total latency |
| `p50/p95/p99_frame_ms` | Latency percentiles |
| `avg_mimi_encode_ms` | Mean Mimi encode time per frame |
| `avg_lm_forward_ms` | Mean main transformer forward time |
| `avg_depformer_ms` | Mean depformer (8 codebook steps) time |
| `avg_mimi_decode_ms` | Mean Mimi decode time per frame |
| `total_frames` | Total frames processed |
| `missed_frames` | Frames exceeding 80ms budget |
| `peak_memory_mb` | Peak MPS memory usage |

**Files to create**:
- `moshi/moshi_mlx/__init__.py`
- `moshi/moshi_mlx/perf_logger.py` -- Performance logging framework
- `moshi/moshi_mlx/benchmark.py` -- Benchmark harness wrapping offline inference

**Test assets used** (already in repo):
- `assets/test/input_assistant.wav` -- User audio for assistant scenario
- `assets/test/input_service.wav` -- User audio for service scenario
- `assets/test/prompt_service.txt` -- Text prompt for service scenario
- Voice prompts `NATF2.pt`, `NATM1.pt` -- Downloaded from HuggingFace `voices/` dir

**Exit criteria**:
- PerfLogger captures per-component timings for both test scenarios
- Baseline JSON files saved with reproducible metrics
- Baseline output WAVs saved as reference for parity testing
- `moshi_mlx` confirmed working on M3 with base Moshi model

---

### Stage 1: Weight Conversion Pipeline (Days 2-3)

**Goal**: Convert PersonaPlex safetensors weights to MLX format with dep_q expansion.

**Dependency**: Stage 0.

**Tasks**:

1. Write `moshi/moshi_mlx/convert_weights.py` that:
   a. Loads `model.safetensors` from `nvidia/personaplex-7b-v1`
   b. Applies the dep_q 0..7->8..15 weight copy (mirroring `loaders.py:232-249`)
   c. Applies the depformer self_attn expansion (mirroring `loaders.py:220-229`)
   d. Converts dtype from bfloat16 to float16 (or keeps bfloat16 -- MLX supports both)
   e. Saves in MLX-compatible format (safetensors or `.npz`)
2. Write `moshi/moshi_mlx/loaders_mlx.py` that loads the converted weights into the
   `moshi_mlx` LM model architecture
3. Verify weight loading by comparing model parameter counts and shapes

**Key implementation detail**: MLX can load safetensors directly via `mx.load()`. The
conversion pipeline should:
```
PyTorch safetensors --> load with safetensors.torch --> apply patches --> save as
MLX safetensors (with mx.save_safetensors) or .npz
```

Alternatively, use `mlx_lm.convert` with `--dtype float16` if the model architecture
matches a supported format, but PersonaPlex's custom architecture likely requires a
custom conversion script.

**Files to create**:
- `moshi/moshi_mlx/convert_weights.py`
- `moshi/moshi_mlx/loaders_mlx.py`

**Exit criteria**: Weights load into MLX model; parameter shapes match PyTorch model.

---

### Stage 2: Offline Inference Parity (Days 4-7)

**Goal**: Run PersonaPlex offline inference entirely in MLX, producing identical output
to the PyTorch version (greedy decoding).

**Dependency**: Stage 1.

**Tasks**:

1. Adapt the `moshi_mlx` LM model to PersonaPlex's configuration:
   a. Set n_q=16, dep_q=8 (inference), delays as in `loaders.py:121`
   b. Configure depformer_multi_linear=True, depformer_weights_per_step=True
   c. Set context=3000, depformer_context=8
   d. Ensure hidden_scale=4.125 matches the feedforward dimensions

2. Create `moshi/moshi_mlx/lm_gen_mlx.py` -- the PersonaPlex LMGen for MLX:
   a. Port the `prepare_step_input()` cache/provided tensor pattern
   b. Port the `step()` method with the graphed_main/graphed_depth flow
   c. Port `process_transformer_output()` -- text sampling + depformer sequential loop
   d. Port the delay-based output collection (`delays_cuda`, `gather`)
   e. Implement greedy sampling using MLX ops (`mx.argmax`)
   f. Implement top-k sampling using MLX ops (`mx.topk`)

3. Create `moshi/moshi_mlx/offline_mlx.py` -- offline inference entry point:
   a. Mirror `moshi/moshi/offline.py` structure
   b. Use `rustymimi` for audio encoding/decoding
   c. Feed user WAV frames, collect agent audio
   d. Write output WAV

4. **Numerical parity test**: Run MLX offline inference on the same test assets used
   for the Stage 0 MPS baseline (same seed=42424242, same voice prompts, same input WAVs).
   Compare against the saved baseline artifacts:
   - **Assistant test**: `assets/test/input_assistant.wav` + `NATF2.pt` voice prompt
     -> compare MLX output against `baseline_assistant_mps.wav` / `.json`
   - **Service test**: `assets/test/input_service.wav` + `NATM1.pt` + service text prompt
     -> compare MLX output against `baseline_service_mps.wav` / `.json`
   - Text token sequences (should be identical or very close)
   - Audio token sequences per codebook
   - Output WAV waveform similarity (cross-correlation > 0.95)

5. **Performance comparison**: Run MLX offline with PerfLogger, compare per-component
   timings against Stage 0 baseline JSON files

**Key adaptation points for LMGen**:

The PersonaPlex `LMGen` has a significantly different state machine from `moshi_mlx`'s
`LmGen`. The key differences:

| Aspect | PersonaPlex LMGen | moshi_mlx LmGen |
|--------|-------------------|-----------------|
| State | `cache` + `provided` tensors + `offset` | `gen_sequence` array + `step_idx` |
| CUDA Graphs | `graphed_main`, `graphed_embeddings`, `graphed_depth` | None (MLX handles this) |
| Voice prompts | Full pipeline with embeddings cache | Not present |
| Input format | `input_tokens`, `moshi_tokens`, `text_token` separately | Combined token sequence |
| Output collection | Delay-based gather from cache | Similar delay-based collection |

The MLX port should follow the PersonaPlex state machine design (cache+provided+offset)
but replace CUDAGraphed calls with direct function calls wrapped in `mx.compile`.

**Files to create**:
- `moshi/moshi_mlx/lm_gen_mlx.py`
- `moshi/moshi_mlx/offline_mlx.py`
- `moshi/moshi_mlx/sampling_mlx.py`

**Exit criteria**: Greedy offline inference produces near-identical output to PyTorch.

---

### Stage 3: Voice Prompt System (Days 8-10)

**Goal**: Port the PersonaPlex voice prompting pipeline to MLX.

**Dependency**: Stage 2.

**Tasks**:

1. Port `LMGen.load_voice_prompt()` to MLX:
   a. Audio loading stays as `sphn.read()` + `sphn.resample()` (CPU, no change)
   b. LUFS normalization stays as `pyloudnorm` (CPU, no change)
   c. Store as numpy array, convert to MLX on demand

2. Port `LMGen._encode_voice_prompt_frames()`:
   a. Use `rustymimi` to encode voice prompt audio frames
   b. Iterate audio in `_frame_size` chunks (1920 samples)

3. Port `LMGen._step_voice_prompt_frame()`:
   a. Step the LM with voice prompt tokens as moshi_tokens
   b. Use sine tokens for input_tokens, zero_text_code for text

4. Port `LMGen._step_voice_prompt_core()`:
   a. Support both audio-based and pre-computed embeddings paths
   b. Yield at checkpoints for async connection checking

5. Port `LMGen.load_voice_prompt_embeddings()`:
   a. Load `.pt` files containing pre-saved embeddings
   b. Convert PyTorch tensors to MLX arrays

6. Port `LMGen.step_embeddings()`:
   a. Accept pre-computed embeddings and run through transformer

7. Port `step_system_prompts()` and `step_system_prompts_async()`:
   a. Text prompt injection via token forcing
   b. Audio silence frames between prompts
   c. Voice prompt frame iteration
   d. Async variant with `is_alive` callback

**Files to modify**:
- `moshi/moshi_mlx/lm_gen_mlx.py` -- add voice prompt methods

**Exit criteria**: Voice prompt + text prompt produces valid agent audio in offline mode.

---

### Stage 4: Real-Time Server Integration (Days 11-14)

**Goal**: Integrate the MLX model into the existing aiohttp WebSocket server.

**Dependency**: Stage 3.

**Tasks**:

1. Create `moshi/moshi_mlx/server_mlx.py` -- MLX-aware server:
   a. Import the aiohttp server structure from `moshi/moshi/server.py`
   b. Replace PyTorch model loading with MLX model loading
   c. Replace PyTorch inference calls with MLX inference calls
   d. Keep Opus I/O on CPU (sphn library, no change)
   e. Keep aiohttp async loop structure

2. Handle CPU/MLX data transfers:
   a. Audio frames arrive as numpy arrays from Opus -- convert to `mx.array`
   b. `rustymimi` handles its own internal encoding
   c. LM step input/output are MLX arrays
   d. Output PCM needs conversion back to numpy for Opus encoding
   e. Text tokens need `.item()` equivalent -- `mx.array.item()` or `int()`

3. Eliminate CPU sync bottlenecks:
   a. MLX's unified memory means no explicit CPU<->GPU transfers
   b. `mx.eval()` replaces `torch.mps.synchronize()`
   c. Strategic `mx.eval()` placement: once per frame, not per operation
   d. Use `mx.async_eval()` if available for pipelining

4. Handle async/threading:
   a. MLX operations run on the main thread by default
   b. The aiohttp event loop is async but single-threaded
   c. Model inference should be wrapped in `asyncio.to_thread()` or
      `loop.run_in_executor()` if it blocks the event loop
   d. Alternatively, interleave MLX eval with async yields

5. Warmup procedure:
   a. Port `ServerState.warmup()` to run 4 dummy frames through MLX model
   b. This primes `mx.compile` caches and allocates memory
   c. Call `mx.eval()` after warmup to force materialization

**Files to create**:
- `moshi/moshi_mlx/server_mlx.py`

**Files to modify**:
- `moshi/moshi/server.py` -- add `--backend mlx` flag to select MLX or PyTorch

**Exit criteria**: WebSocket server runs with MLX backend; real-time audio streams.

---

### Stage 5: Performance Optimization (Days 15-20)

**Goal**: Achieve <80ms per-frame latency through MLX-specific optimizations.

**Dependency**: Stage 4.

**Tasks**:

1. **`mx.compile` the hot path**:
   a. Compile `lm_model.forward_codes()` (main transformer forward)
   b. Compile `depformer_step()` (depformer sequential loop)
   c. Compile the full `lm_gen.step()` if pure (no Python control flow)
   d. Use `mx.compile(fun, inputs=state, outputs=state)` for stateful compilation
   e. Profile compilation overhead and set appropriate warmup

2. **Optimize attention**:
   a. Use `mx.fast.scaled_dot_product_attention` with `mask="causal"` string
   b. This avoids materializing the full attention matrix
   c. For the rotating KV cache context window, the cache capacity already limits
      the window -- no explicit mask needed beyond causal
   d. Benchmark against explicit mask for correctness

3. **Optimize dtype**:
   a. Use float16 for all model weights and activations
   b. Keep RMS norm computation in float32 (MLX handles this via `mx.fast.rms_norm`)
   c. Benchmark bfloat16 on M3 -- Apple Silicon M3+ has native bfloat16 support

4. **Optimize memory access patterns**:
   a. Ensure contiguous memory layout for KV cache updates
   b. Minimize array copies; use in-place updates where MLX supports them
   c. Use `mx.stop_gradient()` everywhere (inference only, no autograd needed)

5. **Quantization** (if needed for performance):
   a. 4-bit quantization via `mlx_lm` quantize tools
   b. Expected 4x memory bandwidth reduction
   c. Trade-off: slight quality degradation
   d. Test with `mlx.nn.QuantizedLinear` for the main transformer layers

6. **Eliminate dual Mimi**:
   a. Analyze whether `other_mimi` is required for PersonaPlex
   b. If it only maintains streaming state for a secondary stream that is discarded,
      remove it to save ~30-40% codec compute
   c. If needed, explore sharing encoder weights between instances

7. **Profile and iterate**:
   a. Use the performance logging framework to identify remaining bottlenecks
   b. Use `mx.metal.start_capture()` / `mx.metal.stop_capture()` for GPU profiling
   c. Target breakdown: LM forward <50ms, DepFormer <20ms, Codec <10ms

**Exit criteria**: Per-frame latency <80ms on M3; zero missed audio in 15-second test.

---

### Stage 6: Production Hardening (Days 21-25)

**Goal**: Make the MLX backend production-ready.

**Dependency**: Stage 5.

**Tasks**:

1. Error handling and graceful degradation
2. Memory leak testing (long-running sessions)
3. Concurrent connection handling (asyncio lock already exists)
4. Comprehensive test suite comparing MLX vs PyTorch outputs
5. Documentation and CLI integration
6. CI/CD for MLX backend tests (requires macOS runner)

**Exit criteria**: Stable server running for hours without degradation.

---

## 5. Performance Logging Framework

### Design

The performance logger tracks per-frame timing, cumulative statistics, and can be
enabled/disabled at zero cost when disabled.

### File: `moshi/moshi_mlx/perf_logger.py`

**Classes**:

```
class PerfLogger:
    """Structured performance logging for real-time inference."""

    def __init__(self, enabled: bool = True, output_path: str = None):
        """
        Args:
            enabled: If False, all methods are no-ops (zero overhead).
            output_path: Path for JSON/CSV output. If None, logs to stderr.
        """

    def frame_start(self) -> None:
        """Mark the start of a new frame."""

    def mark(self, label: str) -> None:
        """Record a named timestamp within the current frame.
        Labels: 'mimi_encode', 'lm_forward', 'depformer', 'mimi_decode',
                'opus_encode', 'opus_decode', 'total'
        """

    def frame_end(self, missed: bool = False) -> None:
        """Mark the end of a frame. Records total time and whether frame was missed."""

    def log_memory(self) -> None:
        """Record current GPU memory usage.
        MLX: mx.metal.get_active_memory(), mx.metal.get_peak_memory()
        PyTorch MPS: torch.mps.current_allocated_memory()
        """

    def summary(self) -> dict:
        """Return summary statistics as a dictionary."""

    def export_json(self, path: str) -> None:
        """Export all frame data as JSON."""

    def export_csv(self, path: str) -> None:
        """Export all frame data as CSV."""
```

**Frame data structure** (one entry per frame):

```json
{
  "frame_idx": 42,
  "timestamp_unix": 1706400000.123,
  "timings_ms": {
    "mimi_encode": 3.2,
    "lm_forward": 45.1,
    "depformer": 18.3,
    "mimi_decode": 4.7,
    "opus_io": 1.2,
    "total": 72.5
  },
  "missed": false,
  "memory_mb": {
    "active": 14200,
    "peak": 15800
  },
  "backend": "mlx",
  "stage": "stage_4"
}
```

**Summary statistics**:

```json
{
  "total_frames": 187,
  "missed_frames": 0,
  "avg_frame_ms": 68.3,
  "p50_frame_ms": 65.1,
  "p95_frame_ms": 78.2,
  "p99_frame_ms": 79.8,
  "max_frame_ms": 82.1,
  "avg_lm_forward_ms": 42.5,
  "avg_depformer_ms": 16.8,
  "avg_codec_ms": 7.9,
  "peak_memory_mb": 15800,
  "backend": "mlx",
  "device": "Apple M3 96GB"
}
```

### Integration Points (in priority order)

1. **Offline benchmark** (`benchmark.py`) -- **FIRST INTEGRATION (Stage 0)**:
   Wraps `moshi.offline.run_inference()` with PerfLogger instrumentation. This is the
   primary benchmarking tool, used for:
   - MPS baseline capture (Stage 0) using existing test assets:
     ```bash
     # Assistant test case
     python -m moshi_mlx.benchmark \
       --voice-prompt "NATF2.pt" \
       --input-wav "assets/test/input_assistant.wav" \
       --seed 42424242 \
       --backend mps \
       --output "baseline_assistant_mps_perf.json"

     # Service test case
     python -m moshi_mlx.benchmark \
       --voice-prompt "NATM1.pt" \
       --text-prompt "$(cat assets/test/prompt_service.txt)" \
       --input-wav "assets/test/input_service.wav" \
       --seed 42424242 \
       --backend mps \
       --output "baseline_service_mps_perf.json"
     ```
   - MLX comparison at each stage (using same inputs, comparing against baseline)
   - Per-component breakdown: Mimi encode, LM forward, depformer, Mimi decode

2. **Comparison mode**: Run both PyTorch MPS and MLX backends on the same input,
   logging to separate JSON files. The benchmark harness can diff the results:
   ```bash
   python -m moshi_mlx.benchmark --compare \
     baseline_assistant_mps_perf.json \
     mlx_assistant_stage2_perf.json
   ```

3. **Server integration** (`server_mlx.py`) -- **LATER (Stage 4)**:
   Wrap the opus_loop's inner processing with `perf.frame_start()` / `perf.frame_end()`:
   ```
   perf.frame_start()
   perf.mark('opus_decode')
   codes = mimi_encode(chunk)
   perf.mark('mimi_encode')
   tokens = lm_gen.step(codes)
   perf.mark('lm_forward')
   pcm = mimi_decode(tokens)
   perf.mark('mimi_decode')
   opus_writer.append_pcm(pcm)
   perf.mark('opus_encode')
   perf.frame_end(missed=frame_time > 80)
   ```

4. **Disable for production**: `PerfLogger(enabled=False)` makes all methods no-ops.
   Use `if self.enabled:` guard at the top of each method for zero-cost disable.

### Timing Mechanism

- MLX: Use `mx.eval()` before `time.perf_counter()` to ensure computation is complete
  (MLX is lazy, so timing without eval would measure graph construction, not execution)
- PyTorch MPS: Use `torch.mps.synchronize()` before `time.perf_counter()`
- Wrap timing in context managers for clean code:
  ```
  with perf.timed('lm_forward'):
      tokens = lm_gen.step(codes)
  ```

---

## 6. MLX Architecture Mapping

### Component-by-Component Mapping

#### 6.1 StreamingTransformer

**PyTorch** (`moshi/modules/transformer.py`):
- `StreamingTransformer` extends `StreamingContainer`
- 32 `StreamingTransformerLayer` instances
- Each layer: `_sa_block()` (self-attention) + `_ff_block()` (FFN)
- Uses `no_compile()` on non-CUDA devices (kills performance on MPS)
- RoPE applied inside attention

**MLX** (`moshi_mlx/modules/transformer.py`):
- `Transformer` class with `TransformerLayer` list
- Each layer: `Attention` + `MlpGating`/`MlpNoGating`
- Uses `mx.fast.scaled_dot_product_attention`
- RoPE applied via `mx.fast.rope` (C++ optimized)
- KV cache managed via `LayerCache` dataclass

**Mapping**:
```
PyTorch StreamingTransformer       --> MLX Transformer
  StreamingTransformerLayer        --> TransformerLayer
    MultiheadAttention._sa_block   --> Attention.__call__
    F.scaled_dot_product_attention --> mx.fast.scaled_dot_product_attention(mask="causal")
    SwiGLU._ff_block               --> MlpGating.__call__
    RMSNorm                        --> mx.fast.rms_norm or mlx.nn.RMSNorm
    RoPE (apply_rope)              --> mx.fast.rope
```

#### 6.2 RoPE (Rotary Position Embeddings)

**PyTorch** (`moshi/modules/rope.py`):
- `RotaryEmbedding` class computes sin/cos buffers
- `apply_rope()` decorated with `@torch_compile_lazy`
- Rotates pairs of dimensions using complex number multiplication

**MLX**:
- `mlx.nn.RoPE` built-in module
- `mx.fast.rope` for optimized C++ implementation
- Supports both traditional (consecutive pairs) and default (stride-half) rotations
- Parameters: `dims`, `traditional=False`, `base=10000.0`

**Mapping**: Replace `RotaryEmbedding` + `apply_rope()` with `mx.fast.rope`. The MLX
implementation accepts an offset parameter for streaming (KV cache position tracking).

#### 6.3 SiLU Gating FFN

**PyTorch** (`moshi/modules/gating.py`):
- `ActivationGating` with `gating_forward_kernel` compiled via `@torch_compile_lazy`
- Computes: `linear1(x) * activation(linear_gate(x))`
- `activation` = SiLU for PersonaPlex

**MLX**:
- `MlpGating` in `moshi_mlx/modules/transformer.py`
- Uses `mlx.nn.Linear` for projections, `mlx.nn.silu` activation
- No compilation needed -- MLX's lazy evaluation handles fusion

**Mapping**: Direct replacement. SiLU is `mx.nn.silu()` or `mx.sigmoid(x) * x`.

#### 6.4 RMS Norm

**PyTorch** (`moshi/modules/transformer.py:55-67`):
- Custom `_rms_norm()` with explicit float32 promotion
- Called 76 times per frame (2x per layer, 32+6 layers)

**MLX**:
- `mx.fast.rms_norm(x, weight, eps)` -- optimized C++ kernel
- Handles precision internally
- Or `mlx.nn.RMSNorm(dims, eps)` module

**Mapping**: Replace `_rms_norm` with `mx.fast.rms_norm`. The float32 promotion is
handled internally by MLX's implementation. This eliminates bottleneck #9 from the
optimization report.

#### 6.5 Attention (SDPA)

**PyTorch** (`moshi/modules/transformer.py:430-441`):
- `F.scaled_dot_product_attention(q, k, v, attn_bias)`
- Constructs boolean mask every call (bottleneck #3)
- No Flash Attention on MPS

**MLX**:
- `mx.fast.scaled_dot_product_attention(q, k, v, mask="causal")
- The string `"causal"` triggers an optimized causal attention kernel
- Supports GQA natively (k,v not pre-tiled)
- Softmax computed in float32 regardless of input precision

**Mapping**: Replace the entire mask construction + SDPA call with:
```
mx.fast.scaled_dot_product_attention(q, k, v, mask="causal", scale=scale)
```
For the context-windowed attention (context=3000), the RotatingKVCache already limits
keys to the last 3000 positions, so no explicit context mask is needed during streaming.

#### 6.6 CUDA Graphs --> mx.compile / Lazy Evaluation

**PyTorch** (`moshi/utils/compile.py`):
- `CUDAGraphed` captures and replays GPU operations
- Used for `forward_codes`, `forward_embeddings`, `depformer_step`
- Disabled on non-CUDA devices (bottleneck #1)

**MLX**:
- `mx.compile(fn)` compiles a function into an optimized compute graph
- Lazy evaluation automatically batches operations
- `mx.eval()` triggers execution of the accumulated graph
- `mx.compile` caches compiled functions for repeated calls

**Mapping**:
```python
# Instead of CUDAGraphed:
# graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
# result = graphed_main(input_)

# Use mx.compile:
compiled_forward = mx.compile(lm_model.forward_codes)
result = compiled_forward(input_)
mx.eval(result)  # Force execution at strategic points
```

For the depformer sequential loop (8 steps), compile the inner function:
```python
compiled_depformer_step = mx.compile(lm_model.forward_depformer)
for cb_index in range(dep_q):
    logits = compiled_depformer_step(cb_index, input_, transformer_out)
```

**Important**: `mx.compile` requires pure functions (no side effects). The KV cache
updates must be handled via the `inputs`/`outputs` parameters:
```python
compiled_fn = mx.compile(fn, inputs=cache_state, outputs=cache_state)
```

#### 6.7 SEANet Encoder/Decoder

**PyTorch** (`moshi/modules/seanet.py`):
- Stacked Conv1d with residual blocks, ELU activation
- Streaming via `StreamingModule` with padding state
- Ratios [8,6,5,4] for downsampling (total 960x)

**MLX** (`moshi_mlx/modules/seanet.py`):
- Already ported in `moshi_mlx`
- Uses `StreamableConv1d` with causal padding and streaming state

**Mapping**: Use the existing `moshi_mlx` SEANet implementation. Only needed if
replacing `rustymimi` with a pure MLX codec.

#### 6.8 Vector Quantization

**PyTorch** (`moshi/quantization/core_vq.py`):
- `EuclideanCodebook` uses `torch.cdist` (bottleneck #7 on MPS)
- `ResidualVectorQuantization` with 8 codebook layers

**MLX** (`moshi_mlx/modules/quantization.py`):
- Already ported in `moshi_mlx`
- Likely uses matmul-based distance (more efficient)

**Mapping**: Use the existing `moshi_mlx` VQ implementation. The `torch.cdist` issue
is automatically resolved since MLX uses its own distance computation.

#### 6.9 Sampling

**PyTorch** (`moshi/utils/sampling.py`):
- `sample_token()`: softmax -> top_k/top_p -> multinomial
- Custom `multinomial()` using Gumbel trick to avoid sync point
- `sample_top_k()`: `torch.topk` + gather

**MLX**:
- `mx.softmax(logits / temp, axis=-1)` for temperature scaling
- `mx.topk(probs, k)` for top-k selection
- `mx.random.categorical(logits)` for sampling
- Or use the `moshi_mlx` sampling module

**Mapping**: Port `sample_token()` to MLX:
```
def sample_token_mlx(logits, use_sampling, temp, top_k):
    if use_sampling and temp > 0:
        probs = mx.softmax(logits / temp, axis=-1)
        if top_k > 0:
            top_probs, indices = mx.topk(probs, top_k, axis=-1)
            token = mx.random.categorical(mx.log(top_probs + 1e-10))
            next_token = indices.take(token, axis=-1)
        else:
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
    else:
        next_token = mx.argmax(logits, axis=-1)
    return next_token
```

#### 6.10 Opus Audio I/O

**Stays as-is.** The `sphn` library handles Opus encoding/decoding on CPU. Audio
arrives as numpy arrays and is returned as numpy arrays. The only change is converting
between numpy and MLX arrays at the boundary:
```python
# numpy -> MLX
chunk_mlx = mx.array(chunk_numpy)

# MLX -> numpy
pcm_numpy = np.array(pcm_mlx)
```

MLX's unified memory makes these conversions nearly zero-cost (shared memory, no copy).

---

## 7. Weight Conversion

### Overview

PersonaPlex weights are stored as `model.safetensors` (bfloat16) at
`nvidia/personaplex-7b-v1` on HuggingFace. The Mimi codec weights are at
`tokenizer-e351c8d8-checkpoint125.safetensors`.

### MLX Safetensors Support

MLX can load safetensors directly:
```python
import mlx.core as mx
weights = mx.load("model.safetensors")  # Returns dict of mx.array
```

MLX supports bfloat16 natively on M3+ (Apple Silicon with bfloat16 hardware support).
For M1/M2, float16 is preferred. The conversion can be done at load time:
```python
weights = {k: v.astype(mx.float16) for k, v in weights.items()}
```

### PersonaPlex-Specific Weight Patches

The conversion script must replicate the logic from `loaders.py:218-258`:

**Patch 1: DepFormer self_attn expansion**

```
For each key containing "depformer" and "self_attn":
    If checkpoint shape != model shape:
        Expand by concatenating the tensor with itself
        (copy_missing_weights=True means duplicate the existing weights)
```

This handles the case where the checkpoint has dep_q=8 weights but the model expects
dep_q=16 for the self-attention layers.

**Patch 2: Copy codebooks 0..7 to 8..15**

```
For missing keys matching patterns [gating, linears, depformer_in, depformer_emb]:
    Replace ".{8-15}." with ".{0-7}." and copy the source weights
```

This duplicates the first 8 codebook-specific parameters to fill in codebooks 8-15.

### Conversion Script Design

File: `moshi/moshi_mlx/convert_weights.py`

```
def convert_personaplex_weights(
    input_path: str,           # Path to model.safetensors
    output_path: str,          # Output path for MLX weights
    dtype: str = "float16",    # Target dtype: "float16" or "bfloat16"
    dep_q_target: int = 16,    # dep_q for weight expansion
) -> None:
    """Convert PersonaPlex PyTorch safetensors to MLX format.

    Steps:
    1. Load safetensors with torch (to handle patches)
    2. Apply dep_q expansion patches (self_attn, codebook copying)
    3. Rename keys to match moshi_mlx naming convention (if different)
    4. Convert to target dtype
    5. Save as MLX safetensors
    """
```

### Key Name Mapping

The PyTorch and MLX models may use different key naming conventions. The mapping
should handle:

```
PyTorch key pattern              --> MLX key pattern
transformer.layers.{N}.          --> transformer.layers.{N}.
depformer.layers.{N}.            --> depformer.layers.{N}.
emb.{N}.weight                   --> emb.{N}.weight
text_emb.weight                  --> text_emb.weight
text_linear.weight               --> text_linear.weight
linears.{N}.weight               --> linears.{N}.weight
depformer_in.{N}.weight          --> depformer_in.{N}.weight
depformer_emb.{N}.weight         --> depformer_emb.{N}.weight
depformer_text_emb.weight        --> depformer_text_emb.weight
out_norm.weight                  --> out_norm.weight
```

The exact mapping depends on the `moshi_mlx` model class attribute names, which should
be inspected from the Kyutai source code during implementation.

### Mimi Weights

The Mimi codec weights (`tokenizer-e351c8d8-checkpoint125.safetensors`) need conversion
only if NOT using `rustymimi`. If using `rustymimi`, it handles its own weight loading
from the Kyutai HuggingFace repo. The PersonaPlex Mimi weights may need to be registered
or served separately.

---

## 8. Streaming State Management

### Overview

The PersonaPlex streaming system has three levels of state:

1. **LMGen state** -- `_LMGenState` with cache, provided, initial, offset
2. **Transformer KV cache** -- `RingKVCache` per attention layer
3. **Conv streaming state** -- Padding buffers in SEANet conv layers

### LMGen State Machine

The PersonaPlex `LMGen` uses a circular cache pattern:

```
_LMGenState:
    cache: [B, num_codebooks, max_delay+3]   -- stores all token streams
    provided: [B, num_codebooks, max_delay+3] -- marks which positions were provided
    initial: [B, num_codebooks, 1]            -- initial tokens
    offset: int                                -- current position (increments each step)
```

**MLX equivalent**:

```python
class LMGenStateMLX:
    def __init__(self, batch_size, num_codebooks, max_delay, device):
        ct = max_delay + 3
        self.cache = mx.full((batch_size, num_codebooks, ct),
                             fill_value=UNGENERATED_TOKEN_ID, dtype=mx.int32)
        self.provided = mx.zeros((batch_size, num_codebooks, ct), dtype=mx.bool_)
        self.initial = None  # Set during init
        self.offset = 0

    def reset(self):
        self.offset = 0
        self.provided = mx.zeros_like(self.provided)
```

**Key difference from PyTorch**: MLX arrays are immutable by default. In-place updates
like `state.cache[:, k, pos] = value` become:
```python
# Option A: Use mx.array.__setitem__ (supported but creates new array)
state.cache = state.cache.at[..., k, pos].set(value)

# Option B: Reconstruct the relevant slice
# This is more efficient for small updates
```

MLX does support `__setitem__` for array mutation, which creates a copy under the hood
but is optimized by the lazy evaluation engine to minimize actual copies. For the
cache update pattern (writing single positions), this should be efficient.

### KV Cache

The PyTorch code uses `RingKVCache` (`transformer.py:239-303`) -- a circular buffer
for key/value pairs that wraps around at `capacity`.

**MLX equivalent** (`moshi_mlx/modules/kv_cache.py`):
- `KVCache`: Standard growing cache with step-based allocation
- `RotatingKVCache`: Fixed-size circular buffer (matches PyTorch `RingKVCache`)

The existing `moshi_mlx` KV cache should work directly. Key parameters:
- Main transformer: `context=3000` -> RotatingKVCache with max_size=3000
- DepFormer: `context=8` -> Small KVCache

**Critical detail**: The `moshi_mlx` `RotatingKVCache` uses an `_idx` write pointer
and `keep` parameter to preserve initial tokens. This matches the PersonaPlex behavior.

### Streaming Convolution State

The SEANet streaming state tracks padding buffers:

**PyTorch** (`conv.py:180-186`):
```python
@dataclass
class _StreamingConv1dState:
    padding_to_add: int
    original_padding_to_add: int
```

**MLX** (`moshi_mlx/modules/conv.py`):
- `StreamableConv1d` maintains `_prev_xs` buffer
- `_left_pad_applied` flag for initial padding

The existing `moshi_mlx` streaming conv handles this. If using `rustymimi`, the Rust
codec manages its own streaming state internally.

### State Reset

Both the PyTorch and MLX versions need state reset between conversations:

```python
# PyTorch:
self.mimi.reset_streaming()
self.other_mimi.reset_streaming()
self.lm_gen.reset_streaming()

# MLX equivalent:
self.mimi.reset_state()          # rustymimi reset
self.lm_gen_mlx.reset_state()    # Reset LMGen state
# Each transformer layer's KV cache resets
# Each conv layer's streaming buffer resets
```

### StreamingModule/StreamingContainer Pattern

**PyTorch** (`streaming.py`):
- `StreamingModule[T]` -- generic base with `_streaming_state: T | None`
- `StreamingContainer` -- propagates streaming to children
- `streaming(batch_size)` context manager initializes/tears down state
- `streaming_forever(batch_size)` -- persistent streaming mode

**MLX**:
The `moshi_mlx` code does NOT use the same generic StreamingModule pattern. Instead,
each module manages its own state via explicit `reset_state()` methods and step counters.
The LMGen tracks state via its own attributes.

**Recommendation**: Follow the `moshi_mlx` pattern (explicit state management) rather
than trying to replicate the generic StreamingModule/StreamingContainer pattern. The
generic pattern adds complexity without benefit in the MLX context, since MLX does not
need the same kind of device-aware state initialization.

---

## 9. Server Integration

### Architecture

```
[Browser] <-- WebSocket --> [aiohttp server] <-- Python --> [MLX Model]
                                |
                            [Opus I/O] (CPU, sphn library)
                                |
                         [Audio Processing] (numpy)
```

### Data Flow (MLX Backend)

```
1. Client sends Opus audio bytes via WebSocket
2. opus_reader.read_pcm() -> numpy float32 array (CPU)
3. Accumulate until frame_size (1920 samples)
4. chunk = mx.array(numpy_chunk)  # nearly zero-cost, shared memory
5. codes = rustymimi.encode(chunk)  # Rust codec
6. tokens = lm_gen_mlx.step(codes)  # MLX inference
7. mx.eval(tokens)  # Force evaluation
8. pcm = rustymimi.decode(tokens)  # Rust codec
9. pcm_numpy = np.array(pcm)  # nearly zero-cost
10. opus_writer.append_pcm(pcm_numpy)  # CPU
11. Send Opus bytes to client via WebSocket
```

### Key Integration Points

**1. Model Loading** (`server_mlx.py`):

```python
def load_mlx_models(args):
    # Load Mimi via rustymimi
    mimi = rustymimi.MimiCodec(...)

    # Load LM via MLX
    weights = mx.load(args.moshi_weight)
    weights = apply_personaplex_patches(weights)
    lm_model = create_lm_model_mlx(weights, dtype=mx.float16)

    # Create LMGen
    lm_gen = LMGenMLX(lm_model, ...)
    return mimi, lm_gen
```

**2. Async Integration**:

The aiohttp event loop is async, but MLX computation is synchronous (blocks until
`mx.eval()` completes). Options:

a. **Run MLX in executor** (recommended for initial version):
```python
async def opus_loop():
    loop = asyncio.get_event_loop()
    ...
    # Run model inference in thread pool to avoid blocking event loop
    tokens = await loop.run_in_executor(None, lm_gen.step, codes)
```

b. **Interleave with async yields** (for advanced optimization):
```python
async def opus_loop():
    ...
    # Submit work to MLX (lazy, returns immediately)
    tokens = lm_gen.step(codes)  # Does not block
    # Let event loop do other work
    await asyncio.sleep(0)
    # Now force evaluation
    mx.eval(tokens)
```

c. **Single-threaded async** (simplest, may work if model is fast enough):
```python
async def opus_loop():
    ...
    tokens = lm_gen.step(codes)
    mx.eval(tokens)  # Blocks, but <80ms should be fine
```

**Recommendation**: Start with option (c) for simplicity. The 80ms budget is generous
enough that blocking the event loop briefly should not cause WebSocket timeouts.
Move to option (a) if async responsiveness becomes an issue.

**3. Text Token Extraction**:

In PyTorch: `text_token = tokens[0, 0, 0].item()` -- forces a CPU sync.

In MLX: `text_token = int(tokens[0, 0, 0])` -- also forces evaluation, but since we
already called `mx.eval(tokens)`, this is nearly free. MLX's unified memory means the
value is already accessible without a copy.

**4. Warmup**:

```python
def warmup_mlx(mimi, lm_gen, device):
    """Run 4 dummy frames to prime mx.compile caches."""
    for _ in range(4):
        chunk = mx.zeros((1, 1, frame_size))
        codes = mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c:c+1])
    mx.eval()  # Force all computation
```

**5. Memory Management**:

MLX on Apple Silicon uses unified memory. With 96GB available:
- 7B model in float16: ~14GB
- KV cache (context=3000): ~2GB
- Mimi codec: ~200MB
- Working memory: ~2GB
- Total: ~18GB (well within 96GB budget)

No explicit memory management needed. MLX handles allocation/deallocation automatically.

---

## 10. Risk Assessment

### Risk 1: MLX Operation Coverage

**Risk**: Some PyTorch operations used in PersonaPlex may not have MLX equivalents.

**Likelihood**: LOW. The `moshi_mlx` port already handles all core operations.

**Mitigation**: The existing `moshi_mlx` codebase proves that all fundamental operations
(attention, conv, VQ, sampling) work in MLX. PersonaPlex-specific additions (cache
management, token manipulation) use basic array operations that MLX supports.

**Fallback**: For any unsupported operation, use numpy as an intermediary (MLX arrays
can be converted to/from numpy with near-zero cost due to shared memory).

### Risk 2: `rustymimi` Incompatibility with PersonaPlex

**Risk**: The `rustymimi` Rust codec may not support PersonaPlex's specific Mimi
configuration (weight format, streaming state management, dual instances).

**Likelihood**: MEDIUM. PersonaPlex uses custom Mimi weights from `nvidia/personaplex-7b-v1`
which may differ from Kyutai's default weights. The dual-Mimi pattern is unusual.

**Mitigation**:
1. Test `rustymimi` with PersonaPlex Mimi weights early (Stage 1)
2. If incompatible, fall back to the pure MLX Mimi implementation from `moshi_mlx`
3. The pure MLX Mimi uses `moshi_mlx/models/mimi.py` + `modules/seanet.py` + `modules/conv.py`
   which are already fully ported

**Fallback**: Use pure MLX Mimi instead of rustymimi. This adds ~5ms latency per
encode/decode but is fully controllable.

### Risk 3: Numerical Divergence

**Risk**: float16 computation may produce different results than bfloat16, causing
quality degradation or incorrect outputs.

**Likelihood**: LOW-MEDIUM. float16 has more precision but less dynamic range than
bfloat16. Most transformer models work fine in float16 for inference.

**Mitigation**:
1. Run numerical parity tests comparing bfloat16 PyTorch vs float16 MLX outputs
2. Keep RMS norm in float32 (MLX does this automatically via `mx.fast.rms_norm`)
3. Test with bfloat16 on M3 (which has hardware bfloat16 support)
4. If issues arise, use mixed precision: float32 for norms, float16 for compute

### Risk 4: Performance Still Not Real-Time

**Risk**: Even with MLX, per-frame latency exceeds 80ms on M3.

**Likelihood**: LOW. MLX benchmarks show that 7B models can generate tokens at 30-50
tokens/second on M3 Max, which is ~20-33ms per token. PersonaPlex needs 1 main token
+ 8 depformer tokens per frame = 9 token equivalents. At 33ms per main token + 8x
~3ms per depformer token = ~57ms, well within budget.

**Mitigation**:
1. 4-bit quantization reduces model size by 4x, proportionally increasing throughput
2. Remove the dual Mimi instance (~30% codec speedup)
3. Profile with `mx.metal.start_capture()` to identify any remaining hotspots
4. Custom Metal kernels for any bottleneck operations

**Fallback**: Use 4-bit quantization (expected to bring 7B model to ~15-20ms per token,
giving substantial headroom).

### Risk 5: `moshi_mlx` API Changes

**Risk**: The Kyutai `moshi_mlx` codebase may change incompatibly between versions.

**Likelihood**: LOW-MEDIUM. The package is at v0.3.0 and appears stable.

**Mitigation**:
1. Pin `moshi_mlx` to a specific version in `requirements.txt`
2. Fork the relevant source files into `moshi/moshi_mlx/` rather than depending on
   the PyPI package at runtime
3. This gives full control over the codebase and allows PersonaPlex-specific modifications

**Recommendation**: Fork the `moshi_mlx` source files directly into the PersonaPlex
project. This avoids version dependency issues and allows direct modification.

### Risk 6: Voice Prompt Embedding Compatibility

**Risk**: Pre-saved `.pt` voice prompt embeddings (PyTorch tensors) may not load
cleanly into MLX.

**Likelihood**: LOW. PyTorch `.pt` files can be loaded with `torch.load()` and
converted to MLX arrays via numpy.

**Mitigation**:
1. Provide a conversion script for `.pt` -> MLX format
2. Support loading both formats
3. Re-generate voice prompt embeddings using the MLX pipeline

### Risk 7: Threading / Async Issues

**Risk**: MLX operations may not play well with Python's asyncio or threading.

**Likelihood**: LOW. MLX operations are thread-safe and the GIL prevents true
concurrent access issues. The `run_in_executor` pattern is well-tested.

**Mitigation**:
1. Use the simple single-threaded approach (option c in Section 9) first
2. MLX is designed for single-process, single-GPU usage on Apple Silicon
3. The asyncio lock already serializes model access

---

## 11. File-by-File Port Reference

### New Files to Create

| File | Purpose | Stage |
|------|---------|-------|
| `moshi/moshi_mlx/__init__.py` | Package init | 0 |
| `moshi/moshi_mlx/perf_logger.py` | Performance logging framework | 0 |
| `moshi/moshi_mlx/benchmark.py` | Benchmark harness | 0 |
| `moshi/moshi_mlx/convert_weights.py` | Weight conversion pipeline | 1 |
| `moshi/moshi_mlx/loaders_mlx.py` | MLX model loading | 1 |
| `moshi/moshi_mlx/lm_gen_mlx.py` | PersonaPlex LMGen for MLX | 2 |
| `moshi/moshi_mlx/sampling_mlx.py` | Token sampling utilities | 2 |
| `moshi/moshi_mlx/offline_mlx.py` | Offline inference entry point | 2 |
| `moshi/moshi_mlx/server_mlx.py` | MLX-aware aiohttp server | 4 |

### Files to Fork from `moshi_mlx` (Kyutai)

These files should be copied from `github.com/kyutai-labs/moshi/tree/main/moshi_mlx`
and modified for PersonaPlex:

| Kyutai File | PersonaPlex Target | Modifications Needed |
|-------------|-------------------|---------------------|
| `models/lm.py` | `moshi/moshi_mlx/models/lm.py` | Add n_q=16, dep_q expansion, PersonaPlex config |
| `models/generate.py` | Merged into `lm_gen_mlx.py` | Rewrite to PersonaPlex state machine |
| `models/mimi.py` | `moshi/moshi_mlx/models/mimi.py` | Keep as fallback to rustymimi |
| `modules/transformer.py` | `moshi/moshi_mlx/modules/transformer.py` | Minimal changes (config) |
| `modules/conv.py` | `moshi/moshi_mlx/modules/conv.py` | No changes expected |
| `modules/seanet.py` | `moshi/moshi_mlx/modules/seanet.py` | No changes expected |
| `modules/kv_cache.py` | `moshi/moshi_mlx/modules/kv_cache.py` | No changes expected |
| `modules/quantization.py` | `moshi/moshi_mlx/modules/quantization.py` | No changes expected |

### Existing Files to Modify

| File | Modification |
|------|-------------|
| `moshi/moshi/server.py` | Add `--backend mlx` flag; import MLX server when selected |
| `moshi/moshi/offline.py` | Add `--backend mlx` flag; dispatch to MLX offline |

### Files That Stay As-Is

| File | Reason |
|------|--------|
| `moshi/moshi/client_utils.py` | Pure utility functions, no model dependency |
| All client/frontend code | WebSocket protocol unchanged |
| `sphn` / Opus handling | CPU-only audio codec, no MLX involvement |
| `sentencepiece` tokenizer | CPU text tokenization, unchanged |

---

## 12. Timeline and Milestones

### Week 1: Foundation (Stages 0-1)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Build PerfLogger + benchmark harness; instrument offline.py | `perf_logger.py`, `benchmark.py` |
| 1 | Run MPS baseline with both test cases (assistant + service) | Baseline JSON + WAV artifacts |
| 2 | Install MLX; verify moshi_mlx base model runs on M3 | MLX environment confirmed |
| 2-3 | Weight conversion script | `convert_weights.py` |
| 3 | MLX model loading, parameter verification | `loaders_mlx.py` |

### Week 2: Core Inference (Stage 2)

| Day | Task | Deliverable |
|-----|------|-------------|
| 4 | Fork moshi_mlx modules, adapt LM config | MLX model architecture |
| 5 | Port LMGen state machine (cache/provided/offset) | `lm_gen_mlx.py` core |
| 6 | Port sampling, depformer loop, delay collection | Complete step() |
| 7 | Numerical parity test (greedy, PyTorch vs MLX) | Test report |

### Week 3: Voice Prompts + Server (Stages 3-4)

| Day | Task | Deliverable |
|-----|------|-------------|
| 8 | Port voice prompt loading and encoding | Voice prompt pipeline |
| 9 | Port voice prompt stepping and system prompts | System prompt pipeline |
| 10 | Integration test: voice+text prompt offline | End-to-end offline test |
| 11 | Server integration: aiohttp + MLX model | `server_mlx.py` |
| 12 | Server integration: opus loop, text streaming | Working WebSocket demo |
| 13 | Server warmup, state reset, connection handling | Stable server |
| 14 | End-to-end real-time test, bug fixes | Real-time demo |

### Week 4: Optimization + Hardening (Stages 5-6)

| Day | Task | Deliverable |
|-----|------|-------------|
| 15 | mx.compile hot paths | Compiled inference |
| 16 | Optimize attention (causal mask string) | Faster attention |
| 17 | Profile and fix remaining bottlenecks | <80ms per frame |
| 18 | Quantization experiments (4-bit) | Optional quantized model |
| 19 | Remove dual Mimi, memory optimization | Lean memory footprint |
| 20 | Long-running stability test, memory leak check | Stability report |
| 21-25 | Production hardening, docs, CI | Production-ready backend |

### Key Milestones

| Milestone | Target Date | Criteria |
|-----------|-------------|----------|
| **M0: MPS Baseline captured** | **Day 1** | **PerfLogger built; both test cases profiled; baseline JSONs + WAVs saved** |
| M1: Weights load in MLX | Day 3 | All parameters loaded, shapes verified |
| M2: Offline inference works | Day 7 | Greedy output matches baseline WAVs (parity test passes) |
| M3: Voice prompts work | Day 10 | Voice+text prompt produces valid audio |
| M4: Server runs real-time | Day 14 | WebSocket streaming with MLX backend |
| M5: Performance target met | Day 17 | Per-frame latency <80ms (verified against M0 baseline) |
| M6: Production ready | Day 25 | Stable, documented, tested |

---

## Appendix A: Quick Reference -- MLX API Equivalents

| PyTorch | MLX |
|---------|-----|
| `torch.Tensor` | `mx.array` |
| `torch.nn.Module` | `mlx.nn.Module` |
| `torch.nn.Linear` | `mlx.nn.Linear` |
| `torch.nn.Embedding` | `mlx.nn.Embedding` |
| `torch.nn.Conv1d` | `mlx.nn.Conv1d` |
| `torch.nn.ConvTranspose1d` | `mlx.nn.ConvTranspose1d` (not built-in, custom) |
| `F.scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` |
| `F.silu` | `mlx.nn.silu` |
| `F.elu` | `mlx.nn.elu` |
| `F.embedding` | `mx.array.__getitem__` (indexing) |
| `torch.compile` | `mx.compile` |
| `torch.no_grad()` | Not needed (MLX has no autograd by default for inference) |
| `tensor.to(device)` | Not needed (unified memory) |
| `tensor.cpu()` | `np.array(mx_array)` (zero-copy) |
| `tensor.item()` | `int(mx_array)` or `float(mx_array)` |
| `torch.topk` | `mx.topk` |
| `torch.softmax` | `mx.softmax` |
| `torch.argmax` | `mx.argmax` |
| `torch.where` | `mx.where` |
| `torch.cat` | `mx.concatenate` |
| `torch.stack` | `mx.stack` |
| `torch.full` | `mx.full` |
| `torch.zeros` | `mx.zeros` |
| `torch.multinomial` | `mx.random.categorical` |
| `torch.gather` | `mx.take_along_axis` |
| `tensor.clone()` | `mx.array(arr)` (copy) |
| `tensor.float()` | `arr.astype(mx.float32)` |
| `tensor.long()` | `arr.astype(mx.int64)` or `mx.int32` |
| `tensor.view/reshape` | `mx.reshape` |
| `tensor.expand` | `mx.broadcast_to` |
| `tensor.unsqueeze(dim)` | `mx.expand_dims(arr, axis=dim)` |
| `tensor.squeeze(dim)` | `mx.squeeze(arr, axis=dim)` |
| `tensor.transpose` | `mx.transpose` |
| `torch.einsum` | `mx.einsum` |
| `nn.LayerNorm` | `mlx.nn.LayerNorm` |
| `nn.RMSNorm` (custom) | `mlx.nn.RMSNorm` or `mx.fast.rms_norm` |
| `RotaryEmbedding` | `mlx.nn.RoPE` or `mx.fast.rope` |
| `nn.ModuleList` | Python list (MLX uses `__setattr__` for parameter registration) |
| `CUDAGraph` | `mx.compile` + lazy evaluation |
| `torch.mps.synchronize()` | `mx.eval()` |

## Appendix B: Bottleneck Resolution Summary

| # | Bottleneck | PyTorch MPS Impact | MLX Resolution |
|---|-----------|-------------------|----------------|
| 1 | CUDA Graphs disabled | CRITICAL (2-5x) | `mx.compile` + lazy eval |
| 2 | torch.compile disabled | CRITICAL (2-4x) | `mx.compile` automatic |
| 3 | No Flash Attention + bool mask | HIGH (1.5-3x) | `mx.fast.scaled_dot_product_attention(mask="causal")` |
| 4 | bfloat16 on MPS | HIGH (1.5-2x) | Native float16 or bfloat16 (M3 hardware) |
| 5 | CPU sync in hot path | HIGH (5-50ms) | Unified memory, zero-copy transfers |
| 6 | index_copy_ workaround | MODERATE | Direct array indexing in MLX |
| 7 | torch.cdist in VQ | MODERATE | MLX VQ uses matmul-based distance |
| 8 | gather operations | MODERATE | `mx.take_along_axis` (native) |
| 9 | RMS norm float32 promotion | LOW-MOD | `mx.fast.rms_norm` (handles internally) |
| 10 | Sequential DepFormer | MODERATE | `mx.compile` the inner loop |
| 11 | Dual Mimi instances | MODERATE | Remove if possible, or use rustymimi |
| 12 | Tight polling loop | LOW-MOD | Less relevant with fast inference |

**Expected combined resolution**: All 12 bottlenecks are addressed by the MLX port,
yielding an estimated 5-10x speedup over PyTorch MPS (from ~400ms to ~40-80ms per frame).
