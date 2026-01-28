# PersonaPlex MPS (Apple Silicon) Performance Analysis

## Observed Performance

| Metric | Value |
|--------|-------|
| Device | Apple M3, 96GB unified memory |
| Conversation duration | 15 seconds |
| Missed audio | 12 seconds (~80% loss) |
| Latency | 10 seconds |
| Effective throughput | ~3s processed in 15s (~5x slower than real-time) |

**Required:** Each frame must complete in **80ms** (12.5 Hz frame rate).
**Actual:** Each frame takes ~**400ms+**, causing catastrophic audio dropout.

---

## Executive Summary

The 5x slowdown on MPS is caused by **five compounding factors**, each of which
degrades performance significantly. Together they explain the full gap:

1. **No Flash Attention** -- SDPA falls back to naive O(n^2) math on MPS (~2-3x slower for attention)
2. **No CUDA Graphs** -- Full Python dispatch overhead every step (~1.5-2x slower)
3. **No torch.compile** -- Gating and RoPE kernels run in eager mode (~1.2-1.5x slower)
4. **bfloat16 limitations** -- MPS has incomplete bfloat16 support in PyTorch 2.4, potential silent float32 upcasting
5. **CPU synchronization points** -- `.item()` and `.cpu()` calls force pipeline stalls

The model architecture (32-layer, 4096-dim, 32-head main transformer + 6-layer depformer)
runs ~14B FLOPs per frame. On CUDA with all optimizations, this fits in the 80ms budget.
On MPS without any of them, it does not.

---

## Detailed Bottleneck Analysis

### 1. Scaled Dot-Product Attention -- No Flash Attention on MPS

**File:** `moshi/modules/transformer.py:441`
```python
x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
```

On CUDA, PyTorch dispatches to FlashAttention or Memory-Efficient Attention kernels
(fused CUDA kernels that are 2-5x faster than naive attention). On MPS in PyTorch 2.4,
`scaled_dot_product_attention` falls back to the **naive math implementation**:
- Materializes the full `[B, H, T, T]` attention matrix
- No kernel fusion
- Higher memory bandwidth consumption

Additionally, the attention uses a **boolean mask** (`attn_bias`) for causal masking
rather than setting `is_causal=True`, which prevents even the CUDA path from using
the most optimized kernels. On MPS this is doubly painful because boolean mask
indexing is slower.

**Impact:** ~2-3x slowdown on the attention computation (the dominant cost per layer).

### 2. CUDA Graphs Disabled

**File:** `moshi/models/lm.py:718`
```python
disable = lm_model.device.type != 'cuda'
graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
graphed_embeddings = CUDAGraphed(lm_model.forward_embeddings, disable=disable)
graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)
```

On CUDA, the entire forward pass (main transformer + depformer + sampling) is captured
as a CUDA Graph after warmup. This eliminates:
- Python interpreter overhead for every layer
- CPU-GPU kernel launch latency
- Framework dispatch overhead

On MPS, CUDA Graphs don't exist, and there is no MPS equivalent. Every inference step
pays the full Python dispatch cost for **every operation in every layer**. For a
32-layer transformer with gating, norm, attention, and projection per layer, this is
thousands of individual kernel launches per frame, each with Python overhead.

**Impact:** ~1.5-2x slowdown from dispatch overhead alone.

### 3. torch.compile Disabled on MPS

**File:** `moshi/modules/transformer.py:612-614`
```python
def forward(self, x: torch.Tensor):
    with ExitStack() as stack:
        if x.device.type != 'cuda':
            stack.enter_context(no_compile())
```

**File:** `moshi/utils/compile.py:62-75` -- `torch_compile_lazy` wrapper

On non-CUDA devices, the `no_compile()` context disables torch.compile. This affects
two key hot-path functions:

- **`apply_rope`** (`modules/rope.py:32`) -- Rotary Position Embeddings, called every layer.
  On CUDA this would be a fused compiled kernel; on MPS it's eager Python with multiple
  temporary tensor allocations.

- **`gating_forward_kernel`** (`modules/gating.py:33`) -- SiLU gating FFN, called every layer.
  This does `linear -> view -> activation * gate -> linear` which torch.compile fuses into
  fewer kernels on CUDA. On MPS: 4+ separate kernel launches per layer.

The inductor backend (used by torch.compile) does not support MPS in PyTorch 2.4.

**Impact:** ~1.2-1.5x slowdown, more for memory-bound operations.

### 4. bfloat16 Limitations on MPS

**File:** `moshi/models/loaders.py:170`
```python
dtype: torch.dtype = torch.bfloat16,
```

The model is loaded in bfloat16. MPS support for bfloat16 in PyTorch 2.4 is
**incomplete**. Many operations silently cast to float32, negating the memory
bandwidth advantage of half-precision. The operations affected include:

- Matrix multiplications (may run in float32 internally)
- RMS normalization (already forced to float32 at `loaders.py:107` via `rms_norm_f32`)
- Attention score computation
- Gating activation functions

**Impact:** Up to 2x memory bandwidth penalty if operations silently upcast.

### 5. CPU Synchronization Points

**File:** `moshi/server.py:233-235`
```python
main_pcm = main_pcm.cpu()           # Sync point: MPS -> CPU transfer
text_token = tokens[0, 0, 0].item() # Sync point: scalar read forces pipeline drain
```

Every frame forces two synchronization points:
- `.cpu()` to transfer decoded audio to CPU for Opus encoding
- `.item()` to read the text token ID for WebSocket transmission

Each sync point forces the MPS command queue to flush, waiting for all pending
GPU operations to complete before the CPU can read the result. This creates a
pipeline bubble where neither CPU nor GPU is doing useful work.

**Impact:** ~5-15ms per sync point, adds up to ~30ms overhead per frame.

### 6. Duplicate Mimi Codec Processing

**File:** `moshi/server.py:224-226, 231-232`
```python
codes = self.mimi.encode(chunk)
_ = self.other_mimi.encode(chunk)       # Redundant encode
main_pcm = self.mimi.decode(tokens[:, 1:9])
_ = self.other_mimi.decode(tokens[:, 1:9])  # Redundant decode
```

Two Mimi codec instances process the same data. Each Mimi model contains an 8-layer
SEANet encoder/decoder plus an 8-layer transformer. The second codec appears to maintain
streaming state for a secondary purpose but its output is discarded. This doubles the
audio codec compute cost.

**Impact:** ~2x audio codec cost (though the codec is a smaller fraction of total time).

### 7. Python-Level Loops in Hot Path

**File:** `moshi/modules/transformer.py:196-199`
```python
for t in range(T):
    y = F.linear(x[:, t], weight[t + offset])
    ys.append(y)
out = torch.stack(ys, 1)
```

The `multi_linear` function uses a Python for-loop. In streaming mode T=1, so this is
just one iteration -- not a major issue. However, the depformer's per-codebook loop
also uses Python loops during inference.

**Impact:** Minor per-step, but adds to dispatch overhead without CUDA Graphs.

---

## Optimisation Proposals (Ranked by Impact)

### Tier 1 -- High Impact

#### 1A. Port to MLX (Apple's ML Framework)

**Expected improvement: 3-5x**

MLX (https://github.com/ml-explore/mlx) is Apple's native ML framework, designed
specifically for Apple Silicon. It provides:
- Unified memory architecture awareness (no CPU/GPU copies)
- Lazy evaluation with automatic graph fusion (equivalent to CUDA Graphs)
- Native Metal shader compilation (equivalent to torch.compile + inductor)
- Efficient bfloat16 and float16 support on Apple Silicon
- Fused attention implementations

This would require rewriting the model in MLX, but the architecture is standard
(transformer + RoPE + SiLU gating) and MLX has all the primitives. The original
Moshi model has community MLX ports that could serve as a starting point. This is
the single highest-impact change possible.

**Key references:**
- mlx-community/moshi -- Community MLX ports of the base Moshi model
- MLX handles streaming/causal attention natively

#### 1B. Use float16 Instead of bfloat16

**Expected improvement: 1.5-2x**

MPS has much better float16 support than bfloat16 in PyTorch 2.4. Changing the dtype
from `torch.bfloat16` to `torch.float16` in `loaders.py:170` would:
- Enable native half-precision math on the M3's GPU cores
- Halve memory bandwidth vs. silent float32 upcasting
- Potentially enable faster SDPA paths

**Risk:** float16 has less dynamic range than bfloat16. May need loss scaling or
careful validation that model quality is preserved.

**Change:** Single line in `moshi/models/loaders.py:170`:
```python
dtype: torch.dtype = torch.float16,  # was torch.bfloat16
```

#### 1C. Eliminate CPU Sync Points

**Expected improvement: 1.2-1.5x**

Replace `.item()` and `.cpu()` calls with async alternatives:
- Use `torch.mps.synchronize()` less frequently
- Buffer multiple frames of audio before syncing to CPU
- Use `.to('cpu', non_blocking=True)` for async transfers
- Read text tokens via indexing into a pre-allocated CPU tensor

### Tier 2 -- Medium Impact

#### 2A. Optimized Attention for MPS

**Expected improvement: 1.5-2x on attention specifically**

Replace the generic `F.scaled_dot_product_attention` with an MPS-optimized path:
- Use `is_causal=True` instead of explicit boolean attention mask (allows PyTorch
  to select a more efficient kernel path)
- Implement a custom Metal kernel for causal attention
- Or use `xformers` if it supports MPS (currently unlikely)
- Consider reducing context window for real-time mode (context=3000 is very large)

#### 2B. Remove Duplicate Mimi Processing

**Expected improvement: ~10-15% overall**

The second Mimi model (`other_mimi`) has its output discarded. Remove the redundant
`other_mimi.encode()` and `other_mimi.decode()` calls in the server loop.

#### 2C. Reduce Model Precision / Quantize

**Expected improvement: 1.5-3x**

Apply quantization to reduce compute and memory bandwidth:
- **INT8 quantization** of linear layers (PyTorch supports this on MPS)
- **INT4 quantization** via GGML/llama.cpp style quantization
- Focus on the main transformer (32 layers) since it dominates compute

#### 2D. Upgrade PyTorch (2.5+)

**Expected improvement: 1.3-1.5x**

PyTorch 2.5+ has significantly improved MPS backend support. Upgrading
(requires loosening the `torch >= 2.2.0, < 2.5` constraint) could enable:
- Better MPS operator support
- Improved SDPA dispatch on MPS
- Experimental torch.compile MPS support

### Tier 3 -- Lower Impact / Experimental

#### 3A. Reduce Context Window

The main transformer uses `context=3000` which means the KV cache stores up to
3000 time steps. For real-time conversational use, a shorter context (e.g., 500-1000)
would reduce attention computation proportionally while still covering ~40-80 seconds
of conversation.

#### 3B. Operator Fusion via Custom Metal Shaders

Write fused Metal Performance Shaders for the hot path:
- Fused RMS Norm + Linear
- Fused attention (Q*K^T / sqrt(d) + mask + softmax + V)
- Fused SiLU gating

This is high effort but would approach CUDA-level performance on Apple Silicon.

#### 3C. Use `torch.compile` with MPS Backend (Experimental)

PyTorch nightly builds have experimental MPS support for `torch.compile`. Testing
with newer PyTorch versions could enable compilation without the full MLX rewrite.

#### 3D. Batch Audio Frames

Instead of processing one audio frame at a time in the streaming loop, batch 2-4
frames together. This increases latency but improves GPU utilization by processing
larger tensors.

---

## Dependency Replacements

| Current | Replacement | Benefit |
|---------|-------------|---------|
| PyTorch 2.4 (MPS) | **MLX** | Native Apple Silicon, 3-5x faster |
| PyTorch 2.4 (MPS) | **PyTorch 2.6+** | Better MPS backend, experimental compile support |
| `F.scaled_dot_product_attention` | Custom Metal attention | Flash-attention-equivalent on MPS |
| bfloat16 | **float16** | Native MPS support, no silent upcasting |
| Eager Python dispatch | **MLX lazy evaluation** | Automatic graph fusion, no dispatch overhead |
| `sphn` Opus codec (CPU) | Keep as-is | Audio codec is a minor fraction of total time |

---

## Recommended Action Plan

1. **Quick wins (minimal code changes):**
   - Switch to float16 (`loaders.py:170`)
   - Remove duplicate Mimi processing (`server.py:225, 232`)
   - Use `non_blocking=True` for CPU transfers (`server.py:233`)
   - Use `is_causal=True` in SDPA (requires refactoring attention mask)

2. **Medium effort (significant but bounded):**
   - Upgrade PyTorch to 2.5+ for better MPS support
   - Implement INT8 quantization for linear layers
   - Reduce context window for real-time mode

3. **High effort (maximum payoff):**
   - Port model to MLX framework
   - Write custom Metal attention kernel
   - Full model quantization pipeline

The most practical path to real-time performance on Apple Silicon is likely the
**MLX port** (option 1A), as it addresses all five root causes simultaneously.
The quick wins (float16 + removing sync points + removing duplicate Mimi) could
provide a ~2x improvement with minimal effort, potentially bringing the model
to ~2.5x slower than real-time -- still not usable, but a step in the right direction.
