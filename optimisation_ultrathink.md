# PersonaPlex/Moshi: MPS Performance Analysis & Optimization Report

## Observed Real-World Performance

| Metric | Value |
|--------|-------|
| Device | Apple M3, 96GB unified memory |
| Backend | PyTorch 2.4.1, MPS device |
| Conversation duration | 15 seconds |
| Missed audio | **12 seconds (~80% loss)** |
| Latency | **10 seconds** |
| Effective throughput | ~3s processed in 15s (**~5x slower than real-time**) |

**Required:** Each frame must complete in **80ms** (12.5 Hz frame rate).
**Actual:** Each frame takes ~**400ms+**, causing catastrophic audio dropout.

---

## Executive Summary

PersonaPlex/Moshi is a real-time streaming speech-to-speech language model designed
and optimized exclusively for NVIDIA CUDA GPUs. When running on Apple Silicon's MPS
(Metal Performance Shaders) backend, performance degrades catastrophically, producing
choppy, unusable audio. This is not caused by a single issue but by a **cascade of at
least ten major bottlenecks**, each of which would independently degrade performance.
Together, they make real-time inference impossible on MPS.

The root causes fall into three categories:

1. **CUDA-exclusive acceleration paths that are completely disabled on MPS** -- CUDA
   Graphs, `torch.compile`, and Flash Attention are the three pillars of the CUDA
   inference optimization. All three are either unavailable or degraded on MPS.

2. **Operations that are slow or unsupported on MPS** -- `index_copy_`, `torch.cdist`,
   boolean attention masks in `scaled_dot_product_attention`, `bfloat16` dtype, and
   `gather` operations all hit slow fallback paths on MPS.

3. **CPU synchronization points in the hot path** -- `.item()`, `.cpu()`, and
   `.numpy()` calls in the streaming loop force MPS to flush its command buffer and
   synchronize, destroying any pipelining benefits.

The model is a ~7B parameter LM (4096-dim, 32 layers, 32 heads) with a 1024-dim
6-layer Depformer, two Mimi codec instances (each with an 8-layer transformer), and
runs at 12.5 Hz frame rate. The latency budget is approximately **80ms per frame**.
On CUDA with all optimizations active, this budget is met. On MPS with all
optimizations disabled and multiple slow-path operations, total per-frame latency
likely exceeds 500-1000ms, making the system 6-12x too slow.

---

## Detailed Bottleneck Analysis

### 1. CUDA Graphs Completely Disabled on MPS

**Files:** `moshi/utils/compile.py:210-308`, `moshi/models/lm.py:718-722`, `moshi/models/compression.py:222-231`

**Severity:** CRITICAL (estimated 2-5x slowdown)

CUDA Graphs capture a sequence of GPU operations and replay them with near-zero CPU
overhead. The codebase uses `CUDAGraphed` wrappers around the three hottest functions:

```python
# lm.py:718-722
disable = lm_model.device.type != 'cuda'
graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
graphed_embeddings = CUDAGraphed(lm_model.forward_embeddings, disable=disable)
graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)
```

```python
# compression.py:222-231
disable = device.type != 'cuda'
graphed_tr_enc = CUDAGraphed(self.encoder_transformer, disable=disable)
graphed_tr_dec = CUDAGraphed(self.decoder_transformer, disable=disable)
```

When `disable=True` (any non-CUDA device), `CUDAGraphed.__call__` at line 244 falls
through to direct function invocation:

```python
# compile.py:244
if self.disable or not _is_cuda_graph_enabled() or in_cuda_graph():
    return self.func(*args, **kwargs)
```

Without CUDA Graphs, every inference step pays full Python dispatch overhead, kernel
launch latency, and driver overhead for every individual operation. For a 32-layer
transformer with attention + FFN per layer, this means hundreds of individual kernel
launches per step vs. a single graph replay.

MPS has no equivalent to CUDA Graphs. The Metal command buffer batching provides some
pipelining, but the granularity and overhead is significantly worse than CUDA Graphs.

---

### 2. torch.compile Disabled for Non-CUDA Devices

**Files:** `moshi/modules/transformer.py:611-614`, `moshi/utils/compile.py:46-75`, `moshi/modules/rope.py:32-33`, `moshi/modules/gating.py:33-34`, `moshi/modules/seanet.py:260-261,413-414`

**Severity:** CRITICAL (estimated 2-4x slowdown)

The transformer layer explicitly disables `torch.compile` when not on CUDA:

```python
# transformer.py:611-614
def forward(self, x: torch.Tensor):
    with ExitStack() as stack:
        if x.device.type != 'cuda':
            stack.enter_context(no_compile())
        x = self._sa_block(x)
        x = self._ff_block(x)
```

The `no_compile()` context manager sets a global flag that prevents `torch_compile_lazy`
from compiling:

```python
# compile.py:66-73
@wraps(fun)
def _wrapped(*args, **kwargs):
    nonlocal fun_compiled
    if _compile_disabled:
        return fun(*args, **kwargs)
    if fun_compiled is None:
        fun_compiled = torch.compile(fun)
    return fun_compiled(*args, **kwargs)
```

This disables compilation for:
- `apply_rope()` (rope.py:32) -- called every attention layer for RoPE
- `gating_forward_kernel()` (gating.py:33) -- called every FFN in every layer (SiLU gating)
- `SEANetEncoder.forward()` (seanet.py:260) -- Mimi encoder
- `SEANetDecoder.forward()` (seanet.py:413) -- Mimi decoder

Without `torch.compile`, every operation runs as individual eager-mode PyTorch ops.
The compiler would fuse operations, eliminate intermediate allocations, and optimize
memory access patterns. On MPS, `torch.compile` with the `inductor` backend does not
support MPS at all (as of PyTorch 2.4). Even with `aot_eager`, the benefit is minimal.

---

### 3. scaled_dot_product_attention with Boolean Mask (No Flash Attention)

**Files:** `moshi/modules/transformer.py:430-441`

**Severity:** HIGH (estimated 1.5-3x slowdown for attention)

The attention implementation uses `F.scaled_dot_product_attention`:

```python
# transformer.py:430-441
if self.causal:
    pos_k = pos_k.view(1, -1)
    pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
    delta = pos_q - pos_k
    attn_bias = (pos_k >= 0) & (delta >= 0)
    if self.context is not None:
        attn_bias = attn_bias & (delta < self.context)
else:
    attn_bias = None
x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
```

There are two compounding problems:

**a) No Flash Attention on MPS:** PyTorch's SDPA dispatches to FlashAttention-2 on
CUDA when conditions are met (no explicit mask, fp16/bf16, etc.). MPS does not
support Flash Attention. It falls back to the naive O(n^2) memory attention
implementation.

**b) Boolean attention mask forces slow path even on CUDA:** The code constructs a
custom boolean mask `attn_bias` for causal + context windowing. Even on CUDA, passing
an explicit `attn_mask` to SDPA disables Flash Attention and forces the "math" backend.
On MPS, this is even worse because the math backend itself is slower. The boolean mask
requires materializing the full [B, H, T, T] attention matrix.

For context=3000 (the LM setting from loaders.py:104), this means a 3000x3000 attention
matrix per head, per batch, per layer.

During streaming (T=1), the KV cache means the key/value dimensions are up to the cache
capacity (context=3000), so each attention step computes over [B, H, 1, capacity].

---

### 4. bfloat16 Dtype -- Limited MPS Support

**Files:** `moshi/models/loaders.py:170`, `moshi/modules/transformer.py:250`

**Severity:** HIGH (estimated 1.5-2x slowdown)

The LM model loads with `torch.bfloat16`:

```python
# loaders.py:170
dtype: torch.dtype = torch.bfloat16,
```

The KV cache defaults to `torch.bfloat16`:

```python
# transformer.py:250
dtype: torch.dtype = torch.bfloat16,
```

Apple Silicon's Neural Engine and GPU natively support `float16` but **not** `bfloat16`.
MPS handles `bfloat16` through software emulation or implicit casting to `float32`,
which:
- Doubles memory bandwidth usage (if promoted to float32)
- Loses the throughput advantage of half-precision compute
- May cause additional data conversion overhead on every operation

---

### 5. CPU Synchronization in the Hot Path

**Files:** `moshi/server.py:233-235`

**Severity:** HIGH (estimated latency spikes of 5-50ms each)

The streaming loop at the core of audio processing forces three CPU sync points
every frame:

```python
# server.py:233-235
main_pcm = main_pcm.cpu()                          # Sync point 1
opus_writer.append_pcm(main_pcm[0, 0].numpy())     # Sync point 2
text_token = tokens[0, 0, 0].item()                # Sync point 3
```

On CUDA, `.cpu()` with CUDA Graphs active can be overlapped because the graph captures
the transfer. On MPS, `.cpu()` triggers a full command buffer flush and synchronization.
`.item()` similarly blocks until the MPS command queue drains.

At 12.5 Hz, this happens 12.5 times per second. Each sync point blocks the Python
thread until all pending MPS operations complete, destroying the ability to pipeline
GPU work with CPU processing.

---

### 6. index_copy_ Workaround for MPS KV Cache

**Files:** `moshi/modules/transformer.py:266-273`

**Severity:** MODERATE (contributes to overall slowdown)

The KV cache update already has an MPS-specific workaround:

```python
# transformer.py:266-273
indexes = torch.arange(T, device=self.end_offset.device, ...) + self.end_offset
indexes = indexes % self.capacity
if k.device.type == "mps":
    self.cache[0][:, :, indexes, :] = k
    self.cache[1][:, :, indexes, :] = v
else:
    self.cache[0].index_copy_(2, indexes, k)
    self.cache[1].index_copy_(2, indexes, v)
```

`index_copy_` is known to be unsupported or buggy on MPS. The fallback uses advanced
indexing which creates temporary tensors and may involve implicit CPU synchronization
for index computation. This runs at every attention layer (32 main + 6 depformer = 38
layers total), for every time step.

---

### 7. torch.cdist in Vector Quantization

**Files:** `moshi/quantization/core_vq.py:183`

**Severity:** MODERATE (affects encode/decode path)

The codebook quantization uses `torch.cdist`:

```python
# core_vq.py:183
dists = torch.cdist(x[None], self.embedding[None], p=2)[0]
```

`torch.cdist` computes pairwise Euclidean distances. On CUDA, this dispatches to
optimized BLAS routines. On MPS, `cdist` may fall back to a naive implementation or
CPU execution. With codebook_size=2048 and dim=256, each quantization step computes
N*2048 distances. The Mimi codec has 8 codebook layers in a residual scheme, so this
runs 8 times per encode and 8 times per decode.

---

### 8. gather Operations

**Files:** `moshi/models/lm.py:950`, `moshi/utils/sampling.py:83,102`

**Severity:** MODERATE

The output collection uses `gather`:

```python
# lm.py:950
out = state.cache.gather(dim=2, index=index)
```

And sampling uses `gather`:

```python
# sampling.py:83
next_token = indices.gather(-1, next_token)
```

`gather` on MPS is supported but can be slower than on CUDA, especially with
non-contiguous tensors. The cache gather at line 950 operates on the full
[B, num_codebooks, max_delay+3] cache tensor.

---

### 9. RMS Norm with Float32 Promotion

**Files:** `moshi/modules/transformer.py:55-67`, `moshi/models/loaders.py:107`

**Severity:** LOW-MODERATE

The model uses `rms_norm_f32` normalization (loaders.py:107):

```python
# transformer.py:55-67
def _rms_norm(x, alpha, dtype, eps):
    x_dtype = x.dtype
    if dtype is not None:
        x = x.to(dtype)            # Cast to float32
    var = eps + torch.mean(x**2, dim=2, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)  # Cast back
    return y
```

Every layer applies this norm twice (before attention and before FFN), and each call
casts the full activation tensor from bfloat16 to float32 and back. On MPS where
bfloat16 is already slow, this doubles the conversion overhead. With 32 main layers
and 6 depformer layers, that is 76 float32 promotions per inference step.

---

### 10. Depformer Per-Step Sequential Processing

**Files:** `moshi/models/lm.py:1129-1177`

**Severity:** MODERATE (structural, difficult to optimize)

The Depformer generates 8 (dep_q=8 during inference, though configured as 16) audio
codebook tokens **sequentially**, one at a time:

```python
# lm.py:1142-1164
with lm_model.depformer.streaming(B):
    for cb_index in range(lm_model.dep_q):
        input_ = prev_token[:, None, None]
        logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
        next_token = sample_token(logits.float(), ...)
        prev_token = torch.where(...)
        depformer_tokens.append(next_token)
```

Each iteration involves:
1. A full forward pass through the 6-layer Depformer transformer
2. A linear projection to get logits
3. Softmax + top-k sampling

This cannot be parallelized because each token depends on the previous one. On CUDA
with CUDA Graphs, the 8 sequential steps are captured as one graph. On MPS, they
execute as 8 x (6 layers x ~10 ops) = ~480 individual kernel launches.

---

### 11. Dual Mimi Instances

**Files:** `moshi/server.py:121-129, 224-232`

**Severity:** MODERATE

The server maintains **two** Mimi codec instances (`mimi` and `other_mimi`), both of
which encode and decode every frame:

```python
# server.py:224-232
codes = self.mimi.encode(chunk)
_ = self.other_mimi.encode(chunk)           # Second encode (discarded!)
...
main_pcm = self.mimi.decode(tokens[:, 1:9])
_ = self.other_mimi.decode(tokens[:, 1:9])  # Second decode (discarded!)
```

Each Mimi instance contains:
- SEANet encoder (multiple conv layers + residual blocks)
- 8-layer transformer encoder
- SEANet decoder
- 8-layer transformer decoder
- 8-level residual VQ

The second instance appears to be for maintaining streaming state for a secondary
audio stream. This doubles the codec compute cost.

---

### 12. Streaming Loop Tight Polling

**Files:** `moshi/server.py:204-243`

**Severity:** LOW-MODERATE

The opus processing loop polls with `await asyncio.sleep(0.001)`:

```python
# server.py:210
await asyncio.sleep(0.001)
```

This creates a 1ms polling interval, which on MPS may interact poorly with the Metal
command buffer scheduling, as MPS submissions happen asynchronously and the tight
polling can cause unnecessary context switches.

---

## Concrete Optimization Proposals

Ranked by likely impact (highest first):

### Tier 1: Critical (Would each provide 2-5x improvement)

#### 1. Port to MLX (Apple's Metal-native ML framework)
**Impact: 5-10x improvement**
**Effort: Very High**

MLX is Apple's framework specifically designed for Apple Silicon. It provides:
- Native Metal GPU acceleration without the PyTorch MPS translation layer
- Unified memory architecture awareness (no CPU/GPU copies needed)
- Lazy evaluation (automatic operation fusion, similar to CUDA Graphs)
- Native `float16` support (no bfloat16 issues)
- Built-in Flash Attention equivalent for Metal

The model architecture (transformer + streaming convolutions) maps well to MLX.
The Kyutai team already ships a Rust/Metal implementation for their original Moshi,
which proves Metal can handle this workload.

**Steps:**
- Rewrite the core transformer, attention, and codec in MLX
- Use `mlx.core.compile` for graph-level optimization
- Use unified memory to eliminate CPU<->GPU transfers

#### 2. Convert to float16 instead of bfloat16
**Impact: 1.5-2x improvement**
**Effort: Low**

Change `loaders.py:170` from `torch.bfloat16` to `torch.float16`. Apple Silicon
has native float16 ALUs and the Neural Engine operates in float16. This alone would
eliminate the bfloat16 emulation overhead.

**Risk:** Possible numerical differences. float16 has less dynamic range than bfloat16.
May need to keep certain operations (like RMS norm) in float32 for stability.

**Change:** In `moshi/models/loaders.py:170`:
```python
dtype: torch.dtype = torch.float16,  # was torch.bfloat16
```
Also update `moshi/modules/transformer.py:250`:
```python
dtype: torch.dtype = torch.float16,  # was torch.bfloat16
```

#### 3. Eliminate CPU Sync Points in Streaming Loop
**Impact: Reduces latency spikes by 10-50ms per frame**
**Effort: Low-Medium**

Batch the CPU transfers and minimize `.item()` calls:

In `server.py:233-235`, the `.cpu()`, `.numpy()`, and `.item()` calls can be
restructured:
- Use `torch.mps.synchronize()` once per frame rather than implicit syncs
- Pre-allocate a CPU tensor for PCM output and use async copy
- Cache the text token retrieval to avoid per-step `.item()` calls

### Tier 2: High Impact (Would each provide 1.5-3x improvement)

#### 4. Replace Boolean Attention Mask with Causal Flag
**Impact: 1.5-3x improvement on attention**
**Effort: Medium**

The current code builds a custom boolean mask for causal attention with context
windowing. For streaming mode (T=1), the mask is trivial -- all past positions within
the context window are valid.

For `F.scaled_dot_product_attention`, passing `is_causal=True` (without an explicit
`attn_mask`) allows PyTorch to use the most efficient implementation available.

For context windowing, the RingKVCache already handles limiting to the context
window by its capacity. During streaming (T=1), the causal constraint is automatically
satisfied since we are only querying the current position against past positions.

The mask construction at transformer.py:430-438 could be simplified or eliminated
for the streaming case.

#### 5. Pre-compute and Cache Attention Masks
**Impact: Reduces per-step allocation overhead**
**Effort: Low**

If the boolean mask cannot be eliminated, pre-allocate it once at KV cache
initialization instead of constructing it fresh every step. The mask at
transformer.py:430-438 creates multiple temporary tensors every forward pass:
`pos_k`, `pos_q`, `delta`, and the boolean operations all allocate new tensors.

#### 6. Use MPS-compatible torch.compile Backend
**Impact: 1.5-2x improvement**
**Effort: Medium**

While `inductor` does not support MPS, the `aot_autograd` backend with the `aot_eager`
compiler can provide some operator fusion on MPS. Additionally, PyTorch 2.5+
has improved MPS compilation support.

Remove the blanket `no_compile()` in transformer.py:613-614 and instead set:
```python
torch._dynamo.config.suppress_errors = True
```
to allow compilation where possible and gracefully fall back where not.

### Tier 3: Moderate Impact

#### 7. Optimize Vector Quantization for MPS
**Impact: 1.2-1.5x improvement on codec path**
**Effort: Medium**

Replace `torch.cdist` in core_vq.py:183 with an explicit matmul-based distance
computation that is better optimized on MPS:

```python
# Instead of torch.cdist:
# dists = torch.cdist(x[None], self.embedding[None], p=2)[0]
# Use:
# ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
x_sq = (x ** 2).sum(dim=-1, keepdim=True)
e_sq = (self.embedding ** 2).sum(dim=-1).unsqueeze(0)
dists = x_sq + e_sq - 2 * torch.mm(x, self.embedding.t())
```

Matrix multiplication is one of the best-optimized operations on MPS.

#### 8. Reduce Model Size or Use Quantization
**Impact: 2-4x improvement (model-dependent)**
**Effort: High**

The 7B parameter model is large for Apple Silicon. Consider:
- **4-bit quantization** using `bitsandbytes` or GPTQ -- would reduce memory
  bandwidth requirements by 4x and fit easily in Apple Silicon unified memory
- **Distillation** to a smaller model (e.g., 1-3B parameters)
- **Pruning** of transformer layers or heads

#### 9. Remove Duplicate Mimi Instance
**Impact: ~30-40% reduction in codec compute**
**Effort: Low**

The `other_mimi` encode/decode results are discarded in the streaming loop
(server.py:225,232). If it is only maintaining streaming state for a secondary
purpose, consider whether it can be eliminated or deferred.

#### 10. MPS Command Buffer Optimization
**Impact: 10-20% improvement**
**Effort: Low**

Add explicit MPS synchronization control:
```python
torch.mps.synchronize()  # Once per frame instead of implicit syncs
```

Use `torch.mps.set_per_process_memory_fraction()` to ensure the model fits
in the MPS memory pool without thrashing.

### Tier 4: Lower Impact / Longer Term

#### 11. Use coremltools to Convert to Core ML
**Impact: Potentially 3-5x for the transformer**
**Effort: Very High**

Apple's Core ML runtime can use the Neural Engine, which provides significantly
higher throughput than the GPU for transformer workloads. The trade-off is that
Core ML has limited support for dynamic shapes and streaming state.

#### 12. Custom Metal Shaders for Hot Paths
**Impact: 1.5-2x for specific operations**
**Effort: Very High**

Write custom Metal compute shaders for:
- Fused RMS norm (avoid float32 promotion roundtrip)
- Fused RoPE application
- Fused attention with causal masking

---

## Dependency Alternatives for Apple Silicon

| Current | Alternative | Benefit |
|---------|------------|---------|
| PyTorch MPS | MLX | Native Metal, lazy eval, unified memory |
| torch.compile (inductor) | mlx.core.compile | Graph-level optimization for Metal |
| CUDA Graphs | MLX lazy evaluation | Automatic operation batching |
| F.scaled_dot_product_attention | MLX flash_attention | Metal-optimized attention |
| bfloat16 | float16 | Native Apple Silicon support |
| torch.cdist | matmul-based distance | Better MPS optimization |
| safetensors | mlx.core.load | Direct Metal buffer loading |

---

## Summary Table of Bottlenecks

| # | Bottleneck | File:Line | Severity | Fix Effort |
|---|-----------|-----------|----------|------------|
| 1 | CUDA Graphs disabled | compile.py:244, lm.py:718 | CRITICAL | Cannot fix on MPS; need MLX |
| 2 | torch.compile disabled | transformer.py:613, compile.py:69 | CRITICAL | Medium (try aot_eager) |
| 3 | No Flash Attention + boolean mask | transformer.py:441 | HIGH | Medium (simplify mask) |
| 4 | bfloat16 on MPS | loaders.py:170 | HIGH | Low (change to float16) |
| 5 | CPU sync in hot path | server.py:233-235 | HIGH | Low (batch transfers) |
| 6 | index_copy_ workaround | transformer.py:268-269 | MODERATE | Already worked around |
| 7 | torch.cdist in VQ | core_vq.py:183 | MODERATE | Medium (matmul-based) |
| 8 | gather operations | lm.py:950 | MODERATE | Low (profile first) |
| 9 | RMS norm float32 promotion | transformer.py:55-67 | LOW-MOD | Medium (fused kernel) |
| 10 | Sequential Depformer | lm.py:1142-1164 | MODERATE | Structural; hard to fix |
| 11 | Dual Mimi instances | server.py:224-232 | MODERATE | Low (remove if possible) |
| 12 | Tight polling loop | server.py:210 | LOW-MOD | Low (increase interval) |

---

## Recommended Action Plan

**For immediate improvement (hours of work):**
1. Switch dtype from bfloat16 to float16
2. Remove/batch CPU sync points in the streaming loop
3. Remove the second Mimi instance if not needed
4. Increase polling interval from 1ms to 5-10ms

**For significant improvement (days of work):**
5. Simplify/eliminate the boolean attention mask for streaming mode
6. Replace torch.cdist with matmul-based distance
7. Experiment with torch.compile aot_eager backend on MPS

**For production-quality MPS support (weeks of work):**
8. Port core inference to MLX
9. Implement 4-bit quantization for the 7B LM
10. Use Core ML for the transformer backbone

**Expected combined improvement from items 1-7: 3-6x**, which may bring the system
close to real-time on high-end Apple Silicon (M2 Pro/Max/Ultra, M3 Pro/Max/Ultra).

**Expected improvement from MLX port: 5-10x**, which would likely achieve real-time
on M2 Pro or better.
