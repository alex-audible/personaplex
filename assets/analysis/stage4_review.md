# Stage 4 Review: MLX Real-Time WebSocket Server

## Review Scope

**Files reviewed:**
- `moshi/moshi_mlx/server_mlx.py` (MLX server under review)
- `moshi/moshi/server.py` (PyTorch reference server)
- `moshi/moshi_mlx/lm_gen_mlx.py` (PersonaPlexLmGen wrapper)
- `moshi/moshi_mlx/loaders_mlx.py` (MLX model loader)
- `.venv/.../moshi_mlx/models/lm.py` (Kyutai MLX LM model)
- `.venv/.../moshi_mlx/models/generate.py` (Kyutai MLX LmGen)
- `.venv/.../moshi_mlx/utils/sampling.py` (MLX sampling)
- `moshi/moshi/models/lm.py` (PyTorch LMGen reference)
- `optimisation.md`, `optimisation_ultrathink.md`, `mlxport.md`

**Reviewer model:** Claude Opus 4.5

---

## 1. Critical Issues

These are bugs or correctness problems that must be fixed before the server can
be considered functional.

### 1.1 CRITICAL: `is_alive()` consumes WebSocket messages, starving `recv_loop`

**File:** `server_mlx.py:387-404`

The `is_alive()` function calls `ws.receive()` with a 0.01s timeout. If a binary
audio message arrives during this window, `is_alive()` will consume it and discard
it -- the message is never passed to `recv_loop`'s `async for message in ws:`
iterator. This silently drops user audio frames.

The PyTorch server has the **exact same bug** (server.py:268-281), but that does
not excuse it in the MLX port. The `is_alive()` check is only called once (before
the handshake), which limits the damage, but if the client sends data before the
handshake completes, it will be lost.

**Severity:** HIGH. Audio messages received during system prompt processing will
be consumed and silently dropped. In practice this is mitigated because
`is_alive()` is only called once and system prompts are processed before the
client starts streaming, but the function is architecturally broken.

**Recommendation:** Replace the `ws.receive()` probe with a check on `ws.closed`
and the `close` flag only. If connection-alive detection is truly needed, use
`ws._writer` or transport-level checks that do not consume from the message queue.

### 1.2 CRITICAL: `step_system_prompts` runs synchronously in `asyncio.to_thread` but PyTorch server uses async with `is_alive` checks

**File:** `server_mlx.py:408-412`

```python
await asyncio.to_thread(
    self.lm_gen.step_system_prompts,
    audio_tokenizer=self.mimi,
    text_token_ids=text_prompt_tokens,
)
```

The PyTorch server calls `step_system_prompts_async(self.mimi, is_alive=is_alive)`,
which is an async method that checks `is_alive()` between every system prompt step.
This allows the server to detect client disconnection during the (potentially long)
system prompt phase and abort early.

The MLX server wraps the synchronous `step_system_prompts` in `asyncio.to_thread`,
which runs it in a thread pool. This means:

1. **No disconnect detection during system prompts.** If the client disconnects
   while voice prompt embeddings are being replayed (which can take many seconds
   for long voice prompts), the server will finish all system prompt steps before
   detecting the disconnect. This wastes compute and delays availability for the
   next connection (due to the asyncio lock).

2. **Thread safety concern.** `self.mimi` (rustymimi Tokenizer) is used inside the
   thread while the event loop continues on the main thread. While the asyncio lock
   prevents concurrent connections, the event loop itself may access `ws` state
   concurrently with the threaded system prompt processing.

**Severity:** HIGH. Long voice prompts (e.g., 10+ seconds of audio) will block the
server even after client disconnection.

**Recommendation:** Convert `step_system_prompts` to an async method that yields
control periodically and checks `is_alive()`, matching the PyTorch server's
`step_system_prompts_async` pattern.

### 1.3 CRITICAL: No `text_prompt_tokens` attribute set on `lm_gen` -- text prompt injection relies on different mechanism

**File:** `server_mlx.py:239-244` vs `server.py:170`

PyTorch server:
```python
self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(
    wrap_with_system_tags(request.query["text_prompt"])
) if len(request.query["text_prompt"]) > 0 else None
```

The PyTorch server sets `self.lm_gen.text_prompt_tokens` as an attribute on the
LMGen object, and `_step_text_prompt_core()` reads it:
```python
for text_prompt_token in self.text_prompt_tokens:
```

The MLX server instead passes `text_token_ids` as a parameter to
`step_system_prompts()`. This is architecturally fine IF
`PersonaPlexLmGen.step_system_prompts()` correctly handles the `text_token_ids`
parameter. Checking `lm_gen_mlx.py:594-646`, the method does accept
`text_token_ids` and passes it to `step_text_prompt()` -- this is correct.

**Verdict:** NOT a bug. The MLX port uses parameter passing instead of attribute
mutation. This is actually cleaner. Downgraded from critical.

### 1.4 CRITICAL: Warmup loop does not iterate over codebook time steps

**File:** `server_mlx.py:156-184` vs `server.py:119-133`

PyTorch warmup:
```python
codes = self.mimi.encode(chunk)
for c in range(codes.shape[-1]):
    tokens = self.lm_gen.step(codes[:, :, c: c + 1])
```

The PyTorch server encodes a chunk and then iterates over every time step in
`codes.shape[-1]` (which may be >1 depending on the Mimi encoder's compression
ratio).

MLX warmup:
```python
codes = self.mimi.encode_step(chunk)
codes_mx = mx.array(codes_np).transpose(0, 2, 1)[:, :8, :]
tokens = self.lm_gen.step(codes_mx)
```

The MLX server calls `encode_step` (streaming single-step) and directly passes one
frame. With `rustymimi.encode_step()`, the output shape is `[1, 1, num_codebooks]`
which after transpose becomes `[1, 8, 1]` -- effectively one time step. This is
likely correct for the streaming case since `encode_step` returns exactly one frame.

However, there is a subtle difference: the PyTorch server uses `mimi.encode(chunk)`
which is a non-streaming call that may produce multiple time steps. The MLX server
uses `encode_step` (streaming) which produces exactly one step. For warmup purposes,
this is acceptable since the goal is just to prime caches.

**Verdict:** Low risk. The warmup achieves its purpose (priming MLX compilation
caches and allocating memory). The different encode API does not affect correctness.

### 1.5 CRITICAL: Missing `lm_gen.streaming_forever()` equivalent -- LMGen state not initialized

**File:** `server_mlx.py:123-143` vs `server.py:106-117`

PyTorch server initialization:
```python
self.lm_gen = LMGen(lm, audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate), ...)
self.mimi.streaming_forever(1)
self.other_mimi.streaming_forever(1)
self.lm_gen.streaming_forever(1)
```

The PyTorch server calls `streaming_forever(1)` on the LMGen, which calls
`_start_streaming(batch_size)` which in turn calls `_init_streaming_state(1)`.
This creates the `_LMGenState` containing the cache, provided, initial tensors,
and CUDA graph wrappers. Without this, the LMGen has no state and `step()` would
fail.

The MLX server does NOT call any equivalent initialization. Looking at
`PersonaPlexLmGen.__init__` in `lm_gen_mlx.py:109-147`, it creates the inner
`LmGen` (Kyutai's version). The Kyutai `LmGen.__init__`
(`.venv/.../models/generate.py:14-46`) does create `gen_sequence` and
`step_idx` directly in `__init__`, so no separate streaming initialization is
needed.

**Verdict:** NOT a bug. The Kyutai MLX LmGen initializes state in its constructor,
unlike the PyTorch LMGen which uses a streaming context. The MLX architecture is
different by design.

### 1.6 CRITICAL: Reset logic is fragile and reaches into internal Kyutai LmGen state

**File:** `server_mlx.py:376-385`

```python
for c in self.lm_gen.model.transformer_cache:
    c.reset()
for c in self.lm_gen.model.depformer_cache:
    c.reset()
self.lm_gen.lm_gen.step_idx = 0
self.lm_gen.lm_gen.gen_sequence = mx.zeros_like(
    self.lm_gen.lm_gen.gen_sequence
)
mx.eval(self.lm_gen.lm_gen.gen_sequence)
```

This reset logic directly manipulates internal state of the Kyutai LmGen. Issues:

1. **Fragile coupling.** Accessing `self.lm_gen.lm_gen.step_idx` and
   `self.lm_gen.lm_gen.gen_sequence` bypasses the PersonaPlexLmGen API. If the
   inner LmGen implementation changes, this breaks silently.

2. **Incomplete reset.** The Kyutai `LmGen.__init__` sets several fields:
   - `gen_sequence` (reset: YES)
   - `step_idx` (reset: YES)
   - `audio_padding_token` (not mutated, OK)
   - `audio_delays` (not mutated, OK)
   - `max_delay` (not mutated, OK)
   - `main_codebooks` (not mutated, OK)
   - `cfg_coef` (not mutated, OK)

   However, the gen_sequence is zeroed rather than filled with the `ungenerated_token`
   value (-2). The Kyutai LmGen initializes with:
   ```python
   self.gen_sequence = mx.full(
       shape=(...), vals=self.ungenerated_token, dtype=mx.int32,
   )
   ```
   where `ungenerated_token = -2`. Zeroing it instead of filling with -2 may cause
   the check `(tokens == self.ungenerated_token).any()` to miss legitimately
   ungenerated positions, or worse, treat 0-tokens as valid when they should be
   flagged as ungenerated.

3. **PersonaPlexLmGen state not reset.** The voice_prompt, voice_prompt_embeddings,
   voice_prompt_cache, and voice_prompt_audio fields on PersonaPlexLmGen are NOT
   reset. This is intentional (voice prompt persists across connections) but should
   be documented.

**Severity:** HIGH. The gen_sequence initialization with zeros instead of -2 could
cause incorrect behavior during the delay buffer fill phase.

**Recommendation:** Replace `mx.zeros_like(...)` with:
```python
mx.full(self.lm_gen.lm_gen.gen_sequence.shape,
        vals=self.lm_gen.lm_gen.ungenerated_token, dtype=mx.int32)
```

Or better, add a `reset()` method to PersonaPlexLmGen that properly delegates to
the inner LmGen.

### 1.7 CRITICAL: `opus_loop` does not match PyTorch's per-codebook-step iteration

**File:** `server_mlx.py:307-328` vs `server.py:226-232`

PyTorch opus_loop:
```python
codes = self.mimi.encode(chunk)
_ = self.other_mimi.encode(chunk)
for c in range(codes.shape[-1]):
    tokens = self.lm_gen.step(codes[:, :, c: c + 1])
    if tokens is None:
        continue
    main_pcm = self.mimi.decode(tokens[:, 1:9])
    _ = self.other_mimi.decode(tokens[:, 1:9])
```

The PyTorch server iterates over `codes.shape[-1]` time steps, calling `step()` for
each. With PyTorch Mimi's `encode()`, multiple time steps can be returned per
frame.

MLX opus_loop:
```python
codes = self.mimi.encode_step(chunk_input)
_ = self.other_mimi.encode_step(chunk_input)
codes_mx = mx.array(codes_np).transpose(0, 2, 1)[:, :8, :]
tokens = self.lm_gen.step(codes_mx)
```

The MLX server uses `encode_step` which returns exactly one time step per call.
There is no loop over codebook time steps.

**Analysis:** With rustymimi's `encode_step`, the output is always a single time
step `[1, 1, num_codebooks]`. This maps to exactly one LM step per audio frame.
The PyTorch Mimi's `encode()` call with streaming mode active also typically
returns one time step per frame-sized input. So this difference is benign in
practice.

**Verdict:** Likely correct but fragile. If rustymimi's encode_step ever returns
multiple time steps, the MLX server would silently ignore them. The PyTorch server's
loop handles this generically.

---

## 2. Missing Features

### 2.1 Missing: `--device` and `--cpu-offload` CLI arguments (expected)

The PyTorch server has `--device` and `--cpu-offload` flags. These are
CUDA/PyTorch-specific and correctly omitted from the MLX server, which always
uses MLX's unified memory.

**Verdict:** Intentionally omitted, correct.

### 2.2 Missing: `save_voice_prompt_embeddings` feature

The PyTorch `LMGen.__init__` accepts `save_voice_prompt_embeddings=False`, and
when enabled, it saves voice prompt embeddings to disk as `.pt` files during the
voice prompt stepping phase. The PyTorch server passes this to LMGen.

The MLX `PersonaPlexLmGen` does not support this feature. The
`_step_voice_prompt_audio` method does not capture or save embeddings.

**Impact:** LOW. This is a development/debugging feature for pre-computing voice
prompt caches. It is not needed for production serving.

### 2.3 Missing: `report_loss` and `return_logits` features

The PyTorch LMGen supports `report_loss` and `return_logits` for debugging.
These are not present in PersonaPlexLmGen.

**Impact:** NONE for production. Debug-only features.

### 2.4 Missing: Dynamic sampling parameter update from query params

The PyTorch server has commented-out lines (server.py:143-146) for reading
`audio_temperature`, `text_temperature`, `text_topk`, `audio_topk` from query
parameters. Since they are commented out in the reference, this is not a missing
feature.

**Verdict:** Not missing (commented out in reference too).

### 2.5 Missing: `torch.no_grad()` wrapper around `main()`

The PyTorch server wraps `main()` in `torch.no_grad()` (server.py:482-483):
```python
with torch.no_grad():
    main()
```

MLX does not have autograd enabled by default for inference, so no equivalent is
needed.

**Verdict:** Correctly omitted for MLX.

### 2.6 Missing: `host` parameter not passed to `web.run_app`

**File:** `server_mlx.py:689` vs `server.py:479`

MLX server:
```python
web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
```

PyTorch server:
```python
web.run_app(app, port=args.port, ssl_context=ssl_context)
```

Interestingly, the MLX server correctly passes `host=args.host` while the PyTorch
server does not. The PyTorch server defaults to binding on all interfaces.

**Verdict:** The MLX server is actually BETTER here. It respects the `--host` flag.

### 2.7 Missing: Async system prompts with `is_alive` disconnect detection

As described in Critical Issue 1.2, the MLX server uses synchronous system prompt
stepping without disconnect detection. The PyTorch server's async variant with
`is_alive` callbacks allows early termination if the client disconnects during the
potentially lengthy system prompt phase.

**Impact:** HIGH for user experience. If a client connects and disconnects quickly,
the server may still spend many seconds processing system prompts.

### 2.8 Missing: `lm_gen.streaming_forever(1)` equivalent reset mechanism

The PyTorch server calls `self.lm_gen.reset_streaming()` (server.py:267) which
recursively resets all streaming state including the LMGen's internal
`_LMGenState`, all transformer layer KV caches, and all depformer KV caches.

The MLX server manually resets caches (server_mlx.py:376-385). This is functionally
equivalent but more error-prone (see Critical Issue 1.6).

**Impact:** MEDIUM. The manual reset works but is fragile.

---

## 3. Improvements

### 3.1 Error handling: Missing try/except around WebSocket operations in opus_loop

**File:** `server_mlx.py:280-344`

The `opus_loop` function does not have any exception handling. If the LM step
raises an exception (e.g., out of memory, shape mismatch, numerical error), the
entire loop crashes without logging.

The PyTorch server has the same issue, but the MLX server should improve on it.

**Recommendation:** Wrap the inner loop body in a try/except that logs errors and
sets `close = True`.

### 3.2 Error handling: Missing try/except around the `handle_chat` method

If `handle_chat` raises an exception (e.g., `FileNotFoundError` from voice prompt
loading at line 221), it propagates as an unhandled exception to aiohttp, which
returns a 500 error. This is acceptable but not graceful.

**Recommendation:** Wrap the body in try/except, log the error, and return a
proper WebSocket close with an error code.

### 3.3 Voice prompt path handling: Defensive check for `None` voice_prompt_dir

**File:** `server_mlx.py:213-225`

The MLX server uses `request.query.get("voice_prompt", "")` which safely defaults
to empty string. The PyTorch server uses `request.query["voice_prompt"]` which
raises KeyError if the query param is missing.

**Verdict:** The MLX server is MORE robust here.

### 3.4 Voice prompt loading guard: Always guarded by `if voice_prompt_path:`

**File:** `server_mlx.py:228-236`

The MLX server only attempts voice prompt loading if `voice_prompt_path` is truthy.
The PyTorch server (server.py:164-169) unconditionally enters the voice prompt
comparison block, which could crash if `voice_prompt_path` is None (accessing
`.endswith('.pt')` on None would TypeError).

**Verdict:** The MLX server is MORE robust here.

### 3.5 Seed handling: Safer default

**File:** `server_mlx.py:246`

```python
seed = int(request.query.get("seed", -1))
```

The MLX server safely defaults to -1 and checks `seed != -1` before seeding.
The PyTorch server uses `int(request["seed"]) if "seed" in request.query else None`
which is more verbose.

**Verdict:** Equivalent safety, MLX version is cleaner.

### 3.6 Code quality: Duplicate warmup log message

**File:** `server_mlx.py:155` and `server_mlx.py:660`

The warmup logs "Warming up the model" inside `warmup()` (line 155) and again
in `main()` (line 660). This results in two log messages.

**Recommendation:** Remove one of the duplicate log lines.

### 3.7 Code quality: `text_prompt_str` variable scope

**File:** `server_mlx.py:239-244`

`text_prompt_str` is used both for tokenization and for logging (line 357). If
the client sends no `text_prompt` query param, this defaults to empty string,
which is safe.

**Verdict:** Acceptable.

### 3.8 Improvement: Consider using `ConnectionError` in is_alive exception handler

**File:** `server_mlx.py:401-403`

The MLX server catches a bare `Exception` as fallback. The PyTorch server
specifically catches `aiohttp.ClientConnectionError` (server.py:279-280).

**Recommendation:** Add `aiohttp.ClientConnectionError` as a specific except
clause before the generic Exception catch.

### 3.9 Improvement: The `decode_step` output transpose may not match expected format

**File:** `server_mlx.py:329-330`

```python
agent_audio = tokens[:, 1:, :]  # [1, 8, 1]
agent_audio_np = np.array(agent_audio).astype(np.uint32)
```

The agent audio tokens are cast to `np.uint32` before passing to
`self.mimi.decode_step()`. The rustymimi API expects uint32 tokens, which is
correct. However, the tokens come from the LmGen as int32, and negative token
values (which should not occur after proper generation but could during the
delay fill phase) would wrap to very large uint32 values.

**Recommendation:** Add an assertion or clamp to verify token values are
non-negative before the uint32 cast.

### 3.10 Improvement: Missing timing/performance instrumentation

The PyTorch server has no built-in performance instrumentation either, but given
the MLX port's purpose (achieving real-time on Apple Silicon), adding PerfLogger
integration (as specified in mlxport.md Stage 5) would be valuable.

---

## 4. Optimization Roadmap

Prioritized list of optimizations, considering the MLX port is now functionally
complete (with fixes from Section 1).

### Priority 1: Correctness Fixes (Must Do)

| # | Fix | Impact | Effort |
|---|-----|--------|--------|
| P1.1 | Fix gen_sequence reset to use ungenerated_token (-2) instead of zeros | Correctness | Low |
| P1.2 | Add async system prompt stepping with is_alive detection | Robustness | Medium |
| P1.3 | Add error handling in opus_loop and handle_chat | Robustness | Low |

### Priority 2: Quick Performance Wins (Do Next)

| # | Optimization | Expected Impact | Effort |
|---|-------------|----------------|--------|
| P2.1 | Wrap `PersonaPlexLmGen.step()` with `mx.compile` | 1.5-3x on LM step | Low |
| P2.2 | Strategic `mx.eval()` placement (once per frame) | Reduce eval overhead | Low |
| P2.3 | Remove `other_mimi` if output is truly discarded | 30-40% codec speedup | Low |
| P2.4 | Increase `asyncio.sleep` from 0.001 to 0.005 | Reduce CPU polling | Trivial |

### Priority 3: Medium-Effort Optimizations (Week 2)

| # | Optimization | Expected Impact | Effort |
|---|-------------|----------------|--------|
| P3.1 | Quantize model to 4-bit or 8-bit | 2-4x memory bandwidth | Medium |
| P3.2 | Compile the full depformer sample loop | 1.3-1.5x depformer speedup | Medium |
| P3.3 | Batch mx.eval calls (evaluate once at frame end) | Reduced eval overhead | Low-Medium |
| P3.4 | Use `mx.async_eval` for pipeline parallelism | Overlap compute with I/O | Medium |
| P3.5 | Pre-allocate numpy buffers for audio I/O | Reduce allocation overhead | Low |

### Priority 4: Architecture Improvements (Week 3+)

| # | Optimization | Expected Impact | Effort |
|---|-------------|----------------|--------|
| P4.1 | Implement proper `reset()` method on PersonaPlexLmGen | Robustness | Low |
| P4.2 | Add PerfLogger for per-frame timing | Diagnostics | Medium |
| P4.3 | Context window reduction (3000 -> 1500) | Proportional attention speedup | Low |
| P4.4 | Investigate Metal Performance Shaders capture for profiling | Diagnostics | Medium |
| P4.5 | Custom fused depformer step | 1.2-1.5x depformer speedup | High |

---

## 5. Quick Wins

These optimizations can be implemented in under an hour each with high expected
impact.

### 5.1 Fix gen_sequence reset value

**Current (buggy):**
```python
self.lm_gen.lm_gen.gen_sequence = mx.zeros_like(
    self.lm_gen.lm_gen.gen_sequence
)
```

**Fix:**
```python
self.lm_gen.lm_gen.gen_sequence = mx.full(
    self.lm_gen.lm_gen.gen_sequence.shape,
    vals=self.lm_gen.lm_gen.ungenerated_token,
    dtype=mx.int32,
)
```

**Impact:** Correctness fix. Without this, the delay compensation may produce
invalid tokens during the first few steps of a conversation.

### 5.2 Add `mx.compile` to the step function

The Kyutai `LmGen._step()` method calls `model._sample()` which runs the full
transformer + depformer. Wrapping this in `mx.compile` would let MLX fuse
operations across the entire forward pass.

However, `_step` has Python control flow (if/else on step_idx), so it cannot be
directly compiled. The inner `model._sample()` is a better candidate:

```python
# In PersonaPlexLmGen.__init__:
self._compiled_sample = mx.compile(model._sample)
```

Note: The sampling module already uses `@partial(mx.compile, ...)` on individual
sampling functions (top_k_sampling, categorical_sampling). Adding compilation at
the model level would provide additional fusion.

**Impact:** Expected 1.5-3x speedup on the LM forward pass. MLX's lazy evaluation
already provides some fusion, but explicit compilation creates a persistent
optimized compute graph.

### 5.3 Remove `other_mimi` dual codec

Both servers maintain two rustymimi Tokenizer instances. In the opus_loop, the
second instance's output is always discarded:

```python
_ = self.other_mimi.encode_step(chunk_input)  # Discarded
_ = self.other_mimi.decode_step(agent_audio_np)  # Discarded
```

If `other_mimi` exists solely to maintain streaming state for a secondary audio
stream (e.g., a debug/monitoring stream), and that stream is not being used, both
encode and decode calls on `other_mimi` can be removed.

**Impact:** ~30% reduction in audio codec compute per frame. With rustymimi, each
encode/decode involves a Rust FFI call plus internal Mimi transformer processing.

**Risk:** If other_mimi serves a purpose not visible in the server code (e.g., a
client expects two decoded streams), removing it would break that feature. Verify
with the PersonaPlex team before removing.

### 5.4 Strategic `mx.eval()` placement

Currently, `mx.eval()` is called in the warmup at the end:
```python
mx.eval(self.lm_gen.lm_gen.gen_sequence)
```

During inference (opus_loop), there is no explicit `mx.eval()` call. MLX's lazy
evaluation means that `np.array(agent_audio)` at line 330 forces evaluation of all
pending computation. This is fine, but explicitly calling `mx.eval(tokens)` after
`self.lm_gen.step(codes_mx)` would make the evaluation boundary clear and enable
future pipeline parallelism.

**Impact:** No immediate speedup, but enables future optimization (P3.4) and makes
performance profiling more accurate.

### 5.5 Increase polling sleep interval

```python
# Current (both loops):
await asyncio.sleep(0.001)  # 1ms
```

With the MLX model, if per-frame latency is ~40-80ms, polling every 1ms means
40-80 wasted polls per frame. Increasing to 5ms would reduce CPU overhead with
negligible latency impact:

```python
await asyncio.sleep(0.005)  # 5ms
```

**Impact:** Minor CPU reduction. More significant on battery-powered laptops.

---

## Appendix A: Line-by-Line Server Comparison Summary

| Feature | PyTorch Server | MLX Server | Match? |
|---------|---------------|------------|--------|
| WebSocket handshake (b"\x00") | Yes | Yes | YES |
| Audio message prefix (b"\x01") | Yes | Yes | YES |
| Text message prefix (b"\x02") | Yes | Yes | YES |
| recv_loop structure | async for ws | async for ws | YES |
| opus_loop structure | while + sleep(0.001) | while + sleep(0.001) | YES |
| send_loop structure | while + sleep(0.001) | while + sleep(0.001) | YES |
| PCM accumulation | numpy concatenate | numpy concatenate | YES |
| Frame size check | `>= self.frame_size` | `>= self.frame_size` | YES |
| Audio encode | `mimi.encode(chunk)` | `mimi.encode_step(chunk_input)` | Different API |
| Per-step iteration | `for c in range(codes.shape[-1])` | Single step | Different (see 1.7) |
| LM step | `self.lm_gen.step(codes)` | `self.lm_gen.step(codes_mx)` | Equivalent |
| Audio decode | `mimi.decode(tokens[:, 1:9])` | `mimi.decode_step(agent_audio_np)` | Different API |
| PCM output | `main_pcm[0, 0].numpy()` | `main_pcm[0, 0]` | Equivalent |
| Text token extract | `tokens[0, 0, 0].item()` | `int(tokens[0, 0, 0].item())` | YES |
| Text token filter | `not in (0, 3)` | `not in (0, 3)` | YES |
| Text encode | id_to_piece + replace | id_to_piece + replace | YES |
| System prompts | `step_system_prompts_async` | `asyncio.to_thread(step_system_prompts)` | Different (see 1.2) |
| Mimi reset after prompts | `reset_streaming()` | `mimi.reset()` | Equivalent |
| LMGen reset between connections | `reset_streaming()` (recursive) | Manual cache + gen_seq reset | Different (see 1.6) |
| Voice prompt loading | `.pt` or WAV | `.pt` or WAV | YES |
| Voice prompt guard | Unconditional | Guarded by `if voice_prompt_path:` | MLX safer |
| Text prompt handling | Attribute mutation | Parameter passing | MLX cleaner |
| Seed handling | `int(request["seed"])` | `request.query.get("seed", -1)` | MLX safer |
| is_alive check | `ws.receive(timeout=0.01)` | `ws.receive(timeout=0.01)` | Same bug |
| Task cancellation | asyncio.wait + cancel | asyncio.wait + cancel | YES |
| Lock for serialization | asyncio.Lock | asyncio.Lock | YES |
| Static file serving | Yes | Yes | YES |
| SSL support | Yes | Yes | YES |
| Gradio tunnel | Yes | Yes | YES |

## Appendix B: Optimization Context from Prior Analysis

The `optimisation.md` and `optimisation_ultrathink.md` documents identified 12
major bottlenecks causing 5x slowdown on PyTorch MPS. The MLX port resolves:

| # | Bottleneck | Status in MLX Port |
|---|-----------|-------------------|
| 1 | CUDA Graphs disabled | RESOLVED: MLX lazy evaluation + mx.compile |
| 2 | torch.compile disabled | RESOLVED: MLX lazy evaluation |
| 3 | No Flash Attention | RESOLVED: mx.fast.scaled_dot_product_attention |
| 4 | bfloat16 on MPS | RESOLVED: Using float16 natively |
| 5 | CPU sync in hot path | MOSTLY RESOLVED: Unified memory; still need mx.eval() |
| 6 | index_copy_ workaround | RESOLVED: MLX array indexing |
| 7 | torch.cdist in VQ | RESOLVED: Using rustymimi (Rust codec) |
| 8 | gather operations | RESOLVED: No gather needed in MLX LmGen |
| 9 | RMS norm float32 promotion | RESOLVED: MLX handles internally |
| 10 | Sequential DepFormer | PARTIALLY: Still sequential; mx.compile can help |
| 11 | Dual Mimi instances | NOT RESOLVED: Still has other_mimi |
| 12 | Tight polling loop | NOT RESOLVED: Still 1ms sleep |

The `mlxport.md` document specified Stages 5 and 6 optimizations that are the
natural next steps after the fixes identified in this review:
- Stage 5: mx.compile hot paths, optimize attention, quantization, remove dual
  Mimi, profiling with Metal capture
- Stage 6: Error handling, memory leak testing, long-running stability,
  documentation, CI/CD

## Appendix C: PersonaPlexLmGen Architecture Analysis

The `PersonaPlexLmGen` wrapper in `lm_gen_mlx.py` delegates to the Kyutai
`moshi_mlx.models.generate.LmGen`. Key observations:

1. **step()** correctly reshapes `[B, 8, 1]` input to `[B, 8]` and calls
   `self.lm_gen._step(other)`. The output assembly (`text_token + audio_tokens`)
   into `[B, dep_q+1, 1]` matches the expected format.

2. **step_text_prompt()** correctly overrides text tokens in gen_sequence and
   sets agent audio to SILENCE tokens with delay compensation. This matches
   the PyTorch `_step_text_prompt_core` logic.

3. **step_audio_silence()** correctly steps with SINE as user audio and overrides
   both text (PAD=3) and agent audio (SILENCE tokens) with delay compensation.

4. **step_with_embedding()** correctly feeds pre-computed embeddings through the
   transformer, bypassing the normal token embedding step. It handles the
   gen_sequence updates for SINE tokens (other audio), PAD (text), and
   delay-compensated agent audio.

5. **_restore_voice_prompt_cache()** correctly maps the PyTorch circular buffer
   positions to MLX linear gen_sequence positions.

6. **step_system_prompts()** follows the correct 4-phase orchestration:
   voice prompt -> silence -> text prompt -> silence.

The architecture is sound. The main concern is the gen_sequence reset value
(Section 1.6) and the lack of a proper reset method.
