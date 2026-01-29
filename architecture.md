# PersonaPlex Architecture

## Overview

PersonaPlex is a real-time, full-duplex speech-to-speech conversational AI system that enables **persona control** through text-based role prompts and audio-based voice conditioning.

**Key characteristics:**

- **Real-time**: Processes 80ms audio frames at 12.5 Hz with <80ms latency per frame
- **Full-duplex**: Simultaneous bidirectional audio (supports natural interruptions)
- **Voice control**: 16 pre-packaged voice embeddings (male/female, natural/varied)
- **Text control**: System prompts enable persona/role customization
- **Multi-backend**: PyTorch (CUDA) and MLX (Apple Silicon) implementations
- **Based on Moshi**: Built on Kyutai's Moshi architecture, fine-tuned for persona control

---

## Directory Structure

```
personaplex/
├── moshi/                           # Core inference engine (Python)
│   ├── moshi/                       # PyTorch implementation
│   │   ├── models/
│   │   │   ├── lm.py               # LMModel (7B transformer) + LMGen (streaming wrapper)
│   │   │   ├── compression.py      # Mimi audio codec
│   │   │   └── loaders.py          # Weight loading utilities
│   │   ├── modules/
│   │   │   ├── transformer.py      # StreamingTransformer with KV cache
│   │   │   ├── seanet.py           # SEANet encoder/decoder
│   │   │   ├── streaming.py        # StreamingModule base class
│   │   │   ├── conv.py             # StreamingConv1d
│   │   │   ├── rope.py             # Rotary positional embeddings
│   │   │   └── gating.py           # Gating mechanisms
│   │   ├── quantization/
│   │   │   ├── vq.py               # Vector quantization (Mimi)
│   │   │   ├── core_vq.py          # Core VQ implementation
│   │   │   └── base.py             # Base quantizer interface
│   │   ├── utils/
│   │   │   ├── sampling.py         # Top-k/top-p sampling
│   │   │   ├── compile.py          # torch.compile utilities
│   │   │   ├── logging.py          # Logging setup
│   │   │   └── connection.py       # SSL/networking
│   │   ├── server.py               # PyTorch WebSocket server
│   │   ├── offline.py              # Offline batch inference
│   │   └── client_utils.py         # CLI utilities
│   │
│   ├── moshi_mlx/                   # MLX implementation (Apple Silicon)
│   │   ├── server_mlx.py           # MLX WebSocket server
│   │   ├── lm_gen_mlx.py           # PersonaPlexLmGen wrapper
│   │   ├── loaders_mlx.py          # MLX weight loading
│   │   ├── offline_mlx.py          # MLX offline inference
│   │   ├── benchmark.py            # Performance benchmarking
│   │   ├── perf_logger.py          # Performance instrumentation
│   │   ├── convert_weights.py      # Weight format conversion
│   │   └── bench_*.py              # Detailed benchmarks
│   │
│   └── pyproject.toml               # Python dependencies
│
├── client/                           # Web UI (TypeScript/React)
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Queue/Queue.tsx      # Homepage with settings
│   │   │   └── Conversation/        # Main conversation interface
│   │   │       ├── Conversation.tsx
│   │   │       ├── hooks/           # useSocket, useServerAudio, useServerText, useModelParams
│   │   │       └── components/      # ServerAudio, UserAudio, AudioVisualizer, TextDisplay, Controls
│   │   ├── protocol/
│   │   │   ├── types.ts            # WebSocket message types
│   │   │   └── encoder.ts          # Binary message encoding/decoding
│   │   ├── decoder/
│   │   │   └── decoderWorker.ts    # Opus decoder Web Worker
│   │   ├── app.tsx                  # Router entry point
│   │   └── audio-processor.ts       # Web Audio worklet (Opus output)
│   ├── package.json
│   └── Dockerfile
│
├── assets/
│   ├── test/                         # Test inputs and benchmarks
│   └── analysis/
│
├── Dockerfile                        # CUDA container
├── docker-compose.yaml               # Docker Compose setup
└── README.md
```

---

## Core Model Architecture

### Main Transformer (PersonaPlex 7B)

The primary language model is a 7-billion parameter autoregressive transformer:

| Parameter | Value |
|-----------|-------|
| Dimensions | 4096 |
| Heads | 32 |
| Layers | 32 |
| FFN dim | 16896 (4.125x) |
| Positional encoding | RoPE |
| Normalization | RMS norm (pre-norm) |
| Gating | SiLU (GLU-style) |
| Context window | 3000 tokens |

The model processes **17 input streams** per step: 1 text token + 8 user audio codebook tokens + 8 agent audio codebook tokens. It produces **9 output streams**: 1 text token + 8 agent audio codebook tokens.

### DepFormer (Audio Codec Generator)

A smaller transformer that generates the 8 audio codebook tokens sequentially within each frame:

| Parameter | Value |
|-----------|-------|
| Dimensions | 1024 |
| Heads | 16 |
| Layers | 6 |
| FFN dim | 4224 (4.125x) |
| Context | 8 |
| Output slices | 8 (one per codebook) |

### Audio Codec (Mimi)

Mimi tokenizes raw audio into discrete codes and reconstructs audio from codes:

```
Encoder:                                  Decoder:
  Raw audio (24 kHz)                        Codes [1, 8, 480]
       │                                         │
  SEANet Conv (ratios [8,6,5,4])            Codebook lookup → [1, 512, 480]
       │                                         │
  Transformer (8L, 8H, d=512)              Transformer (8L)
       │                                         │
  Residual VQ (8 codebooks × 2048 bins)    SEANet Transpose Conv
       │                                         │
  Codes [1, 8, 480]                         Raw audio (24 kHz)
```

- **Frame rate**: 12.5 Hz (80ms per frame, 1920 samples at 24 kHz)
- **Codebooks**: 8 levels, 2048 bins each (11 bits per codebook)
- **Quantization**: Residual — each level predicts the residual from the previous

---

## System Components

### PyTorch Server (`moshi/moshi/server.py`)

Real-time WebSocket server for CUDA-capable GPUs. Key elements:

- **Protocol**: Binary WebSocket with type-prefixed messages
- **Audio I/O**: Opus codec via the `sphn` library
- **Main loop** (`opus_loop`): recv Opus → decode PCM → Mimi encode → LM step → Mimi decode → encode Opus → send
- **State**: `ServerState` manages audio tokenizers, LM instance, and locks
- **Optimization**: CUDA Graphs captured after warmup steps to eliminate Python dispatch overhead

### MLX Server (`moshi/moshi_mlx/server_mlx.py`)

Equivalent server for Apple Silicon, using the MLX framework:

- **Same protocol** as the PyTorch server for full client compatibility
- **Audio codec**: Uses `rustymimi` (Rust-backed) instead of PyTorch Mimi
- **Key wrapper**: `PersonaPlexLmGen` (`lm_gen_mlx.py`) adapts Kyutai's MLX LmGen with PersonaPlex voice/text prompt support
- **Optimization**: Pipelined decode (previous frame) overlaps with encode + LM (current frame), reducing per-frame latency by ~18ms
- **Memory model**: MLX unified memory — no explicit CPU↔GPU transfers, lazy evaluation with strategic `mx.eval()` synchronization

### Offline Inference (`moshi/moshi/offline.py`, `moshi_mlx/offline_mlx.py`)

Batch processing without real-time constraints:

1. Load input WAV and segment into frames
2. Run system prompt phase (voice embeddings + text tokens)
3. Stream audio through LM frame-by-frame
4. Accumulate output, resample, and write WAV + text JSON

### Web Client (`client/`)

React/TypeScript single-page application:

- **Conversation page**: Orchestrates WebSocket connection, microphone input, speaker output, and real-time visualization
- **useSocket hook**: Manages WebSocket lifecycle, binary protocol encoding/decoding, handshake
- **Audio worklet**: Web Audio API processor for low-latency Opus playback with buffer management
- **Decoder worker**: WASM-based Opus decoding in a Web Worker thread

---

## Data Flow

### Real-Time Streaming

```
Client Browser                    Server                         Model
─────────────────────────────────────────────────────────────────────────
  Mic capture (24 kHz)
       │
  Opus encode
       │
  WS send (0x01 + opus)  ──────►  Recv loop
                                      │
                                  Opus decode → PCM
                                      │
                                  Mimi encode → codes [8]
                                      │
                                  LM step (main transformer)
                                      │
                                  DepFormer (8 codebooks)
                                      │
                                  Sampling (temp, top-k, top-p)
                                      │
                                  Mimi decode → PCM
                                      │
  WS recv (0x01 + opus)  ◄──────  Opus encode
  WS recv (0x02 + text)  ◄──────  Text token (parallel)
       │
  Opus decode + play
```

### Prompt Injection Pipeline

**Voice prompt**: Pre-computed `.pt` embeddings (or raw `.wav` audio) are replayed through the transformer at connection time, caching activation states for consistent voice throughout the session.

**Text prompt**: A system prompt string is encoded via SentencePiece, wrapped with `<system>` tags, and injected into the LM input stream. Six frames of silence follow to let the model absorb the prompt before user audio begins.

### WebSocket Binary Protocol

| Byte prefix | Type | Payload |
|-------------|------|---------|
| `0x00` | Handshake | Empty (server signals prompts loaded) |
| `0x01` | Audio | Opus-encoded bytes |
| `0x02` | Text | UTF-8 text token |
| `0x03` | Control | start, endTurn, pause, restart |
| `0x04` | Metadata | JSON |
| `0x05` | Error | UTF-8 message |
| `0x06` | Ping | Empty |

---

## Technology Stack

### Backend

| Library | Purpose |
|---------|---------|
| PyTorch (>=2.2, <2.5) | LM/audio codec inference (CUDA) |
| MLX | LM inference (Apple Silicon) |
| rustymimi (>=0.1.4) | Audio tokenization (MLX path) |
| sphn (>=0.1.4) | Opus codec |
| aiohttp (>=3.10.5) | WebSocket server |
| safetensors | Weight serialization |
| sentencepiece | Text tokenization |
| huggingface-hub | Model downloading |
| numpy | Array operations |
| einops | Tensor rearrangement |

### Frontend

| Library | Purpose |
|---------|---------|
| React 18 | UI framework |
| TypeScript 5 | Type safety |
| Vite 5 | Build tool |
| Tailwind CSS + DaisyUI | Styling |
| opus-recorder | Opus encoding |
| zod | Schema validation |

### Deployment

| Tool | Purpose |
|------|---------|
| Docker + docker-compose | Containerization |
| CUDA 12.4 | GPU runtime |
| Ubuntu 22.04 | Base OS |

---

## Key Design Patterns

### Streaming with KV Cache

The transformer maintains a fixed-size KV cache (3000 tokens for main, 8 for DepFormer). Each step consumes one frame's tokens and appends new key-value pairs, enabling constant-time inference after initial warmup.

### CUDA Graphs (PyTorch)

After 4 warmup steps, the full forward pass is captured as a static CUDA graph, eliminating Python dispatch overhead on subsequent frames. This is disabled on MPS/CPU where no equivalent exists.

### Async I/O Separation

Receive and send loops run in separate threads/coroutines. This allows pipelining: the previous frame's audio decode can overlap with the current frame's LM inference, reducing end-to-end latency.

### Two-Phase Prompt Initialization

1. **Voice prompt phase**: Pre-computed embeddings or WAV audio are fed through the model, caching activations
2. **Text prompt phase**: SentencePiece-encoded system prompt is injected with silence padding

This separates persona setup from regular inference, ensuring consistent voice/behavior without per-turn overhead.

### MLX Lazy Evaluation

MLX uses lazy evaluation with unified memory. Explicit `mx.eval()` calls are placed only at synchronization boundaries (after warmup, after reset), minimizing overhead compared to PyTorch's eager execution on MPS.

### Residual Vector Quantization

Audio is encoded into 8 sequential codebook levels. Each level learns to quantize the residual from previous levels, providing high-quality discrete audio representation at 12.5 Hz.

---

## Performance Characteristics

### Latency Budget (MLX, Apple M3, 96 GB)

| Phase | Time | Budget |
|-------|------|--------|
| Opus decode | ~5 ms | 6% |
| Mimi encode | ~7 ms | 9% |
| LM forward + DepFormer | ~50 ms | 63% |
| Mimi decode | ~8 ms | 10% |
| Opus encode | ~5 ms | 6% |
| Network + overhead | ~5 ms | 6% |
| **Total per frame** | **~80 ms** | **100%** |

### Memory Usage (MLX, float16)

| Component | Size |
|-----------|------|
| Model weights | ~7.5–8 GB |
| KV cache | ~500 MB |
| Activations/buffers | ~500 MB |
| **Total** | **~8.5 GB** |

### Throughput

- **Real-time**: 12.5 frames/sec (80 ms per frame)
- **Offline**: ~9 tokens/sec (including audio encode/decode)

---

## Entry Points

```bash
# PyTorch server (CUDA)
python -m moshi.server --ssl ./ssl_certs

# MLX server (Apple Silicon)
python -m moshi_mlx.server_mlx \
  --hf-repo nvidia/personaplex-7b-v1 \
  --host 0.0.0.0 --port 8998

# Offline inference (PyTorch)
python -m moshi.offline \
  --input-wav input.wav --output-wav output.wav \
  --voice-prompt NATF2.pt --text-prompt "You are an assistant..."

# Offline inference (MLX)
python -m moshi_mlx.offline_mlx \
  --input-wav input.wav --output-wav output.wav

# Performance benchmarks
python -m moshi_mlx.benchmark --compare baseline.json current.json

# Web client (development)
cd client && npm run dev
```

---

## Deployment

### Docker (CUDA)

```bash
docker-compose build
docker-compose up
```

The `docker-compose.yaml` maps port 8998 and reserves one NVIDIA GPU. The `Dockerfile` uses CUDA 12.4 on Ubuntu 22.04.

### Native (Apple Silicon)

Install dependencies from `pyproject.toml` with the `mlx` extra, then run `server_mlx.py` directly. No containerization required — the MLX backend leverages unified memory on M-series chips.

### Configuration

Key CLI arguments shared across servers:

| Flag | Description |
|------|-------------|
| `--hf-repo` | HuggingFace model repository |
| `--moshi-weight` | Local model weights path |
| `--tokenizer` | SentencePiece tokenizer path |
| `--voice-prompt-dir` | Directory of voice embedding `.pt` files |
| `--host`, `--port` | Bind address (default `0.0.0.0:8998`) |
| `--ssl` | SSL certificate directory |
| `--temp-text`, `--temp-audio` | Sampling temperatures |
| `--topk-text`, `--topk-audio` | Top-k sampling values |
| `--max-steps` | Maximum autoregressive steps per session |
