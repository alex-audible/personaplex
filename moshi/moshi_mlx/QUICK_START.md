# Quick Start: PersonaPlex MLX Server

## Installation

```bash
# Install dependencies (if not already installed)
pip install mlx mlx-lm aiohttp sentencepiece sphn rustymimi
```

## Running the Server

### 1. Auto-download from HuggingFace (Recommended)

```bash
cd /Users/220611g/Documents/Projects/personaplex
python -m moshi.moshi_mlx.server_mlx \
    --hf-repo nvidia/personaplex-7b-v1 \
    --host 0.0.0.0 \
    --port 8998
```

The server will automatically download:
- `personaplex_mlx.safetensors` (MLX weights, ~14GB)
- `tokenizer-e351c8d8-checkpoint125.safetensors` (Mimi audio tokenizer)
- `tokenizer_spm_32k_3.model` (Text tokenizer)
- `voices.tgz` (Voice prompt archive)
- `dist.tgz` (Web UI)

### 2. Local Weights (Faster)

If you already have the weights:

```bash
python -m moshi.moshi_mlx.server_mlx \
    --moshi-weight ./personaplex_mlx.safetensors \
    --mimi-weight ./tokenizer-e351c8d8-checkpoint125.safetensors \
    --tokenizer ./tokenizer_spm_32k_3.model \
    --voice-prompt-dir ./voices \
    --static ./dist \
    --port 8998
```

## Accessing the UI

Once the server starts, you'll see:

```
INFO - Access the Web UI directly at http://YOUR_IP:8998
```

Open this URL in a browser to access the web interface.

## Testing

### Basic Test

1. **Start server** (see above)
2. **Open web UI** in browser
3. **Click "Connect"** button
4. **Speak** into your microphone
5. **Listen** for agent response

### Expected Output

```
INFO - loading mimi
INFO - mimi loaded
INFO - loading moshi mlx
INFO - Creating PersonaPlex MLX model (num_slices=8) ...
INFO - Loading MLX weights from ./personaplex_mlx.safetensors ...
INFO - moshi mlx loaded
INFO - warming up the model
INFO - Warming up the model
INFO - Warmup complete
INFO - ======== Running on http://0.0.0.0:8998 ========
INFO - [ABC1] Incoming connection from 127.0.0.1:54321
INFO - [ABC1] accepted connection
INFO - [ABC1] done with system prompts
INFO - [ABC1] sent handshake bytes
```

## Troubleshooting

### "Cannot find module 'aiohttp'"

```bash
pip install aiohttp
```

### "Cannot find module 'rustymimi'"

```bash
pip install rustymimi
```

### "personaplex_mlx.safetensors not found"

Either:
1. Run `convert_weights.py` first to create MLX weights
2. Provide `--hf-repo` to download from HuggingFace

### Port 8998 already in use

```bash
# Use different port
python -m moshi.moshi_mlx.server_mlx --port 8999
```

### Out of memory

```bash
# Reduce max_steps
python -m moshi.moshi_mlx.server_mlx --max-steps 2000
```

## Next Steps

See `SERVER_MLX_README.md` for:
- Detailed architecture
- Advanced configuration
- SSL setup
- Performance tuning
- Development guide

## Quick Reference

**Default Values:**
- Host: `localhost`
- Port: `8998`
- Max steps: `3000`
- Audio temperature: `0.8`
- Text temperature: `0.7`
- Audio top-k: `250`
- Text top-k: `25`

**Query Parameters (client sends):**
- `voice_prompt`: e.g., `NATF2.pt`
- `text_prompt`: e.g., `"You are a helpful assistant"`
- `seed`: e.g., `42424242` (or `-1` for random)

**Message Protocol:**
- Client → Server: `b"\x01" + opus_bytes` (audio)
- Server → Client: `b"\x00"` (handshake)
- Server → Client: `b"\x01" + opus_bytes` (audio)
- Server → Client: `b"\x02" + utf8_bytes` (text)
