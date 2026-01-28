# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Real-time WebSocket server for PersonaPlex using MLX backend.

This is the MLX port of moshi.moshi.server, replacing:
- PyTorch LMModel -> MLX Lm (via loaders_mlx.get_personaplex_lm)
- PyTorch MimiModel -> rustymimi Tokenizer
- PyTorch LMGen -> PersonaPlexLmGen (wrapping Kyutai MLX LmGen)

Key differences from PyTorch server:
- No torch.device, uses string "mlx" for logging
- No .to(device) calls - MLX uses unified memory
- No torch.from_numpy() - use mx.array() directly
- No .cpu() calls - rustymimi returns numpy already
- mx.eval() replaces torch.cuda.synchronize()
- PersonaPlexLmGen has different constructor and API

The WebSocket protocol remains identical to ensure client compatibility:
- b"\x00" handshake after system prompts
- b"\x01" + opus_bytes for audio streams
- b"\x02" + utf8_text for text tokens
"""

import argparse
import asyncio
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sphn

from ..moshi.client_utils import colorize
from ..moshi.utils.connection import create_ssl_context, get_lan_ip
from ..moshi.utils.logging import setup_logger, ColorizedLog
from .loaders_mlx import get_personaplex_lm
from .lm_gen_mlx import PersonaPlexLmGen
from .utils.sampling import Sampler

logger = setup_logger(__name__)

# Constants matching PersonaPlex configuration
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples per frame
DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"
VOICES_TGZ_NAME = "voices.tgz"


def seed_all(seed):
    """Seed all RNGs for reproducibility."""
    mx.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class ServerState:
    """MLX server state holding audio tokenizer, text tokenizer, and LM generator.

    This replaces the PyTorch ServerState with MLX equivalents:
    - mimi/other_mimi: rustymimi.Tokenizer instances
    - lm_gen: PersonaPlexLmGen wrapper
    - device: string "mlx" (for logging only, MLX uses unified memory)
    """
    mimi: rustymimi.Tokenizer
    other_mimi: rustymimi.Tokenizer
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: PersonaPlexLmGen
    lock: asyncio.Lock
    device: str
    voice_prompt_dir: Optional[str]
    frame_size: int

    def __init__(
        self,
        mimi: rustymimi.Tokenizer,
        other_mimi: rustymimi.Tokenizer,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm_gen: PersonaPlexLmGen,
        device: str = "mlx",
        voice_prompt_dir: Optional[str] = None,
    ):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = lm_gen
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = FRAME_SIZE
        self.lock = asyncio.Lock()

        # Initialize streaming mode for both tokenizers
        self.mimi.reset()
        self.other_mimi.reset()

    def warmup(self):
        """Run warmup loop to prime MLX compilation and caches.

        Feeds 4 dummy frames through the full pipeline:
        - rustymimi encode (PCM -> codes)
        - PersonaPlexLmGen step (codes -> tokens)
        - rustymimi decode (tokens -> PCM)

        This matches the PyTorch warmup structure but uses MLX operations.
        """
        logger.info("Warming up the model")
        for _ in range(4):
            # Generate zero PCM frame [1, 1, frame_size]
            chunk = np.zeros((1, 1, self.frame_size), dtype=np.float32)

            # Encode with both mimi instances
            codes = self.mimi.encode_step(chunk)
            _ = self.other_mimi.encode_step(chunk)

            # codes: [1, 1, num_codebooks] from rustymimi
            # Convert to [1, num_codebooks, 1] for PersonaPlexLmGen
            codes_np = np.array(codes)
            codes_mx = mx.array(codes_np).transpose(0, 2, 1)[:, :8, :]  # [1, 8, 1]

            # Step through the LM
            tokens = self.lm_gen.step(codes_mx)
            if tokens is None:
                continue

            # Decode agent audio tokens[:, 1:9] with both mimi instances
            # tokens: [B, dep_q+1, 1] -> agent audio is tokens[:, 1:, :]
            agent_audio = tokens[:, 1:, :]  # [1, 8, 1]
            agent_audio_np = np.array(agent_audio).astype(np.uint32)

            _ = self.mimi.decode_step(agent_audio_np)
            _ = self.other_mimi.decode_step(agent_audio_np)

        # Force evaluation to ensure all ops are compiled
        mx.eval(self.lm_gen.lm_gen.gen_sequence)
        logger.info("Warmup complete")

    async def handle_chat(self, request):
        """WebSocket handler for /api/chat endpoint.

        Handles the full conversation lifecycle:
        1. Parse query params (voice_prompt, text_prompt, seed)
        2. Load voice prompt if specified
        3. Set text prompt tokens
        4. Create opus reader/writer for audio I/O
        5. Reset streaming state
        6. Run system prompts (voice + silence + text + silence)
        7. Send handshake b"\x00"
        8. Run three async loops:
           - recv_loop: receive opus audio from client
           - opus_loop: encode user audio, run LM, decode agent audio
           - send_loop: send opus audio to client

        This closely mirrors the PyTorch server.handle_chat but uses MLX inference.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote
        peer_port = request.transport.get_extra_info("peername")[1]
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # Construct full voice prompt path
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query.get("voice_prompt", "")
            if voice_prompt_filename:
                requested_voice_prompt_path = os.path.join(
                    self.voice_prompt_dir, voice_prompt_filename
                )
                if not os.path.exists(requested_voice_prompt_path):
                    raise FileNotFoundError(
                        f"Requested voice prompt '{voice_prompt_filename}' "
                        f"not found in '{self.voice_prompt_dir}'"
                    )
                voice_prompt_path = requested_voice_prompt_path

        # Load voice prompt if changed
        if voice_prompt_path:
            # Check if we need to reload (matching PyTorch server behavior)
            if self.lm_gen.voice_prompt != voice_prompt_path:
                if voice_prompt_path.endswith('.pt'):
                    # Load pre-saved voice prompt embeddings
                    self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
                else:
                    # Load voice prompt audio WAV
                    self.lm_gen.load_voice_prompt(voice_prompt_path)

        # Set text prompt tokens
        text_prompt_str = request.query.get("text_prompt", "")
        text_prompt_tokens = None
        if len(text_prompt_str) > 0:
            text_prompt_tokens = self.text_tokenizer.encode(
                wrap_with_system_tags(text_prompt_str)
            )

        seed = int(request.query.get("seed", -1))

        async def recv_loop():
            """Receive loop: reads opus audio from client."""
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            """Opus processing loop: encode user audio, run LM, decode agent audio.

            This is the core inference loop that:
            1. Reads PCM from opus_reader (numpy arrays)
            2. Accumulates until frame_size (1920 samples)
            3. Encodes with rustymimi (numpy -> codes)
            4. Converts codes to mx.array
            5. Runs LM step (MLX)
            6. Decodes agent audio (MLX -> numpy)
            7. Writes to opus_writer
            8. Extracts text token and sends to client
            """
            all_pcm_data = None

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))

                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]

                    # Encode user audio with rustymimi
                    # chunk: [frame_size] -> need [1, 1, frame_size]
                    chunk_input = chunk[np.newaxis, np.newaxis, :]
                    codes = self.mimi.encode_step(chunk_input)
                    _ = self.other_mimi.encode_step(chunk_input)

                    # Convert codes to MLX array
                    # codes: [1, 1, num_codebooks] -> [1, num_codebooks, 1]
                    codes_np = np.array(codes)
                    codes_mx = mx.array(codes_np).transpose(0, 2, 1)[:, :8, :]

                    # LM step
                    tokens = self.lm_gen.step(codes_mx)
                    if tokens is None:
                        continue

                    # Decode agent audio
                    # tokens: [B, dep_q+1, 1] -> agent audio is tokens[:, 1:, :]
                    agent_audio = tokens[:, 1:, :]  # [1, 8, 1]
                    agent_audio_np = np.array(agent_audio).astype(np.uint32)

                    main_pcm = self.mimi.decode_step(agent_audio_np)
                    _ = self.other_mimi.decode_step(agent_audio_np)

                    # main_pcm: [1, 1, frame_size]
                    opus_writer.append_pcm(main_pcm[0, 0])

                    # Extract text token
                    text_token = int(tokens[0, 0, 0].item())
                    if text_token not in (0, 3):
                        _text = self.text_tokenizer.id_to_piece(text_token)
                        _text = _text.replace("▁", " ")
                        msg = b"\x02" + bytes(_text, encoding="utf8")
                        await ws.send_bytes(msg)

        async def send_loop():
            """Send loop: writes opus audio to client."""
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if text_prompt_str:
            clog.log("info", f"text prompt: {text_prompt_str}")
        if voice_prompt_path:
            clog.log("info", f"voice prompt: {voice_prompt_path}")

        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

            # Reset streaming state
            self.mimi.reset()
            self.other_mimi.reset()

            # Reset LM generator
            # For PersonaPlexLmGen, we need to reset the inner LmGen caches
            for c in self.lm_gen.model.transformer_cache:
                c.reset()
            for c in self.lm_gen.model.depformer_cache:
                c.reset()
            self.lm_gen.lm_gen.step_idx = 0
            # Reset gen_sequence to zeros
            self.lm_gen.lm_gen.gen_sequence = mx.zeros_like(
                self.lm_gen.lm_gen.gen_sequence
            )
            mx.eval(self.lm_gen.lm_gen.gen_sequence)

            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        return False
                except asyncio.TimeoutError:
                    # No messages -> client probably still alive
                    return True
                except Exception:
                    return False
                return True

            # Run system prompts (voice + silence + text + silence)
            # Use asyncio.to_thread to avoid blocking the event loop
            await asyncio.to_thread(
                self.lm_gen.step_system_prompts,
                audio_tokenizer=self.mimi,
                text_token_ids=text_prompt_tokens,
            )

            # Reset mimi after system prompts (voice prompt encoding may have consumed steps)
            self.mimi.reset()

            clog.log("info", "done with system prompts")

            # Send handshake
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")

                # Start the three async loops
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]

                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                await ws.close()
                clog.log("info", "session closed")

        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(
    voice_prompt_dir: Optional[str], hf_repo: str
) -> Optional[str]:
    """Download and extract voice prompts from HF if needed.

    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, VOICES_TGZ_NAME)
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    """Get path to static web UI files."""
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download(DEFAULT_HF_REPO, "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    """Main entry point for the MLX server."""
    parser = argparse.ArgumentParser(
        description="PersonaPlex MLX real-time WebSocket server"
    )
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument(
        "--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.'
    )
    parser.add_argument(
        "--gradio-tunnel-token",
        help='Provide a custom (secret) token here to keep getting the same URL.',
    )

    parser.add_argument(
        "--tokenizer", type=str, help="Path to a local tokenizer file."
    )
    parser.add_argument(
        "--moshi-weight", type=str, help="Path to PersonaPlex MLX weights (safetensors)."
    )
    parser.add_argument(
        "--mimi-weight", type=str, help="Path to rustymimi Mimi tokenizer weights."
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="HF repo to look into, defaults PersonaPlex. "
        "Use this to select a different pre-trained model.",
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted. "
            "Voice prompt filenames from client requests will be joined with this directory path."
        ),
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )

    # Sampling parameters
    parser.add_argument(
        "--temp-audio", type=float, default=0.8, help="Audio sampling temperature"
    )
    parser.add_argument(
        "--temp-text", type=float, default=0.7, help="Text sampling temperature"
    )
    parser.add_argument(
        "--topk-audio", type=int, default=250, help="Audio top-k sampling"
    )
    parser.add_argument(
        "--topk-text", type=int, default=25, help="Text top-k sampling"
    )
    parser.add_argument(
        "--max-steps", type=int, default=3000, help="Maximum autoregressive steps"
    )

    args = parser.parse_args()

    # Resolve voice prompt directory
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    # Resolve static path
    static_path: Optional[str] = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking
        except ImportError:
            logger.error(
                "Cannot find gradio which is required to activate a tunnel. "
                "Please install with `pip install gradio`."
            )
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    hf_hub_download(args.hf_repo, "config.json")

    # Load rustymimi audio tokenizer
    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, MIMI_NAME)
    mimi = rustymimi.Tokenizer(args.mimi_weight, num_codebooks=8)
    other_mimi = rustymimi.Tokenizer(args.mimi_weight, num_codebooks=8)
    logger.info("mimi loaded")

    # Load text tokenizer
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    # Load PersonaPlex MLX model
    logger.info("loading moshi mlx")
    if args.moshi_weight is None:
        # Try to find in HF cache or download
        try:
            args.moshi_weight = hf_hub_download(
                args.hf_repo, "personaplex_mlx.safetensors"
            )
        except Exception:
            logger.error(
                "Could not find personaplex_mlx.safetensors in HF repo. "
                "Please run convert_weights.py first or provide --moshi-weight path."
            )
            sys.exit(1)

    model = get_personaplex_lm(args.moshi_weight, dtype=mx.float16)
    logger.info("moshi mlx loaded")

    # Create samplers
    text_sampler = Sampler(top_k=args.topk_text, temp=args.temp_text)
    audio_sampler = Sampler(top_k=args.topk_audio, temp=args.temp_audio)

    # Create PersonaPlexLmGen
    lm_gen = PersonaPlexLmGen(
        model=model,
        max_steps=args.max_steps,
        text_sampler=text_sampler,
        audio_sampler=audio_sampler,
        batch_size=1,
    )

    # Create server state
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
        device="mlx",
        voice_prompt_dir=args.voice_prompt_dir,
    )

    logger.info("warming up the model")
    state.warmup()

    # Create web app
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)

    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")

    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
