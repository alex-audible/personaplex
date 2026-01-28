# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""PersonaPlex LMGen wrapper for MLX inference.

Wraps the Kyutai ``moshi_mlx.models.generate.LmGen`` with PersonaPlex-specific
token routing:

- **User audio input** (8 codebooks from rustymimi encode) is fed as
  ``other_audio_tokens`` to the inner LmGen.
- **Agent audio output** (8 codebooks from the depformer) is collected via
  ``last_audio_tokens()`` after delay compensation.
- **Text tokens** are extracted from the inner LmGen's step return value.
- **Voice prompt** injection via pre-computed embeddings (.pt) or WAV audio.
- **Text prompt** injection feeds pre-encoded sentencepiece IDs.
- **Silence/sine** frame injection for warmup and post-prompt padding.

Usage::

    from moshi_mlx.lm_gen_mlx import PersonaPlexLmGen
    from moshi_mlx.models.generate import LmGen
    from moshi_mlx.utils import Sampler

    gen = PersonaPlexLmGen(
        model=model,
        max_steps=steps,
        text_sampler=Sampler(top_k=25, temp=0.7),
        audio_sampler=Sampler(top_k=250, temp=0.8),
    )
    for user_tokens in encode_frames(...):
        out = gen.step(user_tokens)
        if out is not None:
            text_tok, agent_audio = out[:, :1], out[:, 1:]
"""

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx
import numpy as np

from moshi_mlx.models.generate import LmGen
from moshi_mlx.models.lm import Lm
from moshi_mlx.utils.sampling import Sampler

logger = logging.getLogger(__name__)

# Token patterns for silence and a pure-tone sine wave, used as "user audio"
# placeholders during warmup, prompt injection, and post-prompt padding.
# These match the values in PersonaPlex's ``moshi.models.lm``.
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]

# Constants matching PyTorch reference
ZERO_TEXT_CODE = 3          # PAD token for text
AUDIO_SILENCE_FRAME_CNT = 6  # int(0.5 * 12.5) = 6 frames = 0.5 seconds
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples per frame


class PersonaPlexLmGen:
    """PersonaPlex inference wrapper for the Kyutai MLX LmGen.

    This class does **not** reimplement the core autoregressive loop.  It
    delegates entirely to :class:`moshi_mlx.models.generate.LmGen` and adds
    PersonaPlex-specific logic:

    * Reshaping user audio tokens from rustymimi's ``[B, 8, 1]`` output into
      the ``[B, 8]`` expected by LmGen.
    * Collecting delay-compensated agent audio tokens and concatenating them
      with the sampled text token into a single ``[B, dep_q+1, 1]`` tensor.
    * Helper methods for silence stepping, text-prompt injection, and
      voice-prompt injection (embeddings or WAV).

    Args:
        model: A loaded :class:`Lm` model instance (from ``loaders_mlx``).
        max_steps: Maximum number of autoregressive steps.
        text_sampler: Sampler for text logits.
        audio_sampler: Sampler for audio logits.
        batch_size: Batch dimension (default 1).
        cfg_coef: Classifier-free guidance coefficient (default 1.0).
    """

    def __init__(
        self,
        model: Lm,
        max_steps: int,
        text_sampler: Sampler,
        audio_sampler: Sampler,
        batch_size: int = 1,
        cfg_coef: float = 1.0,
    ) -> None:
        self.model = model
        self.batch_size = batch_size

        # Codebook counts from the model config
        self.dep_q: int = model.cfg.depformer.num_slices  # 8 (agent audio)
        self.other_q: int = model.cfg.other_codebooks      # 8 (user audio)

        # Build the inner Kyutai LmGen
        self.lm_gen = LmGen(
            model=model,
            max_steps=max_steps,
            text_sampler=text_sampler,
            audio_sampler=audio_sampler,
            batch_size=batch_size,
            cfg_coef=cfg_coef,
            check=False,
        )

        # Voice prompt state
        self.voice_prompt: Optional[str] = None  # Path to loaded voice prompt
        self.voice_prompt_embeddings: Optional[mx.array] = None
        self.voice_prompt_cache: Optional[mx.array] = None
        self.voice_prompt_audio: Optional[np.ndarray] = None

        logger.info(
            "PersonaPlexLmGen created: dep_q=%d, other_q=%d, max_steps=%d",
            self.dep_q,
            self.other_q,
            max_steps,
        )

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, user_audio_tokens: mx.array) -> Optional[mx.array]:
        """Feed user audio tokens and advance one autoregressive step.

        Args:
            user_audio_tokens: ``[B, 8, 1]`` from rustymimi encode, or
                ``[B, 8]`` if already squeezed.

        Returns:
            ``[B, dep_q+1, 1]`` combining (text token, 8 agent audio tokens),
            or ``None`` if the delay buffer has not yet filled.
        """
        # Ensure shape is [B, other_q] as expected by LmGen._step
        if user_audio_tokens.ndim == 3:
            other = user_audio_tokens[:, :, 0]  # [B, 8]
        else:
            other = user_audio_tokens  # already [B, 8]

        # Inner step returns (text_token [B,1], transformer_out)
        text_token, _ = self.lm_gen._step(other)

        # Collect delay-compensated agent audio tokens
        audio = self.lm_gen.last_audio_tokens()  # [B, dep_q] or None
        if audio is None:
            return None

        # Combine text + agent audio into [B, dep_q+1, 1]
        # text_token is [B, 1], audio is [B, dep_q]
        text_col = text_token[:, :, None] if text_token.ndim == 2 else text_token  # -> [B, 1, 1]
        audio_col = audio[:, :, None]  # [B, dep_q, 1]
        out = mx.concatenate([text_col, audio_col], axis=1)  # [B, dep_q+1, 1]
        return out

    # ------------------------------------------------------------------
    # Silence / sine stepping (for warmup and post-prompt padding)
    # ------------------------------------------------------------------

    def step_silence(self) -> Optional[mx.array]:
        """Step with silence tokens as user audio input.

        Used during warmup and post-prompt padding phases where no real
        user audio is available.

        Returns:
            Same as :meth:`step` -- ``[B, dep_q+1, 1]`` or ``None``.
        """
        silence = mx.array(SILENCE_TOKENS, dtype=mx.int32)
        silence = mx.broadcast_to(silence[None, :], (self.batch_size, self.other_q))
        return self.step(silence)

    def step_sine(self) -> Optional[mx.array]:
        """Step with sine-wave tokens as user audio input.

        Returns:
            Same as :meth:`step` -- ``[B, dep_q+1, 1]`` or ``None``.
        """
        sine = mx.array(SINE_TOKENS, dtype=mx.int32)
        sine = mx.broadcast_to(sine[None, :], (self.batch_size, self.other_q))
        return self.step(sine)

    # ------------------------------------------------------------------
    # Text prompt injection
    # ------------------------------------------------------------------

    def step_text_prompt(self, text_token_ids: list[int]) -> None:
        """Inject text prompt tokens by stepping with SINE as user audio.

        Each text prompt token is force-written into the gen_sequence at
        codebook index 0 (the text channel) after stepping with SINE
        as the user audio input.  This mirrors the PersonaPlex server's
        ``_step_text_prompt_core`` logic: agent audio = SILENCE,
        user audio = SINE, text = forced token.

        Args:
            text_token_ids: List of sentencepiece token IDs for the
                system prompt text.
        """
        for tid in text_token_ids:
            # User audio = SINE (matching PyTorch reference)
            sine = mx.array(SINE_TOKENS, dtype=mx.int32)
            sine = mx.broadcast_to(
                sine[None, :], (self.batch_size, self.other_q)
            )
            # Step the model (ignore output -- we override the text channel)
            self.lm_gen._step(sine)

            # Overwrite the text token that was just sampled
            step = self.lm_gen.step_idx - 1  # _step already incremented
            self.lm_gen.gen_sequence[:, 0, step] = mx.array(
                [tid] * self.batch_size, dtype=mx.int32
            )

            # Override agent audio with SILENCE tokens (matching PyTorch
            # _step_text_prompt_core which uses _encode_zero_frame for moshi_tokens)
            silence = mx.array(SILENCE_TOKENS, dtype=mx.int32)
            for cb_idx, delay in enumerate(
                self.lm_gen.audio_delays[: self.dep_q]
            ):
                gen_idx = step - delay
                if gen_idx >= 0:
                    self.lm_gen.gen_sequence[
                        :, cb_idx + 1, gen_idx
                    ] = mx.broadcast_to(
                        silence[cb_idx : cb_idx + 1],
                        (self.batch_size,),
                    )

    # ------------------------------------------------------------------
    # Voice prompt loading
    # ------------------------------------------------------------------

    def load_voice_prompt(self, path: str) -> None:
        """Load a voice prompt WAV from disk and normalize to -24 LUFS.

        The audio is stored for frame-by-frame encoding with rustymimi
        during the voice prompt injection phase.

        Args:
            path: Path to a WAV voice prompt file.
        """
        import pyloudnorm as pyln
        import sphn

        logger.info("Loading voice prompt WAV from: %s", path)
        raw_audio, _sr = sphn.read(path, sample_rate=SAMPLE_RATE)
        # raw_audio shape: [channels, T]

        # Take first channel and normalize to -24 LUFS
        wav = raw_audio[0].astype(np.float32)
        meter = pyln.Meter(SAMPLE_RATE)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -24.0)

        if wav.ndim == 1:
            wav = wav[None, :]  # [1, T]

        self.voice_prompt = path
        self.voice_prompt_audio = wav
        self.voice_prompt_embeddings = None
        self.voice_prompt_cache = None
        logger.info(
            "Voice prompt loaded: %d samples (%.2fs, %d frames)",
            wav.shape[-1],
            wav.shape[-1] / SAMPLE_RATE,
            wav.shape[-1] // FRAME_SIZE,
        )

    def load_voice_prompt_embeddings(self, path: str) -> None:
        """Load pre-computed voice prompt embeddings from a ``.pt`` file.

        The ``.pt`` file contains:
        - ``embeddings``: ``[N, 1, 1, d_model]`` pre-computed transformer
          input embeddings (bfloat16).
        - ``cache``: ``[1, num_codebooks, CT]`` token cache (int64) that
          captures the circular buffer state after the voice prompt was
          originally processed.

        Args:
            path: Path to the ``.pt`` file.
        """
        import torch

        logger.info("Loading voice prompt embeddings from: %s", path)
        state = torch.load(path, map_location="cpu", weights_only=False)

        # Convert embeddings: bfloat16 -> float32 -> numpy -> MLX float16
        emb_tensor = state["embeddings"].float().numpy()
        self.voice_prompt_embeddings = mx.array(emb_tensor, dtype=mx.float16)
        logger.info(
            "Voice prompt embeddings: shape=%s",
            self.voice_prompt_embeddings.shape,
        )

        # Convert cache: int64 -> numpy -> MLX int32
        cache_tensor = state["cache"].long().numpy()
        self.voice_prompt_cache = mx.array(cache_tensor, dtype=mx.int32)
        logger.info(
            "Voice prompt cache: shape=%s",
            self.voice_prompt_cache.shape,
        )

        self.voice_prompt = path
        self.voice_prompt_audio = None

    # ------------------------------------------------------------------
    # Voice prompt injection: embeddings path
    # ------------------------------------------------------------------

    def step_with_embedding(self, embedding: mx.array) -> None:
        """Feed a single pre-computed embedding through the transformer.

        This bypasses the normal token-embedding step in ``_sample`` and
        feeds the pre-computed embedding directly into the transformer,
        then runs out_norm, text_linear, and depformer sampling.

        The gen_sequence is updated as follows:
        - User audio (other) codebooks: SINE tokens at current step
        - Text codebook: PAD (3) at current step
        - Agent audio codebooks: depformer-sampled tokens (delay-compensated)

        Args:
            embedding: ``[1, 1, d_model]`` pre-computed embedding.
        """
        lm_gen = self.lm_gen
        model = self.model

        if lm_gen.step_idx >= lm_gen.max_steps:
            raise ValueError(f"reached max-steps {lm_gen.max_steps}")

        # Write SINE tokens as other_audio_tokens at current step
        sine = mx.array(SINE_TOKENS, dtype=mx.int32)
        sine = mx.broadcast_to(sine[None, :], (self.batch_size, self.other_q))
        lm_gen.gen_sequence[:, 1 + lm_gen.main_codebooks :, lm_gen.step_idx] = sine

        # Feed embedding directly through transformer (bypassing token embedding)
        transformer_out = model.transformer(
            embedding, cache=model.transformer_cache
        )
        transformer_out = model.out_norm(transformer_out)
        text_logits = model.text_linear(transformer_out)
        text_token, _ = lm_gen.text_sampler(text_logits)

        # Run depformer to sample agent audio tokens
        audio_tokens = model.depformer.sample(
            transformer_out,
            lm_gen.audio_sampler,
            text_token,
            model.depformer_cache,
            cfg_coef=lm_gen.cfg_coef,
        )

        # Write to gen_sequence: text = PAD (3)
        lm_gen.gen_sequence[:, 0, lm_gen.step_idx] = mx.array(
            [ZERO_TEXT_CODE] * self.batch_size, dtype=mx.int32
        )

        # Write agent audio tokens with delay compensation
        if audio_tokens is not None:
            for cb_idx, delay in enumerate(
                lm_gen.audio_delays[: lm_gen.main_codebooks]
            ):
                gen_idx = lm_gen.step_idx - delay
                if gen_idx >= 0:
                    lm_gen.gen_sequence[:, cb_idx + 1, gen_idx] = audio_tokens[
                        :, cb_idx, 0
                    ]

        lm_gen.step_idx += 1

    def _restore_voice_prompt_cache(self) -> None:
        """Overwrite the last CT positions in gen_sequence with the saved cache.

        After replaying all voice prompt embeddings through the transformer,
        the KV cache is correct but the gen_sequence tokens may differ from
        the original processing (because the depformer re-sampled them).
        The saved ``voice_prompt_cache`` from the ``.pt`` file contains the
        exact token values from the original processing.  This method
        maps the PyTorch circular buffer ``[1, 17, CT]`` to the corresponding
        linear positions in the MLX gen_sequence ``[1, 17, max_steps]``.
        """
        if self.voice_prompt_cache is None:
            return

        saved_cache = self.voice_prompt_cache  # [1, num_codebooks, CT]
        CT = saved_cache.shape[2]
        step_idx = self.lm_gen.step_idx

        logger.info(
            "Restoring voice prompt cache: CT=%d, step_idx=%d", CT, step_idx
        )

        for p in range(CT):
            mlx_step = step_idx - CT + p
            if mlx_step >= 0:
                self.lm_gen.gen_sequence[:, :, mlx_step] = saved_cache[:, :, p]

    def _step_voice_prompt_embeddings(self) -> None:
        """Replay all pre-computed embeddings through the transformer.

        For each embedding in ``voice_prompt_embeddings``, calls
        ``step_with_embedding`` to populate the transformer KV cache.
        After all embeddings are processed, restores the saved token
        cache to gen_sequence for exact parity with PyTorch.
        """
        if self.voice_prompt_embeddings is None:
            return

        num_embeddings = self.voice_prompt_embeddings.shape[0]
        logger.info(
            "Replaying %d voice prompt embeddings through transformer",
            num_embeddings,
        )

        for i in range(num_embeddings):
            # voice_prompt_embeddings shape: [N, 1, 1, d_model]
            # Index by i to get [1, 1, d_model] (B=1, T=1, d_model)
            emb = self.voice_prompt_embeddings[i]  # [1, 1, d_model]
            self.step_with_embedding(emb)

        # Force materialization before cache restore
        mx.eval(self.lm_gen.gen_sequence)

        # Overwrite gen_sequence with saved token cache
        self._restore_voice_prompt_cache()
        mx.eval(self.lm_gen.gen_sequence)

        logger.info(
            "Voice prompt embeddings replay complete. step_idx=%d",
            self.lm_gen.step_idx,
        )

    # ------------------------------------------------------------------
    # Voice prompt injection: audio WAV path
    # ------------------------------------------------------------------

    def _step_voice_prompt_frame(self, voice_tokens: mx.array) -> None:
        """Step with voice prompt tokens as agent audio.

        Runs a normal step with SINE as user audio, then overrides:
        - Text codebook with PAD (3)
        - Agent audio codebooks with the provided voice_tokens

        Args:
            voice_tokens: ``[B, dep_q]`` agent audio tokens from encoding
                one frame of the voice prompt.
        """
        # Step with SINE as user audio
        sine = mx.array(SINE_TOKENS, dtype=mx.int32)
        sine = mx.broadcast_to(sine[None, :], (self.batch_size, self.other_q))
        self.lm_gen._step(sine)

        # Override text with PAD
        step = self.lm_gen.step_idx - 1
        self.lm_gen.gen_sequence[:, 0, step] = mx.array(
            [ZERO_TEXT_CODE] * self.batch_size, dtype=mx.int32
        )

        # Override agent audio with voice prompt tokens (delay-compensated)
        for cb_idx, delay in enumerate(
            self.lm_gen.audio_delays[: self.dep_q]
        ):
            gen_idx = step - delay
            if gen_idx >= 0:
                self.lm_gen.gen_sequence[:, cb_idx + 1, gen_idx] = (
                    voice_tokens[:, cb_idx]
                )

    def _step_voice_prompt_audio(self, audio_tokenizer) -> None:
        """Encode the loaded WAV voice prompt and step frame-by-frame.

        For each frame of the voice prompt audio:
        1. Encode with rustymimi to get 8 agent audio codebook tokens
        2. Call ``_step_voice_prompt_frame`` to inject them

        Args:
            audio_tokenizer: A ``rustymimi.Tokenizer`` instance.
        """
        if self.voice_prompt_audio is None:
            return

        audio = self.voice_prompt_audio  # [1, T] numpy
        total_samples = audio.shape[-1]
        n_frames = total_samples // FRAME_SIZE
        logger.info(
            "Encoding voice prompt audio: %d frames (%.2fs)",
            n_frames,
            n_frames / FRAME_RATE,
        )

        # Reset audio tokenizer streaming state for clean encoding
        audio_tokenizer.reset()

        for i in range(n_frames):
            pcm = audio[:, i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
            # rustymimi expects [1, 1, FRAME_SIZE]
            pcm_input = pcm[0:1][np.newaxis, :, :]
            codes = audio_tokenizer.encode_step(pcm_input)
            # codes: [1, 1, num_codebooks] -> [1, num_codebooks, 1]
            codes_np = np.array(codes)
            codes_mx = mx.array(codes_np).transpose(0, 2, 1)
            # Take first dep_q codebooks for agent audio, squeeze time dim
            voice_tokens = codes_mx[:, : self.dep_q, 0]  # [1, dep_q]
            self._step_voice_prompt_frame(voice_tokens)

        # Reset audio tokenizer after voice prompt encoding
        audio_tokenizer.reset()

        mx.eval(self.lm_gen.gen_sequence)
        logger.info(
            "Voice prompt audio encoding complete. step_idx=%d",
            self.lm_gen.step_idx,
        )

    # ------------------------------------------------------------------
    # Audio silence (post-prompt padding)
    # ------------------------------------------------------------------

    def step_audio_silence(
        self, n_frames: int = AUDIO_SILENCE_FRAME_CNT
    ) -> None:
        """Step through silence frames between prompts.

        Matches PyTorch ``_step_audio_silence_core``:
        - Agent audio = SILENCE tokens
        - User audio = SINE tokens
        - Text = PAD (3)

        Args:
            n_frames: Number of silence frames (default 6 = 0.5s).
        """
        for _ in range(n_frames):
            # Step with SINE as user audio
            sine = mx.array(SINE_TOKENS, dtype=mx.int32)
            sine = mx.broadcast_to(
                sine[None, :], (self.batch_size, self.other_q)
            )
            self.lm_gen._step(sine)

            step = self.lm_gen.step_idx - 1

            # Override text with PAD
            self.lm_gen.gen_sequence[:, 0, step] = mx.array(
                [ZERO_TEXT_CODE] * self.batch_size, dtype=mx.int32
            )

            # Override agent audio with SILENCE tokens
            silence = mx.array(SILENCE_TOKENS, dtype=mx.int32)
            for cb_idx, delay in enumerate(
                self.lm_gen.audio_delays[: self.dep_q]
            ):
                gen_idx = step - delay
                if gen_idx >= 0:
                    self.lm_gen.gen_sequence[
                        :, cb_idx + 1, gen_idx
                    ] = mx.broadcast_to(
                        silence[cb_idx : cb_idx + 1], (self.batch_size,)
                    )

    # ------------------------------------------------------------------
    # System prompt orchestration
    # ------------------------------------------------------------------

    def step_system_prompts(
        self,
        audio_tokenizer=None,
        text_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Run the full system prompt pipeline.

        Matches the PyTorch ``step_system_prompts`` orchestration:
        1. Voice prompt (embeddings or audio)
        2. Audio silence (6 frames = 0.5s)
        3. Text prompt
        4. Audio silence (6 frames = 0.5s)

        Args:
            audio_tokenizer: A ``rustymimi.Tokenizer`` instance (required
                for WAV voice prompts, not needed for embeddings).
            text_token_ids: Sentencepiece token IDs for the system text
                prompt. If None or empty, skips text prompt phase.
        """
        # Phase 1: Voice prompt
        if self.voice_prompt_embeddings is not None:
            logger.info("System prompt phase 1: voice prompt (embeddings)")
            self._step_voice_prompt_embeddings()
        elif self.voice_prompt_audio is not None:
            if audio_tokenizer is None:
                raise ValueError(
                    "audio_tokenizer is required for WAV voice prompts"
                )
            logger.info("System prompt phase 1: voice prompt (audio)")
            self._step_voice_prompt_audio(audio_tokenizer)
        else:
            logger.info("System prompt phase 1: no voice prompt loaded")

        # Phase 2: Post-voice-prompt silence
        logger.info("System prompt phase 2: audio silence (6 frames)")
        self.step_audio_silence()
        mx.eval(self.lm_gen.gen_sequence)

        # Phase 3: Text prompt
        if text_token_ids:
            logger.info(
                "System prompt phase 3: text prompt (%d tokens)",
                len(text_token_ids),
            )
            self.step_text_prompt(text_token_ids)
            mx.eval(self.lm_gen.gen_sequence)
        else:
            logger.info("System prompt phase 3: no text prompt")

        # Phase 4: Post-text-prompt silence
        logger.info("System prompt phase 4: audio silence (6 frames)")
        self.step_audio_silence()
        mx.eval(self.lm_gen.gen_sequence)

        logger.info(
            "System prompts complete. step_idx=%d", self.lm_gen.step_idx
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def step_idx(self) -> int:
        """Current step index in the inner LmGen."""
        return self.lm_gen.step_idx

    @property
    def prompt_step_count(self) -> int:
        """Estimate the number of steps consumed by the system prompt pipeline.

        This accounts for voice prompt frames (or embeddings), two silence
        phases (6 frames each), and placeholder for text prompt tokens.
        The actual text prompt token count must be added separately.
        """
        n = 0
        if self.voice_prompt_embeddings is not None:
            n += self.voice_prompt_embeddings.shape[0]
        elif self.voice_prompt_audio is not None:
            n += self.voice_prompt_audio.shape[-1] // FRAME_SIZE
        # Two silence phases
        n += 2 * AUDIO_SILENCE_FRAME_CNT
        return n
