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
- **Voice prompt** injection is stubbed for Stage 3.
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

from moshi_mlx.models.generate import LmGen
from moshi_mlx.models.lm import Lm
from moshi_mlx.utils.sampling import Sampler

logger = logging.getLogger(__name__)

# Token patterns for silence and a pure-tone sine wave, used as "user audio"
# placeholders during warmup, prompt injection, and post-prompt padding.
# These match the values in PersonaPlex's ``moshi.models.lm``.
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]


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
      voice-prompt loading (Stage 3 stub).

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

        # Voice prompt state (Stage 3 stubs)
        self.voice_prompt_embeddings: Optional[mx.array] = None

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
            Same as :meth:`step` — ``[B, dep_q+1, 1]`` or ``None``.
        """
        silence = mx.array(SILENCE_TOKENS, dtype=mx.int32)
        silence = mx.broadcast_to(silence[None, :], (self.batch_size, self.other_q))
        return self.step(silence)

    def step_sine(self) -> Optional[mx.array]:
        """Step with sine-wave tokens as user audio input.

        Returns:
            Same as :meth:`step` — ``[B, dep_q+1, 1]`` or ``None``.
        """
        sine = mx.array(SINE_TOKENS, dtype=mx.int32)
        sine = mx.broadcast_to(sine[None, :], (self.batch_size, self.other_q))
        return self.step(sine)

    # ------------------------------------------------------------------
    # Text prompt injection
    # ------------------------------------------------------------------

    def step_text_prompt(self, text_token_ids: list[int]) -> None:
        """Inject text prompt tokens by stepping with silence audio.

        Each text prompt token is force-written into the gen_sequence at
        codebook index 0 (the text channel) after stepping with silence
        as the user audio input.  This mirrors the PersonaPlex server's
        ``step_system_prompts`` logic for the text prompt phase.

        Args:
            text_token_ids: List of sentencepiece token IDs for the
                system prompt text.
        """
        for tid in text_token_ids:
            silence = mx.array(SILENCE_TOKENS, dtype=mx.int32)
            silence = mx.broadcast_to(
                silence[None, :], (self.batch_size, self.other_q)
            )
            # Step the model (ignore output — we override the text channel)
            self.lm_gen._step(silence)

            # Overwrite the text token that was just sampled
            step = self.lm_gen.step_idx - 1  # _step already incremented
            self.lm_gen.gen_sequence[:, 0, step] = mx.array(
                [tid] * self.batch_size, dtype=mx.int32
            )

    # ------------------------------------------------------------------
    # Voice prompt (Stage 3 stub)
    # ------------------------------------------------------------------

    def load_voice_prompt(self, path: str) -> None:
        """Load a voice prompt from disk (Stage 3 — not yet implemented).

        Args:
            path: Path to voice prompt file (.wav or .pt embeddings).
        """
        logger.warning(
            "Voice prompt loading is not yet implemented in MLX. "
            "Skipping voice prompt from: %s",
            path,
        )

    def load_voice_prompt_embeddings(self, path: str) -> None:
        """Load pre-computed voice prompt embeddings (Stage 3 stub).

        Args:
            path: Path to the ``.pt`` file with cached embeddings.
        """
        logger.warning(
            "Voice prompt embeddings loading is not yet implemented in MLX. "
            "Skipping: %s",
            path,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def step_idx(self) -> int:
        """Current step index in the inner LmGen."""
        return self.lm_gen.step_idx
