# Deep Research Report: PersonaPlex, Moshi, and DualMind+1

**Date:** January 29, 2026
**Scope:** Architecture, training data, prompting strategies, generation parameters, and techniques for eliciting interesting model behavior

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Moshi: The Foundation Model](#2-moshi-the-foundation-model)
   - 2.1 [Architecture Overview](#21-architecture-overview)
   - 2.2 [Helium: The Text LLM Backbone](#22-helium-the-text-llm-backbone)
   - 2.3 [Mimi: The Neural Audio Codec](#23-mimi-the-neural-audio-codec)
   - 2.4 [RQ-Transformer: Temporal + Depth](#24-rq-transformer-temporal--depth)
   - 2.5 [Inner Monologue](#25-inner-monologue)
   - 2.6 [Acoustic Delay Mechanism](#26-acoustic-delay-mechanism)
3. [Moshi Training Data](#3-moshi-training-data)
   - 3.1 [Helium Text Pre-training Data](#31-helium-text-pre-training-data)
   - 3.2 [Unsupervised Audio Data](#32-unsupervised-audio-data)
   - 3.3 [Fisher Dataset](#33-fisher-dataset)
   - 3.4 [Supervised Multi-Stream Dataset](#34-supervised-multi-stream-dataset)
   - 3.5 [Synthetic Instruction Data](#35-synthetic-instruction-data)
   - 3.6 [Mimi Codec Training Data](#36-mimi-codec-training-data)
4. [Moshi Training Procedure](#4-moshi-training-procedure)
5. [PersonaPlex: NVIDIA's Extension](#5-personaplex-nvidias-extension)
   - 5.1 [What PersonaPlex Adds](#51-what-personaplex-adds)
   - 5.2 [PersonaPlex Architecture](#52-personaplex-architecture)
   - 5.3 [PersonaPlex Training Data](#53-personaplex-training-data)
   - 5.4 [Voice Conditioning System](#54-voice-conditioning-system)
   - 5.5 [Text Prompt System](#55-text-prompt-system)
   - 5.6 [PersonaPlex Evaluation Results](#56-personaplex-evaluation-results)
6. [DualMind+1: Conference Architecture](#6-dualmind1-conference-architecture)
   - 6.1 [What DualMind+1 Does](#61-what-dualmind1-does)
   - 6.2 [Architecture: Two-Agent Conference](#62-architecture-two-agent-conference)
   - 6.3 [Audio Mixing Bridge](#63-audio-mixing-bridge)
   - 6.4 [Prompting Strategies](#64-prompting-strategies)
   - 6.5 [Preset Persona Prompts](#65-preset-persona-prompts)
   - 6.6 [Runtime Prompt Injection](#66-runtime-prompt-injection)
   - 6.7 [Tricks and Techniques](#67-tricks-and-techniques)
7. [Generation Parameters Reference](#7-generation-parameters-reference)
   - 7.1 [Core Sampling Parameters](#71-core-sampling-parameters)
   - 7.2 [Parameter Effects Guide](#72-parameter-effects-guide)
   - 7.3 [Special Token Values](#73-special-token-values)
8. [Best Practices for Prompting](#8-best-practices-for-prompting)
   - 8.1 [What the Training Data Tells Us](#81-what-the-training-data-tells-us)
   - 8.2 [Effective Prompt Patterns](#82-effective-prompt-patterns)
   - 8.3 [DualMind+1 Lessons](#83-dualmind1-lessons)
   - 8.4 [Parameter Tuning for Different Effects](#84-parameter-tuning-for-different-effects)
9. [PersonaPlex Repository Analysis](#9-personaplex-repository-analysis)
   - 9.1 [Repo Structure](#91-repo-structure)
   - 9.2 [Inference Pipeline](#92-inference-pipeline)
   - 9.3 [Performance Benchmarks](#93-performance-benchmarks)
10. [Appendices](#10-appendices)
    - A. [Complete Model Configuration](#a-complete-model-configuration)
    - B. [All Available Voice Prompts](#b-all-available-voice-prompts)
    - C. [Sources](#c-sources)

---

## 1. Executive Summary

This report covers three interconnected projects:

- **Moshi** (Kyutai, 2024): The foundational 7B-parameter speech-text model that introduced real-time full-duplex speech conversation. It processes two simultaneous audio streams (model + user) at 12.5 Hz using a hierarchical Temporal+Depth Transformer architecture, with an "inner monologue" mechanism that generates time-aligned text tokens as scaffolding for speech.

- **PersonaPlex** (NVIDIA, January 2026): A fine-tuned extension of Moshi that adds controllable voice identity and role-based text prompting. Trained on ~3,467 hours of blended data (1,217 hours Fisher English + 2,250 hours synthetic), it achieves state-of-the-art conversational dynamics while maintaining persona consistency.

- **DualMind+1** (dg1kjd, 2026): A conference-mode application that runs two independent PersonaPlex instances on separate GPUs, feeding each agent's audio output into the other's input. This creates emergent multi-agent conversations with natural turn-taking, interruptions, and backchanneling. A human participant can optionally join as the "+1."

The key insight across all three: **Moshi's full-duplex architecture, trained on real conversational data (Fisher corpus), produces emergent conversational behaviors (backchanneling, barge-ins, laughter, "umm"s) that make the model sound remarkably natural.** PersonaPlex adds persona control on top, and DualMind+1 exploits the full-duplex nature by connecting two instances in a feedback loop for entertainment.

---

## 2. Moshi: The Foundation Model

**Paper:** "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue" (arXiv:2410.00037)
**Authors:** Defossez, Mazare, Orsini, Royer, Perez, Jegou, Grave, Zeghidour
**Institution:** Kyutai (Paris-based non-profit AI lab)

### 2.1 Architecture Overview

Moshi solves three problems with traditional ASR→LLM→TTS pipelines:
1. **Latency** — cascade introduces multi-second delays
2. **Information loss** — text strips emotion, prosody, non-speech sounds
3. **Rigid turn-taking** — cannot handle overlaps, interruptions, or backchannels

The solution: cast spoken dialogue as **speech-to-speech generation** with three components:

```
┌─────────────────────────────────────────────────────┐
│                    Moshi System                      │
│                                                      │
│  ┌──────────┐   ┌─────────────────┐   ┌──────────┐ │
│  │  Mimi    │   │  RQ-Transformer │   │  Mimi    │ │
│  │ Encoder  │──▶│  (Helium 7B +   │──▶│ Decoder  │ │
│  │          │   │   Depformer)    │   │          │ │
│  └──────────┘   └─────────────────┘   └──────────┘ │
│   User audio      Text + Audio tokens   Agent audio │
└─────────────────────────────────────────────────────┘
```

### 2.2 Helium: The Text LLM Backbone

Helium is Kyutai's custom 7B autoregressive language model.

| Parameter | Value |
|---|---|
| Parameters | 7 billion |
| Layers | 32 |
| Model dimension | 4,096 |
| MLP dimension | 11,264 |
| Attention heads | 32 |
| Context length | 4,096 tokens |
| Vocabulary | 32,000 (SentencePiece unigram) |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Attention | FlashAttention |
| Activation | GLU with SiLU |
| MMLU Score | 54.3% (outperforms Llama 2, Falcon, MPT in 7B class) |

### 2.3 Mimi: The Neural Audio Codec

Mimi converts 24 kHz audio into discrete tokens at 12.5 Hz — an 80ms frame rate enabling real-time streaming.

| Parameter | Value |
|---|---|
| Input sample rate | 24 kHz |
| Frame rate | 12.5 Hz (80ms per frame) |
| Bitrate | 1.1 kbps |
| Streaming latency | 80ms |
| Latent dimension | 512 |
| Quantizers | 8 (1 semantic + 7 acoustic) |
| Codebook size | 2,048 per quantizer |
| Architecture | SeaNet autoencoder with causal convolutions |
| Encoder strides | (4, 5, 6, 8) + final stride 2 |
| Transformer | 8 layers, 8 heads, 250-frame context (20s) |

**Split-RVQ architecture:** The first codebook captures semantic content (distilled from WavLM self-supervised features with 100x loss weight), while codebooks 2-8 capture acoustic details (voice, prosody, emotion) via residual vector quantization with 1x loss weight. This separation is critical — it lets the model reason about *what* is being said separately from *how* it sounds.

### 2.4 RQ-Transformer: Temporal + Depth

The generation process is factored into two dimensions:

**Temporal Transformer (large — 7B):**
- Identical architecture to Helium, initialized from Helium weights
- Processes the full conversation history across time at 12.5 Hz
- At each time step, receives the sum of embeddings from all 17 input sub-sequences
- Produces a temporal context vector z_s and text logits

**Depth Transformer (small — ~100M):**
- 1,024 dim, 6 layers, 16 heads
- Per-codebook linear layers and embeddings
- Context length of 8-9 (one per codebook)
- Takes temporal context z_s plus previously generated codebook tokens
- Autoregressively generates the 8 audio codebook tokens for a single time step

This factorization means the expensive 7B backbone runs only **once per 80ms frame**, while the lightweight Depformer runs 8 times (once per codebook). This makes real-time inference possible.

**Per time step, the model processes 17 sub-sequences:**

| Index | Stream | Content |
|---|---|---|
| 0 | Text | Moshi's inner monologue token |
| 1 | Moshi audio | Semantic codebook (VQ) |
| 2-8 | Moshi audio | Acoustic codebooks (RVQ, delayed by τ) |
| 9 | User audio | Semantic codebook (VQ) |
| 10-16 | User audio | Acoustic codebooks (RVQ, delayed by τ) |

### 2.5 Inner Monologue

Inner Monologue is Moshi's most important architectural innovation. At each time step, before generating audio tokens, the model first predicts a **text token** representing the word currently being spoken.

**How it works:**
- Text tokens are derived from Whisper transcriptions with word-level timestamps, mapped to the 12.5 Hz frame rate
- ~35% of frames contain actual text tokens; ~65% are PAD/EPAD markers
- The text token is generated by the Temporal Transformer and fed to the Depth Transformer as the first input, anchoring subsequent audio generation

**Why it matters:**
1. Dramatically improves linguistic quality by grounding speech in text-domain knowledge
2. Prevents garbled speech and hallucinations
3. Enables streaming ASR (sample only text tokens, use ground-truth audio)
4. Enables streaming TTS (feed text ahead of audio)
5. Works per-frame, compatible with real-time generation (unlike SpeechGPT/Spectron which need full text first)

### 2.6 Acoustic Delay Mechanism

A delay of τ=1-2 steps between semantic and acoustic tokens lets the Temporal Transformer process semantic content before acoustic details need prediction. This improves quality substantially.

- Pre-training: τ=2 (160ms acoustic delay)
- Post-training/inference: τ=1 (80ms acoustic delay)
- **Total theoretical latency: 160ms** (80ms frame + 80ms delay)
- **Practical latency: ~200ms** (on L4 GPU or Apple M3)

For comparison, average human conversation turn-taking latency is ~230ms.

---

## 3. Moshi Training Data

Understanding the training data is essential for effective prompting — the model can only exhibit behaviors it was exposed to during training.

### 3.1 Helium Text Pre-training Data

**Volume:** 2.1 trillion tokens of English text

| Source | Proportion | Details |
|---|---|---|
| Wikipedia | Part of 12.5% curated | 5 dumps (2017-2022) |
| Wikibooks, Wikisource, Wikinews | Part of 12.5% curated | Multiple dumps |
| StackExchange | Part of 12.5% curated | Q&A data |
| Scientific articles (peS2o) | Part of 12.5% curated | Large collection |
| Filtered CommonCrawl | 87.5% | 10 crawls (2018-2023) |

**Processing pipeline:** FNV-1a line-level deduplication → fastText fuzzy dedup → English-only filtering (threshold 0.85) → 9-category quality classifier.

### 3.2 Unsupervised Audio Data

- **7 million hours** of readily available English speech audio
- Transcribed with **Whisper large v3** for text alignment
- Used for Phase 1 pre-training (single audio stream only)
- Provides broad exposure to English speech patterns, accents, and topics

### 3.3 Fisher Dataset

The Fisher English corpus is **the most important dataset for Moshi's conversational abilities.**

| Attribute | Value |
|---|---|
| Source | LDC2004T19 (Fisher English Training Speech) |
| Conversations | 7,303 telephone conversations |
| Duration | ~2,000 hours (1,217 hours actively used by PersonaPlex) |
| Recording | Separate channels per speaker (critical for multi-stream training) |
| Original sample rate | 8 kHz |
| Processing | Upsampled to 24 kHz via AudioSR |
| Participants | Randomly paired speakers discussing assigned topics |

**Why Fisher matters:** It is the only dataset with real, naturally overlapping two-party conversations recorded on separate channels. This is what gives Moshi its ability to handle interruptions, backchannels, and overlapping speech. The separate channels provide ground-truth speaker separation that simulated data cannot match.

**Fisher topic categories** (what speakers discussed):
- Current events, hobbies, entertainment, sports
- Family and relationships
- Science and technology
- Politics and social issues
- General small talk and personal stories

This means Moshi has strong exposure to **casual conversational English** across diverse topics.

### 3.4 Supervised Multi-Stream Dataset

- **170 hours** of high-quality natural and scripted conversation
- Multiple pairs of participants, separate channel recording
- Used to train the TTS component (not Moshi directly, but the TTS that generates synthetic data for Phase 4)

### 3.5 Synthetic Instruction Data

| Component | Volume |
|---|---|
| Synthetic conversation scripts | ~100,000 "oral-style" dialogues |
| Synthetic speech audio | 20,000+ hours |
| Voice styles covered | 70+ emotions and speaking styles |
| Voice actor source | Single licensed voice actor ("Alice" for Moshika) |

**Types of synthetic training content:**

1. **Self-description:** Paragraphs about Moshi and Kyutai lab used as context — this is why Moshi "knows" it's an AI assistant built by Kyutai
2. **Voice instructions:** User asks Moshi to speak with specific emotions (angry, happy, pirate accent, etc.)
3. **Single-turn interactions:** Tell a sentence, monologue, or poem about a specific topic using a particular voice
4. **Mispronunciation robustness:** Instructions with misspellings, training Moshi to ask for clarification
5. **False facts:** Questions containing misleading information (e.g., "Is the Eiffel Tower in Beijing?"), teaching Moshi to correct
6. **Basic knowledge:** Math, grammar, trivia Q&A (noted as initially weak)

### 3.6 Mimi Codec Training Data

- Distillation from **WavLM** (self-supervised speech model) for semantic codebook
- WavLM processes 16 kHz audio → 1024-dim features at 50 Hz → downsampled to 12.5 Hz
- Cosine similarity loss (weight 100) between Mimi's first VQ and WavLM vectors

---

## 4. Moshi Training Procedure

Training proceeds through 4 phases on **1,016 H100 GPUs** (127 DGX nodes from Scaleway):

### Phase 1: Pre-training on Unsupervised Audio
| Attribute | Value |
|---|---|
| Steps | 1,000,000 |
| Batch size | 16 hours of audio |
| Sequence length | 5 minutes per item |
| LR (Temporal) | 3 × 10⁻⁵ |
| LR (Depth) | 2 × 10⁻⁴ |
| Acoustic delay | τ = 2 |
| Text preservation | 50% of batches are text-only (from Helium data) |
| Text embed LR scaling | 0.75× for text in audio batches |
| Padding loss weight | 0.5× for PAD tokens |

### Phase 2: Post-training with Simulated Multi-stream
| Attribute | Value |
|---|---|
| Steps | 100,000 |
| Batch size | 8 hours of audio |
| LR | 5 × 10⁻⁶ |
| Acoustic delay | τ = 1 |
| Diarization | PyAnnote speaker separation |
| Limitation | No overlapping speech; silent inactive channel |

### Phase 3: Fisher Fine-tuning (Real Multi-stream)
| Attribute | Value |
|---|---|
| Steps | 10,000 |
| Batch size | 40 minutes of audio |
| LR | 4 × 10⁻⁶ |
| Source | Fisher English corpus (2,000 hours, 8kHz → 24kHz via AudioSR) |
| Key feature | Real overlapping speech on separate channels |

### Phase 4: Instruction Fine-tuning (Synthetic)
| Attribute | Value |
|---|---|
| Steps | 30,000 |
| Batch size | 2.7 hours of audio |
| LR | 2 × 10⁻⁶ |
| Source | 100K synthetic conversations → 20K+ hours of TTS audio |
| User stream augmentation | Random gain (-24 to +15 dB) 50% of time; DNS noise 30% of time |

---

## 5. PersonaPlex: NVIDIA's Extension

**Paper:** "PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models" (ICASSP 2026)
**Authors:** Roy, Raiman, Lee, Ene, Kirby, Kim, Kim, Catanzaro (NVIDIA)
**Release:** January 15, 2026

### 5.1 What PersonaPlex Adds

Base Moshi has a **fixed persona** baked in during training. PersonaPlex adds two runtime control mechanisms:

1. **Voice prompting** — audio embeddings that control vocal characteristics
2. **Text prompting** — natural language role descriptions that control behavior

The key challenge PersonaPlex solves: traditional ASR→LLM→TTS pipelines allow easy voice/role customization but produce rigid, unnatural conversations. Full-duplex models like Moshi produce natural conversations but lock you into a fixed persona. PersonaPlex breaks this trade-off.

### 5.2 PersonaPlex Architecture

PersonaPlex uses Moshi's exact architecture (7B parameters) with the same Temporal + Depth Transformer structure. The model weights are initialized from Moshi's pre-trained checkpoint, then fine-tuned with persona-supervised data.

**Key architectural constants (from codebase):**

```python
# Main Transformer
d_model = 4096
num_heads = 32
num_layers = 32
dim_feedforward = 16896   # Note: 4.125 × 4096, not 4×
context = 3000
positional_embedding = "rope"
norm = "rms_norm"

# Depth Transformer (Depformer)
d_model = 1024
num_heads = 16
num_layers = 6
dim_feedforward = 4224    # 4.125 × 1024
context = 8
num_slices = 8

# Vocabularies
audio_vocab_size = 2049   # 2048 + 1
text_in_vocab_size = 32001
text_out_vocab_size = 32000
audio_codebooks = 16      # training
audio_codebooks_inference = 8  # inference (depformer slices)
```

### 5.3 PersonaPlex Training Data

PersonaPlex uses **under 5,000 hours of directed data** on top of Moshi's pre-trained weights:

| Dataset | Source | Hours | Conversations |
|---|---|---|---|
| Fisher English (real) | LDC2004T19 | 1,217 | 7,303 |
| Synthetic Assistant | Qwen3-32B + GPT-OSS-120B → Chatterbox TTS | 410 | 39,322 |
| Synthetic Customer Service | Qwen3-32B + GPT-OSS-120B → Chatterbox TTS | 1,840 | 105,410 |
| **Total** | | **~3,467** | **~152,035** |

**Fisher back-annotation process:**
- GPT-OSS-120B (an LLM) retrospectively generates contextual and personality descriptors for each Fisher speaker
- This creates persona-supervised training pairs: (voice, text prompt, conversation)
- Different detail levels are used, from brief hints to detailed backstories

**Synthetic data generation:**
- Dialogue scripts generated by Qwen3-32B and GPT-OSS-120B
- Audio synthesized using **Chatterbox TTS** (by Resemble AI) with synthetic voices from **TortoiseTTS**
- Assistant role: question-answering focused conversations
- Customer service role: domain-specific scenarios (waste management, appliance repair, etc.)

**Key training insight:** Real Fisher data provides natural speech patterns (backchanneling, interruptions, emotional expression) while synthetic data provides task-following behavior. Blending them via the shared prompt format bridges naturalness and task adherence.

### 5.4 Voice Conditioning System

PersonaPlex ships with **16 pre-computed voice embeddings** (`.pt` files):

**Natural voices (more conversational):**
- Female: NATF0, NATF1 (Sophia), NATF2 (Elena), NATF3 (Maya)
- Male: NATM0, NATM1 (Marcus), NATM2 (James), NATM3 (Oliver)

**Variety voices (more diverse characteristics):**
- Female: VARF0, VARF1, VARF2, VARF3, VARF4
- Male: VARM0, VARM1, VARM2, VARM3, VARM4

**Voice prompt processing pipeline:**
1. Load voice prompt embeddings or encode voice WAV (normalized to -24 LUFS)
2. Step through transformer, populating KV cache
3. Add 6 frames of silence (~0.5 seconds)
4. Inject text prompt tokens
5. Add 6 more frames of silence
6. Begin conversation

The silence gaps between voice and text conditioning are important — they let the model "settle" its internal state between different conditioning signals.

### 5.5 Text Prompt System

Text prompts define the agent's personality, role, and behavioral constraints. They are wrapped with system tags:

```
<system> {user_prompt} <system>
```

Note: both opening and closing tags are `<system>` (not `</system>`). This is the format PersonaPlex was trained with.

**Text prompt injection mechanism:**
1. Text is tokenized via SentencePiece (32K vocabulary)
2. During system prompt phase, model is fed silence audio (zero PCM frames)
3. Text tokens are force-fed one at a time, overriding the model's own text predictions
4. Agent audio output during this phase uses SILENCE tokens (suppressed)
5. User audio channel is filled with SINE tokens (a reference signal)

**Example prompts from the training data / documentation:**

**Assistant (QA-focused):**
```
You are a wise and friendly teacher. Answer questions or provide advice
in a clear and engaging way.
```

**Customer Service (domain-specific):**
```
You work for CitySan Services which is a waste management and your name
is Ayelen Lucero. Information: Verify customer name Omar Torres. Current
schedule: every other week. Upcoming pickup: April 12th. Compost bin
service available for $8/month add-on.
```

**Casual Conversation:**
```
You enjoy having a good conversation. Have a casual discussion about
eating at home versus dining out.
```

**Out-of-Distribution (emergent generalization):**
```
You enjoy having a good conversation. Have a technical discussion about
fixing a reactor core on a spaceship to Mars. You are an astronaut on a
Mars mission. Your name is Alex. You are already dealing with a reactor
core meltdown. Several ship systems are failing...
```

The out-of-distribution example is particularly notable: PersonaPlex successfully retrieves technical vocabulary and domain reasoning absent from its training data, demonstrating emergent generalization.

### 5.6 PersonaPlex Evaluation Results

Evaluated on **FullDuplexBench** and **ServiceDuplexBench**:

| Metric | PersonaPlex | Moshi | Freeze Omni |
|---|---|---|---|
| Smooth Turn-Taking TOR | **0.908** | 0.87 | 0.72 |
| User Interruption TOR | **0.950** | 0.91 | 0.78 |
| Smooth TT Latency | **0.170s** | -- | -- |
| Interruption Latency | **0.240s** | -- | -- |
| Voice Similarity (WavLM) | **0.650** | N/A | N/A |

PersonaPlex outperforms Moshi, Freeze Omni, Gemini Live, and Qwen 2.5 Omni across conversational dynamics, latency, and task adherence.

---

## 6. DualMind+1: Conference Architecture

**Repository:** [github.com/dg1kjd/dualmind-plus-one](https://github.com/dg1kjd/dualmind-plus-one)
**Author:** Jens David Consulting (dg1kjd)
**License:** MIT (code), NVIDIA Open Model License (weights)

### 6.1 What DualMind+1 Does

DualMind+1 runs **two independent PersonaPlex instances** simultaneously on separate GPUs, feeding each agent's audio output into the other's input. An optional human participant joins via microphone (the "+1"). The result is a three-way full-duplex conversation where two AI agents and a human can all talk simultaneously.

This is described as "entertainment-focused" and produces full-duplex, sub-250ms voice-to-voice AI conversations with natural phenomena like back-channeling, barge-ins, laughter, "umm"s, and spontaneous exclamations.

### 6.2 Architecture: Two-Agent Conference

```
┌──────────────────┐                    ┌──────────────────┐
│   PersonaA       │                    │   PersonaB       │
│   (GPU 0)        │                    │   (GPU 1)        │
│                  │                    │                  │
│   Mimi Encoder   │    Audio Mix       │   Mimi Encoder   │
│   Temporal LM    │◄──────────────────▶│   Temporal LM    │
│   Depformer      │    Bridge          │   Depformer      │
│   Mimi Decoder   │   (12.5 Hz)       │   Mimi Decoder   │
└────────┬─────────┘                    └────────┬─────────┘
         │                                       │
         └───────────────┬───────────────────────┘
                         │
                    ┌────▼────┐
                    │  Human  │
                    │  (+1)   │
                    └─────────┘
```

**Key design decisions:**

1. **OS-level process isolation:** Each persona runs in a separate `mp.Process` with `spawn` start method, giving each its own CUDA context. This prevents GPU memory conflicts.

2. **Lock-free ring buffers:** Audio passes between processes through `SharedAudioBuffer` using `multiprocessing.Array(ctypes.c_float)` — a single-producer/single-consumer ring buffer avoiding serialization overhead.

3. **Previous-frame rule:** Each persona hears the *previous* frame's output from the other persona, creating a natural ~80ms delay that prevents feedback loops while maintaining conversational flow.

### 6.3 Audio Mixing Bridge

The `process_bridge()` coroutine runs at 12.5 Hz (matching the model's native frame rate) and mixes audio as follows:

```python
# PersonaA hears: 50% of PersonaB's output + user audio
mix_for_a = 0.5 * b_output + user_gain * user_input

# PersonaB hears: 50% of PersonaA's output + user audio
mix_for_b = 0.5 * a_output + user_gain * user_input

# User hears: normalized mix of both personas
user_output = normalize(a_output + b_output)
```

The 0.5 gain factors prevent clipping while maintaining audibility. In "micless mode" (`mic_disabled=1`), silence frames are generated on a timer instead of from microphone input, allowing pure AI-to-AI conversation.

### 6.4 Prompting Strategies

**System prompt format:** All text prompts are wrapped in `<system>` tags:

```python
def wrap_with_system_tags(text: str) -> str:
    return f"<system> {text.strip()} <system>"
```

**System prompt processing sequence:**
1. Voice conditioning (load `.pt` embeddings or encode WAV)
2. Silence gap (6 frames = ~0.5 seconds)
3. Text role prompt (force-fed token by token during silence audio)
4. Silence gap (6 frames)

**Default prompts:**
- Server default: `"You enjoy having a good conversation."`
- PersonaA default: `"You enjoy having a good conversation. You are curious and ask thoughtful questions."`
- PersonaB default: `"You enjoy having a good conversation. You share interesting stories and perspectives."`

**Default voices:**
- PersonaA: NATF1 (Sophia — female)
- PersonaB: NATM1 (Marcus — male)

Using different voice genders helps the model differentiate speakers in the cross-talk feedback loop.

### 6.5 Preset Persona Prompts

DualMind+1 includes **7 preset personality buttons** in its conference UI that showcase what works well:

| Preset | Full Prompt Text |
|---|---|
| **Philosopher** | "You are a deep thinker who loves exploring existential questions. You often quote philosophers and challenge assumptions." |
| **Comedian** | "You are a witty comedian who finds humor in everything. You love puns, wordplay, and absurdist observations." |
| **Police Detective** | "You are a grizzled police detective investigating a candy theft. A little girl reported her lollipop stolen." |
| **Candy Robber** | "You are a sneaky but not very bright candy thief. You definitely stole that little girl's lollipop but you're trying to act innocent." |
| **Poet** | "You are a romantic soul who sees beauty everywhere. You speak in metaphors and occasionally break into verse." |
| **ML Engineer** | "You are an ML engineer obsessed with transformers and gradient descent. You relate everything to neural networks." |
| **Psychologist** | "You are a practicing psychologist who can't stop analyzing everyone. You ask probing questions about childhood and feelings." |

The **Detective + Candy Robber** pairing is designed specifically for comedic interrogation scenarios between the two AIs. This illustrates a key strategy: **complementary, adversarial personas produce the most entertaining conversations.**

### 6.6 Runtime Prompt Injection

DualMind+1 supports mid-conversation prompt changes without resetting the session. The client sends a `0x0B` WebSocket message:

```javascript
sendPromptInjection(persona, promptText) {
    const payload = JSON.stringify({ persona, prompt: promptText });
    // message type 0x0B
}
```

The server processes this by temporarily swapping text prompt tokens and running `_step_text_prompt()` inline. This allows steering an ongoing conversation in new directions without losing accumulated context.

### 6.7 Tricks and Techniques

**1. Cross-Talk Audio Feedback Loop (Core Innovation)**
The most important technique: feeding each AI's output into the other's input. This exploits Moshi's full-duplex architecture — trained on real two-party conversations — to produce emergent multi-agent dynamics. The AIs naturally develop turn-taking, interruptions, and backchanneling because the underlying model learned these behaviors from the Fisher corpus.

**2. Complementary Persona Pairing**
Pairing personas with contrasting roles (Detective + Candy Robber, Philosopher + ML Engineer) creates natural conversational tension that drives more interesting dialogue.

**3. Different Voice Genders**
Using female (Sophia) and male (Marcus) default voices helps the model internally differentiate between the two speakers in the cross-talk, improving conversation quality.

**4. Previous-Frame Delay**
Using the previous frame's output as the current frame's input creates ~80ms communication delay — similar to natural conversation propagation delay — preventing echo/feedback while maintaining flow.

**5. Warm-Up Phase**
Before any session, 4 frames of dummy inference warm up CUDA graphs and stabilize latency.

**6. System Prompt Sequencing with Silence Gaps**
The careful ordering of voice-prompt → silence → text-prompt → silence gives the model time to stabilize internal state between conditioning signals. Skipping the silence gaps degrades quality.

**7. Micless Mode for Pure AI-to-AI**
Setting `mic_disabled=1` creates a self-sustaining conversation loop where both AIs talk to each other indefinitely with no human input, generating synthetic silence frames on a timer.

**8. Runtime Prompt Steering**
Changing prompts mid-conversation without resetting allows dynamic "plot development" in multi-agent scenarios.

---

## 7. Generation Parameters Reference

### 7.1 Core Sampling Parameters

| Parameter | Default | Range (UI) | Controls |
|---|---|---|---|
| `temp` (audio) | **0.8** | 0.2 - 1.2 | Audio token sampling temperature |
| `temp_text` | **0.7** | 0.2 - 1.2 | Text (inner monologue) token sampling temperature |
| `top_k` (audio) | **250** | 10 - 500 | Top-k candidates for audio tokens (out of 2,048 codebook) |
| `top_k_text` | **25** | 10 - 500 | Top-k candidates for text tokens (out of 32,000 vocab) |
| `repetition_penalty` | **1.0** | 1.0 - 2.0 | Repetition penalty (1.0 = disabled) |
| `repetition_penalty_context` | **64** | 0 - 200 | Window size for repetition penalty |
| `pad_mult` | **0** | -4 to 4 | Padding multiplier |
| `use_sampling` | **True** | -- | False = greedy argmax decoding |
| `cfg_coef` | **1.0** | -- | Classifier-free guidance (1.0 = disabled) |
| `seed` | **42424242** | -- | Random seed for reproducibility |
| `max_steps` | **3000** | -- | Maximum sequence length (~4 minutes) |

**Sampling implementation** (`sample_token` in `moshi/moshi/utils/sampling.py`):
```python
if use_sampling and temp > 0.0:
    probs = torch.softmax(logits / temp, dim=-1)
    if top_p > 0.0:
        next_token = sample_top_p(probs, p=top_p)
    elif top_k > 0:
        next_token = sample_top_k(probs, k=top_k)
    else:
        next_token = multinomial(probs, num_samples=1)
else:
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
```

Note: `top_p` is implemented but **not exposed in any UI or CLI** — only `top_k` is used in practice.

### 7.2 Parameter Effects Guide

| Goal | temp | temp_text | top_k | top_k_text | Notes |
|---|---|---|---|---|---|
| **Stable, predictable** | 0.6 | 0.5 | 100 | 15 | Good for customer service |
| **Default (balanced)** | 0.8 | 0.7 | 250 | 25 | Recommended baseline |
| **Creative, varied** | 0.9 | 0.8 | 300 | 50 | Good for entertainment |
| **Maximum chaos** | 1.1 | 1.0 | 400 | 100 | Unpredictable, possibly incoherent |
| **Greedy/deterministic** | 0.0 | 0.0 | 1 | 1 | Reproducible, monotone |
| **TTS-style** | 0.6 | 0.6 | 250 | 25 | With cfg_coef=3.0 (Kyutai recommendation) |

**Key observations:**
- Audio temperature affects vocal expressiveness — higher values produce more varied intonation, laughter, and non-verbal sounds
- Text temperature affects lexical diversity — higher values produce more creative word choices but risk incoherence
- Audio top_k=250 is generous (out of 2,048) — the model already learned which codebook entries are valid
- Text top_k=25 is restrictive (out of 32,000) — this keeps the model on-topic and prevents hallucination
- The asymmetry between audio and text top_k reflects the different nature of the two modalities

### 7.3 Special Token Values

```python
# Audio codebook tokens for silence (8 codebooks)
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]

# Audio codebook tokens for reference sine wave (8 codebooks)
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]

# Text tokens
ZERO_TEXT_CODE = 3               # PAD token
TEXT_VOCAB_SIZE = 32000
AUDIO_VOCAB_SIZE = 2049          # 2048 + 1

# Timing
SAMPLE_RATE = 24000              # Hz
FRAME_RATE = 12.5                # Hz
FRAME_SIZE = 1920                # samples per frame (24000 / 12.5)
AUDIO_SILENCE_FRAME_CNT = 6     # ~0.5 seconds at 12.5 Hz
```

---

## 8. Best Practices for Prompting

### 8.1 What the Training Data Tells Us

Understanding the training data composition reveals what the model responds to best:

**From Moshi's training:**
- The model was trained on casual English conversation (Fisher corpus) — it excels at informal, conversational tone
- It learned from 70+ speaking styles and emotions — it can express anger, happiness, excitement, and more
- It was exposed to voice instruction data — asking it to change tone/style mid-conversation can work
- It was trained on false-fact correction — it can push back on incorrect premises
- It was trained on self-description — it "knows" it's an AI (from Kyutai/Moshi context paragraphs)
- Its math/trivia abilities are noted as "initially weak" — don't rely on it for factual knowledge

**From PersonaPlex's training:**
- Fisher conversations were back-annotated with personality descriptors — the model understands persona prompts ranging from brief hints to detailed backstories
- Synthetic assistant data covers QA-style interactions — good at answering questions within a role
- Synthetic customer service data covers domain-specific scenarios — excellent at role-playing with specific company names, products, and policies
- The model generalizes beyond training domains — the astronaut/reactor example shows it can handle novel scenarios

### 8.2 Effective Prompt Patterns

Based on analysis of all three projects, here are the patterns that work best:

**Pattern 1: Short Personality + Behavioral Tendency**
```
You enjoy having a good conversation. You are curious and ask thoughtful questions.
```
Why it works: Matches the Fisher back-annotation style (brief personality descriptor + behavioral tendency).

**Pattern 2: Role + Name + Domain Context**
```
You work for SwiftPlex Appliances which is a appliance repair company
and your name is Farhod Toshmatov. Information: The dishwasher model
is out of stock for replacement parts; we can use an alternative part
with a 3-day delay. Labor cost remains $60 per hour.
```
Why it works: Directly matches the synthetic customer service training data format.

**Pattern 3: Character + Exaggerated Trait**
```
You are a witty comedian who finds humor in everything. You love puns,
wordplay, and absurdist observations.
```
Why it works: Vivid, single-trait characters are easier for the model to maintain consistency with. This matches the voice instruction training where specific styles were requested.

**Pattern 4: Scenario + Role + Active Situation**
```
You are a grizzled police detective investigating a candy theft.
A little girl reported her lollipop stolen.
```
Why it works: Places the model in an active situation (not just a static identity), giving it something concrete to react to and ask about.

**Pattern 5: Complementary Pair (for DualMind+1)**
```
PersonaA: "You are a police detective investigating a candy theft."
PersonaB: "You are a candy thief trying to act innocent."
```
Why it works: Creates natural conversational tension with opposing objectives, driving sustained back-and-forth.

**Anti-patterns to avoid:**
- Very long prompts (>200 tokens) — the text prompt system force-feeds tokens during silence, excessively long prompts delay conversation start
- Technical formatting (bullet points, URLs, code) — the model was explicitly noted to handle these poorly since TTS training data avoided them
- Requesting specific factual knowledge — the model's knowledge comes from Helium's text training, not the prompt
- Abstract instructions without concrete context — "Be helpful" is less effective than "You are a teacher named Jenny"

### 8.3 DualMind+1 Lessons

The DualMind+1 project reveals several key lessons about getting the most out of PersonaPlex/Moshi:

1. **The model is inherently conversational** — it doesn't need complex prompts to produce natural dialogue. The Fisher training data gave it strong conversational instincts.

2. **Emergent behavior from feedback loops** — connecting two instances produces behaviors neither was explicitly prompted for: laughter, "umm"s, false starts, topic drift. This suggests these are latent capabilities from training data.

3. **Visual/concrete scenarios outperform abstract ones** — "candy theft investigation" produces more engaging conversation than "discuss philosophy." The model responds better to specific, imaginable situations.

4. **Short prompts work** — DualMind+1's most entertaining presets are 1-2 sentences. The model fills in the details itself.

5. **Voice differentiation matters** — using distinct voices (male + female, or different voice embeddings) helps the model maintain separate identities in multi-agent settings.

6. **The defaults are good** — DualMind+1 uses PersonaPlex's default generation parameters (temp=0.8/0.7, top_k=250/25) without modification. The fun comes from the architecture and prompts, not exotic parameter tuning.

### 8.4 Parameter Tuning for Different Effects

**For entertainment/comedy:**
- Use default parameters (temp=0.8, temp_text=0.7)
- Choose vivid, exaggerated persona prompts
- Pair complementary/adversarial characters
- Different voice genders
- Allow the model to be creative — don't over-constrain

**For customer service simulation:**
- Lower text temperature (0.5-0.6) for more focused responses
- Include specific domain facts in the prompt
- Name the agent and the company
- Include key information points the agent should reference

**For maximum expressiveness:**
- Higher audio temperature (0.9-1.0) produces more varied intonation
- Can trigger laughter, exclamations, and emotional vocal variety
- Risk: higher temperatures also increase audio artifacts

**For stable, professional output:**
- Lower both temperatures (0.6/0.5)
- Lower audio top_k (100) for cleaner audio
- Lower text top_k (15) for more focused responses
- Use repetition_penalty=1.2 to prevent loops

---

## 9. PersonaPlex Repository Analysis

### 9.1 Repo Structure

```
personaplex/
├── README.md                          # Main documentation
├── Dockerfile                         # NVIDIA CUDA container
├── docker-compose.yaml                # Docker compose config
│
├── moshi/                             # Main Python package
│   ├── moshi/                         # PyTorch implementation
│   │   ├── server.py                  # WebSocket server (CUDA)
│   │   ├── offline.py                 # Offline inference pipeline
│   │   ├── models/
│   │   │   ├── loaders.py             # HuggingFace model loading
│   │   │   ├── lm.py                  # LMModel + LMGen (core inference)
│   │   │   ├── compression.py         # Mimi audio codec
│   │   ├── modules/
│   │   │   ├── transformer.py         # Streaming transformer + KV cache
│   │   │   ├── seanet.py              # SEANet audio encoder/decoder
│   │   │   ├── rope.py                # Rotary positional embeddings
│   │   ├── quantization/              # VQ implementation
│   │   ├── utils/
│   │       ├── sampling.py            # Top-k, temperature sampling
│   │
│   ├── moshi_mlx/                     # Apple Silicon (MLX) implementation
│   │   ├── server_mlx.py             # MLX WebSocket server
│   │   ├── offline_mlx.py            # MLX offline inference
│   │   ├── lm_gen_mlx.py             # PersonaPlexLmGen wrapper
│   │   ├── loaders_mlx.py            # MLX model loading + config
│   │   ├── convert_weights.py         # PyTorch → MLX weight conversion
│   │   ├── benchmark.py              # Performance benchmarking
│
├── client/                            # React/TypeScript web UI
│   ├── src/
│       ├── pages/Conversation/        # Main conversation interface
│       ├── protocol/                  # WebSocket binary protocol
│
├── assets/
    ├── test/
        ├── basic_prompt.txt           # "You are an assistant named Jenny..."
        ├── prompt_service.txt         # "You work for SwiftPlex Appliances..."
        ├── results/
            ├── PERFORMANCE_SUMMARY.md # Benchmark results
```

### 9.2 Inference Pipeline

**Per-frame (every 80ms at 12.5 Hz):**

```
User Microphone
    │
    ▼
Opus Decode (client→server)
    │
    ▼
Mimi Encode: PCM [1,1,1920] → 8 audio codes
    │
    ▼
LM Step: 17 input sub-sequences → 9 output tokens
    │     (1 text + 8 agent audio codes)
    ▼
Mimi Decode: 8 audio codes → PCM [1,1,1920]
    │
    ▼
Opus Encode (server→client)
    │
    ▼
Speaker Output
```

**WebSocket binary protocol:**

| Byte 0 | Type | Payload |
|---|---|---|
| 0x00 | Handshake | Server ready signal |
| 0x01 | Audio | Opus-encoded bytes |
| 0x02 | Text | UTF-8 text token |
| 0x03 | Control | Pause/restart |
| 0x04 | Metadata | JSON |
| 0x05 | Error | Error message |
| 0x06 | Ping | Keep-alive |

### 9.3 Performance Benchmarks

**Platform:** Apple Silicon M3, 96GB unified memory

| Stage | Test | P95 Latency | Within 80ms Budget |
|---|---|---|---|
| 1 | Forward pass (LM only) | 50.3ms | PASS |
| 1 | Full sample (LM + Depformer + sampling) | 60.5ms | PASS |
| 3 | Voice prompt + text prompt frame | 63.5ms | PASS |

**MLX vs PyTorch MPS speedup: 5.2-5.3×**

| Optimization | Impact |
|---|---|
| Flash Attention (MLX native) | 2-3× |
| Automatic fusion (lazy eval) | 1.5-2× |
| mx.compile on hot paths | 1.2-1.5× |
| Native bfloat16 | 1.2× |
| Unified memory (no transfers) | 1.1× |

**Quantization support:**
- int8: Reduced memory, minimal quality loss
- int4: ~50% memory reduction, some quality degradation
- Group size: 64 (default)

---

## 10. Appendices

### A. Complete Model Configuration

**PersonaPlex LM (personaplex_config):**
```python
{
    "transformer": {
        "d_model": 4096,
        "num_heads": 32,
        "num_layers": 32,
        "dim_feedforward": 16896,
        "causal": True,
        "norm_first": True,
        "bias_ff": False,
        "bias_attn": False,
        "positional_embedding": "rope",
        "context": 3000,
        "gating": True,
        "norm": "rms_norm"
    },
    "depformer": {
        "d_model": 1024,
        "num_heads": 16,
        "num_layers": 6,
        "dim_feedforward": 4224,
        "context": 8,
        "positional_embedding": "none",
        "num_slices": 8
    },
    "audio_vocab_size": 2049,
    "text_in_vocab_size": 32001,
    "text_out_vocab_size": 32000,
    "audio_codebooks": 16,
    "audio_delays": [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
}
```

**Mimi Audio Codec:**
```python
{
    "encoder/decoder": {
        "channels": 1,
        "dimension": 512,
        "causal": True,
        "n_filters": 64,
        "ratios": [8, 6, 5, 4],
        "kernel_size": 7,
        "activation": "ELU"
    },
    "transformer": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 8,
        "context": 250
    },
    "quantizer": {
        "dimension": 256,
        "n_q": 32,
        "bins": 2048
    }
}
```

### B. All Available Voice Prompts

| Voice ID | Name | Gender | Type |
|---|---|---|---|
| NATF0 | -- | Female | Natural |
| NATF1 | Sophia | Female | Natural |
| NATF2 | Elena | Female | Natural |
| NATF3 | Maya | Female | Natural |
| NATM0 | -- | Male | Natural |
| NATM1 | Marcus | Male | Natural |
| NATM2 | James | Male | Natural |
| NATM3 | Oliver | Male | Natural |
| VARF0 | -- | Female | Variety |
| VARF1 | -- | Female | Variety |
| VARF2 | -- | Female | Variety |
| VARF3 | -- | Female | Variety |
| VARF4 | -- | Female | Variety |
| VARM0 | -- | Male | Variety |
| VARM1 | -- | Male | Variety |
| VARM2 | -- | Male | Variety |
| VARM3 | -- | Male | Variety |
| VARM4 | -- | Male | Variety |

### C. Sources

**Papers:**
- [Moshi: A Speech-Text Foundation Model for Real-Time Dialogue (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models (ICASSP 2026)](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
- [FullDuplexBench: Evaluating Real-Time Full-Duplex Conversational AI (arXiv:2503.04721)](https://arxiv.org/abs/2503.04721)

**Model Repositories:**
- [NVIDIA PersonaPlex-7B-v1 on HuggingFace](https://huggingface.co/nvidia/personaplex-7b-v1)
- [NVIDIA PersonaPlex on GitHub](https://github.com/NVIDIA/personaplex)
- [Kyutai Moshi on GitHub](https://github.com/kyutai-labs/moshi)
- [Kyutai Moshi Fine-tuning](https://github.com/kyutai-labs/moshi-finetune)
- [Kyutai Moshiko PyTorch BF16 on HuggingFace](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
- [DualMind+1 on GitHub](https://github.com/dg1kjd/dualmind-plus-one)

**Blog Posts and Documentation:**
- [NVIDIA PersonaPlex Research Page](https://research.nvidia.com/labs/adlr/personaplex/)
- [Kyutai Moshi Release Blog](https://kyutai.org/blog/2024-09-18-moshi-release)
- [Kyutai Helium Blog](https://kyutai.org/blog/2025-04-30-helium)
- [Kyutai Neural Audio Codec Explainer](https://kyutai.org/codec-explainer)
- [HuggingFace Transformers Moshi Documentation](https://huggingface.co/docs/transformers/en/model_doc/moshi)

**Local Codebase (key files):**
- `moshi/moshi/models/lm.py` — LMModel + LMGen core inference
- `moshi/moshi/utils/sampling.py` — Token sampling implementation
- `moshi/moshi_mlx/lm_gen_mlx.py` — PersonaPlexLmGen wrapper
- `moshi/moshi_mlx/loaders_mlx.py` — Model configuration constants
- `moshi/moshi_mlx/server_mlx.py` — Real-time WebSocket server
- `assets/test/basic_prompt.txt` — Example assistant prompt
- `assets/test/prompt_service.txt` — Example service prompt

---

*Report generated from deep analysis of the PersonaPlex codebase, Moshi paper, PersonaPlex paper, and DualMind+1 repository.*
