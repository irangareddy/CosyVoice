# CosyVoice Architecture & Methods

## System Overview

CosyVoice is a multilingual TTS system with three sequential components:

```
Text + Speaker → [LLM] → Tokens → [Flow] → Mel-Spec → [HiFiGAN] → Waveform
```

**Versions**: CosyVoice 1.0 (300M), CosyVoice 2.0 (0.5B streaming), CosyVoice 3.0 (latest)

---

## Architecture Components

### 1. LLM (Language Model)
**Purpose**: Convert text + speaker embedding → discrete speech tokens

**CosyVoice 1.0**:
- TransformerLM with Conformer text encoder
- 300M parameters
- Non-causal (requires full text)

**CosyVoice 2.0**:
- Qwen2-based architecture (0.5B parameters)
- Causal modeling for streaming
- Pretrained on BlankEN dataset

**Input**: Text tokens + speaker embedding (256-dim from CAM++)
**Output**: Discrete speech tokens (25 Hz frame rate)

**Location**: `cosyvoice/llm/llm.py`

---

### 2. Flow (Flow Matching)
**Purpose**: Convert discrete tokens → continuous mel-spectrograms

**Architecture**:
- Conditional flow matching model
- Diffusion-based decoder with ODE solver
- Duration predictor for alignment

**CosyVoice 2.0 Enhancement**:
- Causal flow decoder (chunk_size: 25)
- Enables streaming inference
- Maintains quality with lower latency

**Input**: Speech tokens (25 Hz)
**Output**: Mel-spectrogram (80 bins, 24000 Hz audio)

**Location**: `cosyvoice/flow/`

---

### 3. HiFiGAN (Vocoder)
**Purpose**: Convert mel-spectrogram → high-quality waveform

**Architecture**:
- HiFT (HiFiGAN with F0 predictor)
- Multi-period discriminator (MPD)
- Multi-resolution discriminator (MRD)

**Key Features**:
- F0 (pitch) conditioning for natural prosody
- GAN training with adversarial + feature matching losses
- 24000 Hz sample rate output

**Input**: Mel-spectrogram (80 bins) + F0 contour
**Output**: Waveform (24000 Hz, 16-bit)

**Location**: `cosyvoice/hifigan/`

---

## Inference Methods

### 1. `inference_sft()` - Supervised Fine-Tuning
**Use case**: Use preset speaker voices

**Parameters**:
- `tts_text`: Text to synthesize
- `spk_id`: Preset speaker ID (e.g., '中文女')

**Characteristics**:
- Fastest inference
- Consistent quality
- Limited to training speakers
- No speaker embedding needed

**Example**:
```python
for audio in cosyvoice.inference_sft(
    tts_text="Hello world",
    spk_id="中文女"
):
    save_audio(audio['tts_speech'])
```

---

### 2. `inference_zero_shot()` - Zero-Shot Voice Cloning
**Use case**: Clone any voice from 3-10s reference audio

**Parameters**:
- `tts_text`: Text to synthesize
- `prompt_text`: Transcription of reference audio
- `prompt_speech_16k`: Reference audio (16kHz tensor)

**Characteristics**:
- Most flexible
- Works with any speaker
- Requires reference audio + transcription
- Speaker embedding extracted via CAM++

**Example**:
```python
ref_audio = load_wav('reference.wav', 16000)
for audio in cosyvoice.inference_zero_shot(
    tts_text="Hello world",
    prompt_text="Reference text",
    prompt_speech_16k=ref_audio
):
    save_audio(audio['tts_speech'])
```

**Method explored**: Used for baseline pretrained HiFiGAN samples

---

### 3. `inference_instruct2()` - Instruction-Based Synthesis
**Use case**: Control emotion, style, dialect via natural language

**Parameters**:
- `tts_text`: Text to synthesize
- `instruct_text`: Instruction (e.g., "Speak with anger")
- `prompt_speech_16k`: Reference audio (optional)

**Characteristics**:
- Emotion control without emotion labels
- Style transfer (formal, casual, etc.)
- Combines instruction + reference
- More natural than discrete emotion classes

**Example**:
```python
ref_audio = load_wav('angry_ref.wav', 16000)
for audio in cosyvoice.inference_instruct2(
    tts_text="You broke everything!",
    instruct_text="Speak with an angry emotion, clear and expressive",
    prompt_speech_16k=ref_audio
):
    save_audio(audio['tts_speech'])
```

**Method explored**: Used for emotion sample generation (epochs 0, 4, pretrained)

---

### 4. `inference_cross_lingual()` - Cross-Lingual Synthesis
**Use case**: Synthesize language X with voice from language Y

**Parameters**:
- `tts_text`: Text in target language
- `prompt_speech_16k`: Reference in source language

**Characteristics**:
- Transfer speaker identity across languages
- Maintains speaker characteristics
- Language-agnostic speaker encoder
- Quality depends on phonetic similarity

**Example**:
```python
# English voice speaking Chinese
english_ref = load_wav('english_speaker.wav', 16000)
for audio in cosyvoice.inference_cross_lingual(
    tts_text="你好世界",  # Chinese
    prompt_speech_16k=english_ref  # English
):
    save_audio(audio['tts_speech'])
```

**Method explored**: Not used in emotional TTS experiments

---

## Training Strategies

### Component-Wise Training
**Approach**: Train LLM, Flow, HiFiGAN independently

**Benefits**:
- Faster iteration (train only what needs improvement)
- More stable (avoid joint optimization complexity)
- Easier debugging (isolate component issues)

**Drawbacks**:
- Components may not be optimally aligned
- Requires more engineering

---

### Fine-Tuning Strategies

#### 1. LLM Fine-Tuning
**Goal**: Improve text-to-token modeling for specific domains/languages

**Config**:
```yaml
model: llm
lr: 1e-5  # Lower than pretrain
scheduler: constantlr
use_spk_embedding: true  # SFT mode
```

**Status**: Not explored (pretrained LLM sufficient)

---

#### 2. Flow Fine-Tuning
**Goal**: Improve token-to-mel for specific prosody/style

**Config**:
```yaml
model: flow
lr: 1e-4  # Attempt 1 - FAILED (gradient explosion)
lr: 2e-5  # Attempt 2 - FAILED (divergence)
grad_clip: 0.5
```

**Status**: Failed (unstable on 14k samples)
**Lesson**: Requires >50k utterances or freeze encoder layers

---

#### 3. HiFiGAN Fine-Tuning ✅
**Goal**: Improve mel-to-waveform for emotional expressiveness

**Config**:
```yaml
model: hifigan
lr: 0.0002  # Standard GAN
scheduler: constantlr
grad_clip: 5.0
accum_grad: 1  # CRITICAL: No accumulation
```

**Status**: SUCCESS (CV loss 100.71 → 80.59 by epoch 4)
**Lesson**: Most stable component to fine-tune

---

## Method Comparison

| Method | Pretrained LLM | Pretrained Flow | Pretrained HiFiGAN | Fine-tuned Component | Stability | Data Needed | Result |
|--------|----------------|-----------------|-------------------|---------------------|-----------|-------------|--------|
| Full fine-tune | ❌ | ❌ | ❌ | All | Low | >100k | Not attempted |
| LLM-only | ❌ | ✅ | ✅ | LLM | Medium | >50k | Not attempted |
| Flow-only | ✅ | ❌ | ✅ | Flow | Low | >50k | ❌ Failed (diverged) |
| HiFiGAN-only | ✅ | ✅ | ❌ | HiFiGAN | High | 14k | ✅ **SUCCESS** |
| LLM + HiFiGAN | ❌ | ✅ | ❌ | LLM, HiFiGAN | Medium | >50k | Not attempted |

**Chosen approach**: HiFiGAN-only fine-tuning (most stable, works with small emotional datasets)

---

## Key Decisions

### 1. Why HiFiGAN-Only Fine-Tuning?
- **Stability**: GAN training more robust than flow matching
- **Data efficiency**: Works with 14k samples (Flow needs >50k)
- **Faster iteration**: ~20 hours for 30 epochs vs. days for full pipeline
- **Sufficient quality**: Vocoder quality directly impacts perceived emotion

### 2. Why `inference_instruct2()` for Evaluation?
- **Natural emotion control**: Instruction text more intuitive than labels
- **Combines speaker + emotion**: Reference audio + instruction = target output
- **Consistent with training**: ESD dataset has emotion labels, can map to instructions
- **Fair comparison**: Same method for pretrained vs. fine-tuned

### 3. Why Pretrained Baseline?
- **Objective quality reference**: Shows what's possible with well-trained HiFiGAN
- **Diagnose issues**: Separates fine-tuning problems from pipeline issues
- **Realistic expectations**: Epoch 4 won't match 200-epoch pretrained model

---

## Practical Takeaways

### For Emotional TTS:
1. Fine-tune HiFiGAN only (pretrained LLM + Flow sufficient)
2. Use `inference_instruct2()` for emotion control
3. Need 10-20 epochs minimum for usable quality
4. Generate comparison samples with pretrained baseline

### For Other Domains:
- **New language**: Fine-tune LLM (if not in pretrain) + HiFiGAN
- **New style**: Fine-tune HiFiGAN only (prosody in mel-spec)
- **New speaker**: Use zero-shot inference (no fine-tuning needed)
- **Production quality**: Train all components jointly (if >100k data)
