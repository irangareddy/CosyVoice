# Dataset Documentation

**Generated**: 2025-12-01
**Dataset**: Emotional Speech Dataset for CosyVoice Fine-Tuning
**Purpose**: Comprehensive dataset reference for training and reproducibility

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Dataset Statistics](#dataset-statistics)
3. [Data Format](#data-format)
4. [Emotion Categories](#emotion-categories)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Quality Control](#quality-control)
7. [Usage Guidelines](#usage-guidelines)

---

## Dataset Overview

### Source Datasets

This dataset combines three publicly available emotional speech datasets:

**1. ESD (Emotional Speech Dataset)**
- **Files**: ~17,500 utterances
- **Speakers**: 20 speakers (10 English, 10 Chinese)
- **Emotions**: 10 emotions (neutral, happy, sad, angry, surprise, fear, disgust, contempt, boredom, frustration)
- **Languages**: English, Chinese
- **Status**: **INCLUDED** in fine-tuning

**2. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- **Files**: ~1,440 utterances
- **Speakers**: 24 professional actors (12 female, 12 male)
- **Emotions**: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Language**: North American English
- **Status**: **INCLUDED** in fine-tuning

**3. CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
- **Files**: ~7,442 utterances
- **Speakers**: 91 actors (diverse age, gender, ethnicity)
- **Emotions**: 6 emotions (angry, disgust, fear, happy, neutral, sad)
- **Language**: English
- **Status**: **EXCLUDED** from fine-tuning (filtered out)

### Dataset Rationale

**Why ESD + RAVDESS?**
- More diverse emotional coverage (10 vs 8 vs 6 emotions)
- Higher audio quality and clearer emotional expression
- Better speaker diversity (professional actors + native speakers)
- Sufficient data size (~15K samples) for fine-tuning

**Why exclude CREMA-D?**
- Focuses on 6 core emotions (subset of ESD/RAVDESS)
- Crowd-sourced data may have quality variations
- Reduces training time without sacrificing emotion coverage

---

## Dataset Statistics

### Overall Statistics

**Total Files**: 26,382 WAV files
**Total Size**: 4.4 GB (audio + manifests)
**Sample Rate**: 24,000 Hz (24kHz)
**Audio Format**: WAV (PCM)
**Channels**: Mono

### Split Distribution

| Split | Total Samples | ESD (est.) | RAVDESS (est.) | CREMA-D (filtered) |
|-------|--------------|------------|----------------|-------------------|
| **Train** | 20,884 | ~13,500 | ~1,100 | ~6,284 (excluded) |
| **Val** | 2,368 | ~1,500 | ~120 | ~748 (excluded) |
| **Test** | 3,130 | ~2,000 | ~220 | ~910 (excluded) |
| **Total** | 26,382 | ~17,000 | ~1,440 | ~7,942 (excluded) |

**After Filtering** (ESD + RAVDESS only):
- **Train**: ~14,600 samples
- **Val**: ~1,620 samples
- **Test**: ~2,220 samples
- **Total**: ~18,440 samples

### Duration Statistics

**Per-Sample Duration**:
- Mean: ~1.5-2.5 seconds
- Min: ~0.5 seconds
- Max: ~5 seconds
- Total: ~12-15 hours (after filtering)

### Emotion Distribution

**ESD Emotions** (10 categories):
1. Neutral
2. Happy
3. Sad
4. Angry
5. Surprise
6. Fear
7. Disgust
8. Contempt
9. Boredom
10. Frustration

**RAVDESS Emotions** (8 categories):
1. Neutral
2. Calm
3. Happy
4. Sad
5. Angry
6. Fearful
7. Disgust
8. Surprised

**Unified Emotion Mapping**:
```
ESD         → RAVDESS     → Used in Training
neutral     → neutral     → <neutral>
happy       → happy       → <happy>
sad         → sad         → <sad>
angry       → angry       → <angry>
surprise    → surprised   → <surprised>
fear        → fearful     → <fearful>
disgust     → disgust     → <disgust>
contempt    → (none)      → <contempt>
boredom     → calm        → <bored> / <calm>
frustration → (none)      → <frustrated>
```

---

## Data Format

### Directory Structure

```
data/full/
├── manifests/              # JSONL manifest files
│   ├── all.jsonl          # All 26,382 samples
│   ├── train.jsonl        # 20,884 training samples
│   ├── val.jsonl          # 2,368 validation samples
│   └── test.jsonl         # 3,130 test samples
├── wav24k/                 # Audio files (24kHz WAV)
│   ├── cremad/            # CREMA-D dataset
│   │   └── {speaker_id}/
│   │       └── {emotion}/
│   │           └── *.wav
│   ├── esd/               # ESD dataset
│   │   └── {speaker_id}/
│   │       └── {emotion}/
│   │           └── *.wav
│   └── ravdess/           # RAVDESS dataset
│       └── {speaker_id}/
│           └── {emotion}/
│               └── *.wav
├── mels/                   # Pre-computed mel-spectrograms (optional)
│   └── {dataset}/
│       └── {speaker_id}/
│           └── {emotion}/
│               └── *.mel.npy
└── reports/                # Dataset statistics and reports
```

### JSONL Manifest Format

Each line in the manifest is a JSON object:

```json
{
  "wav_path": "outputs/full/wav24k/esd/0011/Angry/0011_000351.wav",
  "mel_path": "outputs/full/mels/esd/0011/angry/0011_000351.mel.npy",
  "text": "the nine the eggs, i keep.",
  "chars": ["t","h","e"," ","n","i","n","e"," ","t","h","e"," ","e","g","g","s",",","..."],
  "n_mels": 100,
  "n_frames": 153,
  "duration_s": 1.6213333333333333,
  "speaker_id": "0011",
  "emotion": "angry",
  "dataset": "ESD",
  "lang": "en",
  "speech_rate": 16.036184210526315,
  "split": "train"
}
```

**Field Descriptions**:

| Field | Type | Description |
|-------|------|-------------|
| `wav_path` | string | Relative path to WAV file |
| `mel_path` | string | Relative path to pre-computed mel-spectrogram (optional) |
| `text` | string | Transcription text (lowercase, normalized) |
| `chars` | array | Character-level tokenization |
| `n_mels` | int | Number of mel bins (80 for CosyVoice) |
| `n_frames` | int | Number of mel frames |
| `duration_s` | float | Audio duration in seconds |
| `speaker_id` | string | Speaker identifier (e.g., "0011") |
| `emotion` | string | Emotion label (lowercase) |
| `dataset` | string | Source dataset (ESD/RAVDESS/CREMA-D) |
| `lang` | string | Language code (en/zh) |
| `speech_rate` | float | Characters per second |
| `split` | string | Data split (train/val/test) |

### Kaldi Format (After Preprocessing)

After running `scripts/prepare_data.sh`, data is converted to Kaldi format:

**wav.scp** - Utterance to audio mapping:
```
ESD_0011_angry_0011_000351 /workspace/data/full/wav24k/esd/0011/Angry/0011_000351.wav
RAVDESS_01_angry_01-01-05-01-01-01-01 /workspace/data/full/wav24k/ravdess/01/angry/01-01-05-01-01-01-01.wav
```

**text** - Utterance to text with emotion prefix:
```
ESD_0011_angry_0011_000351 <angry> the nine the eggs, i keep.
RAVDESS_01_angry_01-01-05-01-01-01-01 <angry> dogs are sitting by the door
```

**utt2spk** - Utterance to composite speaker:
```
ESD_0011_angry_0011_000351 0011_angry
RAVDESS_01_angry_01-01-05-01-01-01-01 01_angry
```

**spk2utt** - Speaker to utterances:
```
0011_angry ESD_0011_angry_0011_000351 ESD_0011_angry_0011_000352 ...
01_angry RAVDESS_01_angry_01-01-05-01-01-01-01 RAVDESS_01_angry_01-01-05-01-01-02-01 ...
```

---

## Emotion Categories

### Emotion Taxonomy

**Primary Emotions** (Ekman's 6 basic emotions):
1. **Happy** - Joy, pleasure, contentment
2. **Sad** - Sorrow, grief, melancholy
3. **Angry** - Rage, frustration, irritation
4. **Fear** - Anxiety, terror, apprehension
5. **Disgust** - Revulsion, contempt, aversion
6. **Surprise** - Astonishment, amazement, shock

**Extended Emotions**:
7. **Neutral** - Baseline, no strong emotion
8. **Calm** - Relaxed, peaceful, composed
9. **Contempt** - Scorn, disdain (similar to disgust)
10. **Boredom** - Disinterest, monotony
11. **Frustration** - Annoyance, exasperation (milder than anger)

### Emotion Characteristics in Dataset

**High-Arousal Emotions** (Energetic, Intense):
- Angry, Happy, Fear, Surprise

**Low-Arousal Emotions** (Calm, Subdued):
- Sad, Neutral, Calm, Boredom

**Positive Valence** (Pleasant):
- Happy, Calm

**Negative Valence** (Unpleasant):
- Sad, Angry, Fear, Disgust, Contempt, Frustration, Boredom

**Neutral Valence**:
- Neutral, Surprise

### Prosodic Features by Emotion

| Emotion | Pitch | Energy | Speech Rate | Characteristics |
|---------|-------|--------|-------------|-----------------|
| **Angry** | High | High | Fast | Sharp, loud, tense |
| **Happy** | High | High | Fast | Upbeat, energetic |
| **Sad** | Low | Low | Slow | Monotone, quiet, breathy |
| **Fear** | High | Medium | Fast | Tremulous, breathy |
| **Disgust** | Low | Medium | Slow | Harsh, nasal |
| **Surprise** | High | Medium | Fast | Sudden pitch changes |
| **Neutral** | Medium | Medium | Medium | Baseline, conversational |
| **Calm** | Medium | Low | Slow | Relaxed, smooth |

---

## Preprocessing Pipeline

### Stage 0: Format Conversion

**Input**: JSONL manifests
**Output**: Kaldi format files
**Tool**: `local/prepare_emotional_data.py`

**Transformations**:
1. Filter datasets (keep ESD + RAVDESS, exclude CREMA-D)
2. Generate utterance IDs: `{dataset}_{speaker}_{emotion}_{filename}`
3. Add emotion prefix to text: `<emotion> text`
4. Create composite speaker IDs: `{speaker}_{emotion}`
5. Write Kaldi files: wav.scp, text, utt2spk, spk2utt

**Example**:
```python
# Input (JSONL)
{
  "wav_path": "outputs/full/wav24k/esd/0011/Angry/0011_000351.wav",
  "text": "the nine the eggs, i keep.",
  "speaker_id": "0011",
  "emotion": "angry",
  "dataset": "ESD"
}

# Output (Kaldi text)
ESD_0011_angry_0011_000351 <angry> the nine the eggs, i keep.

# Output (utt2spk)
ESD_0011_angry_0011_000351 0011_angry
```

### Stage 1: Speaker Embeddings

**Input**: wav.scp, utt2spk
**Output**: utt2embedding.pt, spk2embedding.pt
**Tool**: `tools/extract_embedding.py`
**Model**: CAM++ ONNX (campplus.onnx)

**Process**:
1. Load audio and resample to 16kHz
2. Extract 80-dim Fbank features using Kaldi
3. Use CAM++ model to extract 192-dim speaker embeddings
4. Create per-utterance embeddings (utt2embedding.pt)
5. Average per-speaker embeddings (spk2embedding.pt)

**Output Format**:
```python
# utt2embedding.pt
{
  'ESD_0011_angry_0011_000351': tensor([0.123, -0.456, ..., 0.789]),  # 192-dim
  'ESD_0011_happy_0011_000352': tensor([0.234, -0.567, ..., 0.890]),
  ...
}

# spk2embedding.pt
{
  '0011_angry': tensor([0.111, -0.444, ..., 0.777]),  # Averaged from all angry samples
  '0011_happy': tensor([0.222, -0.555, ..., 0.888]),  # Averaged from all happy samples
  ...
}
```

### Stage 2: Speech Tokens

**Input**: wav.scp
**Output**: utt2speech_token.pt
**Tool**: `tools/extract_speech_token.py`
**Model**: speech_tokenizer_v2.onnx

**Process**:
1. Load audio and resample to 16kHz
2. Convert to mono if stereo
3. Extract Whisper-style log mel-spectrogram (128 bins)
4. Use speech tokenizer to quantize to discrete tokens
5. Skip audio longer than 30 seconds

**Output Format**:
```python
# utt2speech_token.pt
{
  'ESD_0011_angry_0011_000351': [1234, 5678, 2345, 6789, ...],  # Variable length
  'ESD_0011_happy_0011_000352': [3456, 7890, 1234, 5678, ...],
  ...
}
```

**Token Properties**:
- Codebook size: 6561 (for CosyVoice 2.0)
- Frame rate: 25 Hz (40ms per token)
- Range: 0-6560

### Stage 3: Parquet Creation

**Input**: All files from stages 0-2
**Output**: Parquet files (*.tar)
**Tool**: `tools/make_parquet_list.py`

**Process**:
1. Read all Kaldi files and PyTorch tensors
2. Package into Parquet format (1000 utterances per file)
3. Create parquet mapping files (utt2parquet, spk2parquet)
4. Generate data.list file

**Parquet Schema**:
```python
{
  'utt': str,              # Utterance ID
  'wav': str,              # Audio file path
  'audio_data': bytes,     # Binary audio data
  'text': str,             # Text with emotion prefix
  'spk': str,              # Composite speaker ID
  'utt_embedding': array,  # 192-dim speaker embedding
  'spk_embedding': array,  # 192-dim averaged speaker embedding
  'speech_token': array,   # Discrete token sequence
}
```

**Output Files**:
```
data/emotional_speech/train/parquet/
├── parquet_000000000.tar  # 1000 utterances
├── parquet_000000001.tar
├── ...
├── parquet_000000015.tar
├── utt2parquet_000000000.json
├── spk2parquet_000000000.json
└── data.list              # List of all parquet files
```

---

## Quality Control

### Data Validation

**Automated Checks**:
1. ✓ All WAV files exist and are readable
2. ✓ Sample rate is 24kHz
3. ✓ Audio duration > 0.5s and < 30s
4. ✓ Text is non-empty
5. ✓ Emotion labels are valid
6. ✓ Speaker IDs are consistent

**Manual Checks** (Recommended):
- Listen to random samples from each emotion
- Verify emotion labels match audio
- Check text transcriptions for accuracy
- Verify speaker diversity

### Known Issues

**ESD Dataset**:
- Some samples have background noise
- Emotion intensity varies by speaker
- Chinese samples not used (English-only fine-tuning)

**RAVDESS Dataset**:
- Professional actors (may be "over-acted")
- Limited speaker diversity (24 speakers)
- Consistent recording quality

**General**:
- Emotion labels are subjective
- Some emotions overlap (angry/frustrated, calm/bored)
- Text content varies (some more emotional than others)

---

## Usage Guidelines

### Training Best Practices

**Data Augmentation** (Optional):
- Speed perturbation: 0.9x, 1.0x, 1.1x
- Pitch shifting: ±1-2 semitones
- Background noise injection
- SpecAugment on mel-spectrograms

**Emotion Balancing**:
- ESD has 10 emotions, RAVDESS has 8
- Some emotions have more samples (neutral, happy)
- Consider weighted sampling or oversampling rare emotions

**Train/Val/Test Split**:
- Train: 79% (for learning)
- Val: 9% (for hyperparameter tuning)
- Test: 12% (for final evaluation)
- Speaker-independent splits (same speaker not in multiple splits)

### Inference Usage

**Emotion Prefix Format**:
```python
# Correct
text = "<angry> I cannot believe you did this!"
text = "<happy> This is absolutely wonderful!"

# Incorrect (will not work)
text = "I cannot believe you did this! [angry]"  # Wrong format
text = "angry: I cannot believe you did this!"   # Wrong format
```

**Supported Emotions**:
```python
emotions = [
    'neutral', 'happy', 'sad', 'angry',
    'surprise', 'fear', 'disgust', 'contempt',
    'calm', 'boredom', 'frustration'
]
```

**Speaker Conditioning**:
- Use reference audio from target speaker
- Emotion in text should match reference audio emotion
- Composite speaker embeddings separate by emotion

---

## Dataset Citation

If you use this dataset, please cite the original sources:

**ESD (Emotional Speech Dataset)**:
```
@inproceedings{zhou2022emotional,
  title={Emotional Voice Conversion: Theory, Databases and ESD},
  author={Zhou, Kun and Sisman, Berrak and Liu, Rui and Li, Haizhou},
  booktitle={Speech Communication},
  year={2022}
}
```

**RAVDESS**:
```
@article{livingstone2018ryerson,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R and Russo, Frank A},
  journal={PLoS ONE},
  year={2018}
}
```

**CREMA-D**:
```
@article{cao2014crema,
  title={CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset},
  author={Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
  journal={IEEE Transactions on Affective Computing},
  year={2014}
}
```

---

## Appendix

### Emotion-to-Color Mapping (for Visualization)

```python
emotion_colors = {
    'neutral': '#808080',    # Gray
    'happy': '#FFD700',      # Gold
    'sad': '#4169E1',        # Royal Blue
    'angry': '#FF0000',      # Red
    'surprise': '#FF69B4',   # Hot Pink
    'fear': '#800080',       # Purple
    'disgust': '#228B22',    # Forest Green
    'contempt': '#8B4513',   # Saddle Brown
    'calm': '#87CEEB',       # Sky Blue
    'boredom': '#A9A9A9',    # Dark Gray
    'frustration': '#FF4500' # Orange Red
}
```

### Sample Data Loading Code

```python
import json
import torchaudio

# Load manifest
with open('data/full/manifests/train.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

# Filter to ESD + RAVDESS
samples_filtered = [s for s in samples if s['dataset'] in ['ESD', 'RAVDESS']]

# Load audio
sample = samples_filtered[0]
waveform, sample_rate = torchaudio.load(f"data/full/{sample['wav_path']}")

print(f"Emotion: {sample['emotion']}")
print(f"Text: {sample['text']}")
print(f"Duration: {sample['duration_s']:.2f}s")
print(f"Sample rate: {sample_rate}Hz")
print(f"Waveform shape: {waveform.shape}")
```

### Emotion Distribution Histogram

```python
import matplotlib.pyplot as plt
from collections import Counter

# Count emotions
emotions = [s['emotion'] for s in samples_filtered]
emotion_counts = Counter(emotions)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.title('Emotion Distribution (ESD + RAVDESS)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('emotion_distribution.png')
```

---

## Change Log

**2025-12-01**:
- Initial dataset documentation
- Documented JSONL and Kaldi formats
- Added preprocessing pipeline details
- Included quality control guidelines

---

## Future Improvements

**Potential Enhancements**:
1. Add more datasets (IEMOCAP, MELD, etc.)
2. Include multi-lingual emotions (Chinese, Japanese, etc.)
3. Add emotion intensity levels (mild/moderate/strong)
4. Include prosody annotations (pitch contours, pauses)
5. Add speaker demographics (age, gender, accent)

**Data Augmentation Ideas**:
- Synthetic emotion transfer using trained model
- Cross-lingual emotion synthesis
- Emotion interpolation (blend emotions)
- Real-world noise augmentation

---

**Document Status**: Complete
**Last Updated**: 2025-12-01
**Maintained By**: Project team
**Contact**: See SYSTEM_CONFIGURATION.md for support info
