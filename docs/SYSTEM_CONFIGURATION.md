# System Configuration Documentation

**Generated**: 2025-12-01
**System**: CosyVoice Emotional Fine-Tuning Setup
**Purpose**: Complete system configuration reference for reproducibility

---

## Hardware Configuration

### GPU
- **Model**: NVIDIA GeForce RTX 5090
- **VRAM**: 32,607 MiB (31.8 GB)
- **Driver Version**: 581.04
- **CUDA Support**: Yes (CUDA 12.1 via PyTorch)
- **Compute Capability**: 9.x (Ada Lovelace architecture)

### Host System
- **OS**: Windows (detected via paths and environment)
- **User**: akshith
- **Working Directory**: `C:\Users\akshith\Desktop\voxe-cosy-finetune\CosyVoice`

### Storage
- **Project Size**: ~5GB (code + configs)
- **Data Size**: 4.4 GB (26,382 WAV files + manifests)
- **Expected Training Output**: ~15-20 GB (checkpoints + logs)
- **Total Space Required**: ~50 GB recommended

---

## Software Stack

### Host Environment

**Docker**:
- Version: 28.5.1 (build e180ab8)
- Runtime: NVIDIA Container Toolkit enabled
- Compose: v3.8 specification

**Python** (Host):
- Version: 3.13.9 (not used for training, training uses Docker)

### Container Environment

**Base Image**: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`

**Python Environment**:
- Version: 3.10.x (via Miniforge conda)
- Environment Name: `cosyvoice`
- Package Manager: conda/mamba + pip

**Key Dependencies** (from requirements.txt):
```
torch==2.3.1 (CUDA 12.1)
torchaudio==2.3.1
transformers==4.51.3
onnxruntime-gpu==1.18.0 (Linux only)
onnxruntime==1.18.0 (Windows/macOS)
deepspeed==0.15.1 (Linux only)
tensorrt-cu12==10.0.1 (Linux only)
diffusers==0.29.0
gradio==5.4.0
fastapi==0.115.6
librosa==0.10.2
modelscope==1.20.0
```

**System Packages**:
- git
- git-lfs
- ffmpeg
- sox
- pynini==2.1.5 (conda-forge)

---

## Docker Configuration

### Container Specification

**Container Name**: `cosyvoice-finetune`

**Image**: `cosyvoice-train:latest`

**Runtime Settings**:
- GPU Runtime: NVIDIA
- GPU Count: 1 (device 0)
- Shared Memory: 16 GB
- Working Directory: `/workspace/CosyVoice`

**Environment Variables**:
```bash
NVIDIA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
PYTHONUNBUFFERED=1
PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
```

**Volume Mounts**:
```yaml
# Main codebase (read-write)
C:/Users/akshith/Desktop/voxe-cosy-finetune/CosyVoice → /workspace/CosyVoice

# Pretrained models (read-only)
C:/Users/akshith/Desktop/voxe-cosy-finetune/CosyVoice/pretrained_models → /workspace/pretrained_models:ro

# Data directory (read-only)
C:/Users/akshith/Desktop/voxe-cosy-finetune/CosyVoice/data → /workspace/data:ro

# Experiment outputs (read-write)
C:/Users/akshith/Desktop/voxe-cosy-finetune/CosyVoice/exp → /workspace/CosyVoice/exp

# TensorBoard logs (read-write)
C:/Users/akshith/Desktop/voxe-cosy-finetune/CosyVoice/tensorboard → /workspace/CosyVoice/tensorboard
```

**GPU Configuration**:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Dataset Configuration

### Dataset Overview

**Source Datasets**:
1. **ESD** (Emotional Speech Dataset)
   - Files: ~17,500
   - Emotions: 10 emotions
   - Languages: English, Chinese

2. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - Files: ~1,440
   - Emotions: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)

3. **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
   - Files: ~7,442
   - Status: **Filtered out for this fine-tuning** (only ESD + RAVDESS used)

### Dataset Statistics

**Total Files**: 26,382 WAV files

**Manifest Breakdown**:
- `all.jsonl`: 26,382 samples (all datasets)
- `train.jsonl`: 20,884 samples (79.2%)
- `val.jsonl`: 2,368 samples (9.0%)
- `test.jsonl`: 3,130 samples (11.8%)

**After Filtering** (ESD + RAVDESS only):
- Train: ~15,080 samples (estimated)
- Val: ~1,700 samples (estimated)
- Test: ~2,200 samples (estimated)

**Audio Format**:
- Sample Rate: 24,000 Hz (24kHz)
- Format: WAV (PCM)
- Channels: Mono (converted if stereo)
- Location: `data/full/wav24k/{dataset}/{speaker_id}/{emotion}/*.wav`

**Manifest Format** (JSONL):
```json
{
  "wav_path": "outputs/full/wav24k/esd/0011/Angry/0011_000351.wav",
  "mel_path": "outputs/full/mels/esd/0011/angry/0011_000351.mel.npy",
  "text": "the nine the eggs, i keep.",
  "chars": ["t","h","e"," ","n","i","n","e",...],
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

**Pre-computed Features**:
- Mel-spectrograms: `data/full/mels/` (optional, will be recomputed during training)
- Reports: `data/full/reports/` (dataset statistics)

---

## Model Configuration

### Pretrained Models

**Base Model**: CosyVoice2-0.5B

**Location**: `pretrained_models/CosyVoice2-0.5B/`

**Model Files**:
```
llm.pt                      # 2.0 GB - Language Model (Qwen2-based)
flow.pt                     # 450 MB - Flow Matching Model
hift.pt                     # 83 MB - HiFiGAN Vocoder (HiFT variant)
campplus.onnx              # 28 MB - Speaker Encoder (CAM++)
speech_tokenizer_v2.onnx   # 496 MB - Speech Tokenizer
CosyVoice-BlankEN/         # Qwen2 tokenizer and config
  ├── config.json
  ├── tokenizer_config.json
  ├── vocab.json
  └── merges.txt
```

### Architecture Specifications

**CosyVoice 2.0 Architecture**:

**1. LLM (Language Model)**:
- Type: Qwen2LM (0.5B parameters)
- Input Size: 896
- Output Size: 896
- Speech Token Size: 6561 (codebook)
- Text Encoder: Qwen2Encoder
- Mix Ratio: [5, 15]
- Sampling: RAS (top_p=0.8, top_k=25)

**2. Flow (Flow Matching)**:
- Type: CausalMaskedDiffWithXvec
- Input Size: 512
- Output Size: 80 (mel-spectrogram bins)
- Speaker Embedding: 192 dimensions
- Token Frame Rate: 25 Hz
- Token-to-Mel Ratio: 2
- Encoder: UpsampleConformerEncoder (6 blocks)
- Decoder: CausalConditionalCFM (causal for streaming)
- Chunk Size: 25 tokens (for streaming inference)

**3. HiFiGAN (Vocoder)**:
- Type: HiFTGenerator (HiFiGAN + F0 predictor)
- Input: 80 mel bins
- Base Channels: 512
- Harmonics: 8
- Upsample Rates: [8, 5, 3] = 120 (24kHz / 200Hz frame rate)
- F0 Predictor: ConvRNNF0Predictor

**Sample Rate**: 24,000 Hz (CosyVoice 2.0 standard)

**Token Frame Rate**: 25 Hz (vs 50 Hz in CosyVoice 1.0)

---

## Training Configuration

### Fine-Tuning Strategy

**Approach**: LLM-only Supervised Fine-Tuning (SFT)

**Rationale**:
- LLM learns emotion-to-speech token mapping (most critical)
- Flow and HiFiGAN are general-purpose (can use pretrained)
- Much faster than full model fine-tuning
- Lower VRAM requirements

**Configuration File**: `conf/cosyvoice2_emotional_sft.yaml`

### Hyperparameters

**Optimizer**:
```yaml
optim: adam
lr: 5e-6  # Lower than pretraining (1e-3) to prevent catastrophic forgetting
```

**Scheduler**:
```yaml
scheduler: constantlr  # No warmup for SFT
warmup_steps: 500      # Minimal warmup
```

**Training**:
```yaml
max_epoch: 30                    # Sufficient for 15K samples
grad_clip: 5                     # Gradient clipping
accum_grad: 2                    # Gradient accumulation (effective batch 2x)
log_interval: 50                 # Frequent logging
save_per_step: -1                # Save per epoch (not per step)
```

**Batch Configuration**:
```yaml
batch_type: 'dynamic'
max_frames_in_batch: 3000        # Optimized for RTX 4090 (24GB)
                                 # Can increase to 4000+ on RTX 5090 (32GB)
```

**Data Processing**:
```yaml
use_spk_embedding: True          # CRITICAL for SFT
shuffle_size: 1000
sort_size: 500
```

**Audio Processing**:
```yaml
sample_rate: 24000
n_fft: 1920
num_mels: 80
hop_size: 480
win_size: 1920
fmin: 0
fmax: 8000
```

**Filtering**:
```yaml
max_length: 40960     # Max audio length in samples
min_length: 100
token_max_length: 200 # Max text tokens
token_min_length: 1
```

### Emotion Conditioning

**Text Format**: `<emotion> transcription text`

**Examples**:
- `<angry> I cannot believe you did this!`
- `<happy> This is absolutely wonderful!`
- `<sad> I miss you so much.`
- `<neutral> The meeting is at three o'clock.`

**Speaker ID Format**: `{speaker_id}_{emotion}` (composite)
- Treats each emotion as a separate speaker
- Better emotion separation during training
- Example: `0011_angry`, `0011_happy`, etc.

**Filtered Datasets**: ESD + RAVDESS only (excludes CREMA-D)

---

## Training Scripts

### Data Preprocessing Pipeline

**Script**: `scripts/prepare_data.sh`

**Stages**:

1. **Stage 0**: JSONL → Kaldi Format (5 min)
   - Input: `data/full/manifests/{train,val,test}.jsonl`
   - Output: `data/emotional_speech/{train,val,test}/{wav.scp,text,utt2spk,spk2utt}`
   - Tool: `local/prepare_emotional_data.py`

2. **Stage 1**: Extract Speaker Embeddings (45-60 min)
   - Input: `wav.scp`, `utt2spk`
   - Output: `utt2embedding.pt`, `spk2embedding.pt`
   - Tool: `tools/extract_embedding.py`
   - Model: CAM++ ONNX (192-dim)

3. **Stage 2**: Extract Speech Tokens (90-120 min)
   - Input: `wav.scp`
   - Output: `utt2speech_token.pt`
   - Tool: `tools/extract_speech_token.py`
   - Model: speech_tokenizer_v2.onnx (6561 codebook)

4. **Stage 3**: Create Parquet Files (45-60 min)
   - Input: All files from stages 0-2
   - Output: `parquet/*.tar`, `data.list`
   - Tool: `tools/make_parquet_list.py`
   - Configuration: 1000 utterances per parquet

**Total Time**: 3-5 hours

### Training Scripts

**1. LLM Training** (`scripts/train_emotional_sft.sh`):
```bash
--model llm
--checkpoint pretrained_models/CosyVoice2-0.5B/llm.pt
--model_dir exp/cosyvoice2_emotional_sft/llm
```

**2. Flow Training** (`scripts/train_emotional_flow.sh`):
```bash
--model flow
--checkpoint pretrained_models/CosyVoice2-0.5B/flow.pt
--model_dir exp/cosyvoice2_emotional_sft/flow
```

**3. HiFiGAN Training** (`scripts/train_emotional_hifigan.sh`):
```bash
--model hifigan
--checkpoint pretrained_models/CosyVoice2-0.5B/hift.pt
--model_dir exp/cosyvoice2_emotional_sft/hifigan
```

**Common Parameters**:
```bash
--train_engine torch_ddp
--qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN
--ddp.dist_backend nccl
--num_workers 4
--prefetch 100
--pin_memory
--use_amp  # FP16 automatic mixed precision
```

---

## Directory Structure

```
CosyVoice/
├── .claude/                      # Claude Code configuration
├── .git/                         # Git repository
├── asset/                        # Demo assets
├── conf/                         # Configuration files
│   └── cosyvoice2_emotional_sft.yaml  # SFT config (CREATED)
├── cosyvoice/                    # Main source code
│   ├── bin/                      # Training scripts
│   ├── cli/                      # Inference API
│   ├── dataset/                  # Data processing
│   ├── flow/                     # Flow matching model
│   ├── hifigan/                  # Vocoder
│   └── llm/                      # Language model
├── data/                         # Dataset
│   └── full/                     # Emotional speech data
│       ├── manifests/            # JSONL manifests
│       ├── wav24k/               # Audio files (24kHz)
│       ├── mels/                 # Pre-computed mels
│       └── reports/              # Statistics
├── docker/                       # Docker configuration
│   ├── Dockerfile.train          # Training Dockerfile (MODIFIED)
│   └── docker-compose.train.yml  # Docker Compose config
├── docs/                         # Documentation (CREATED)
│   └── SYSTEM_CONFIGURATION.md   # This file
├── examples/                     # Training examples
│   └── libritts/
│       └── cosyvoice2/           # CosyVoice 2.0 examples
├── exp/                          # Training outputs (generated)
│   └── cosyvoice2_emotional_sft/
│       ├── llm/                  # LLM checkpoints
│       ├── flow/                 # Flow checkpoints
│       └── hifigan/              # HiFiGAN checkpoints
├── local/                        # Local scripts (CREATED)
│   └── prepare_emotional_data.py # JSONL → Kaldi converter
├── pretrained_models/            # Pretrained models
│   └── CosyVoice2-0.5B/          # Base model (~3GB)
├── runtime/                      # Deployment runtime
├── scripts/                      # Training scripts (CREATED)
│   ├── prepare_data.sh           # Data preprocessing
│   ├── train_emotional_sft.sh    # LLM training
│   ├── train_emotional_flow.sh   # Flow training
│   └── train_emotional_hifigan.sh # HiFiGAN training
├── tensorboard/                  # TensorBoard logs (generated)
├── third_party/                  # Dependencies
│   └── Matcha-TTS/               # Submodule (required)
├── tools/                        # Data processing tools
├── CLAUDE.md                     # Project instructions
├── PARALLEL_TRAINING_GUIDE.md    # Multi-GPU guide (CREATED)
├── QUICK_PARALLEL_SETUP.md       # Quick reference (CREATED)
├── requirements.txt              # Python dependencies
└── webui.py                      # Web interface
```

---

## Network Configuration

### Ports

**TensorBoard**: 6006 (default)
```bash
tensorboard --logdir tensorboard/cosyvoice2_emotional_sft --port 6006
```

**WebUI** (Gradio): 50000 (default)
```bash
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

**FastAPI Server**: 50000 (configurable)
```bash
cd runtime/python/fastapi
python server.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

**gRPC Server**: 50000 (configurable)
```bash
cd runtime/python/grpc
python server.py --port 50000 --max_conc 4
```

### Docker Networking

**Container Network**: Bridge mode (default)
- Host can access container via localhost
- Container can access host network if needed

---

## Performance Benchmarks

### Expected Training Times (RTX 5090)

| Component | Dataset Size | Training Time | VRAM Usage |
|-----------|--------------|---------------|------------|
| **Data Preprocessing** | 15K samples | 3-5 hours | CPU-bound |
| **LLM Training** | 15K samples, 30 epochs | 5-6 hours | 18-22 GB |
| **Flow Training** | 15K samples, 30 epochs | 4-5 hours | 14-18 GB |
| **HiFiGAN Training** | 15K samples, 30 epochs | 6-8 hours | 20-24 GB |

### Comparison: RTX 4090 vs RTX 5090

| Metric | RTX 4090 | RTX 5090 |
|--------|----------|----------|
| **VRAM** | 24 GB | 32 GB |
| **Recommended Batch Size** | 3000 frames | 4000-5000 frames |
| **LLM Training Speed** | ~15 min/epoch | ~10-12 min/epoch (estimated) |
| **Headroom** | Tight (22/24 GB) | Comfortable (22/32 GB) |

---

## Security and Permissions

### Docker Volumes

**Read-Only Mounts**:
- Pretrained models (prevents accidental modification)
- Data directory (source data protection)

**Read-Write Mounts**:
- Main codebase (for development)
- Experiment outputs (for checkpoint saving)
- TensorBoard logs (for monitoring)

### File Permissions

Container runs as `root` by default (standard for Docker).
Files created in mounted volumes inherit host user permissions.

---

## Troubleshooting References

### Common Issues

1. **OOM (Out of Memory)**:
   - Reduce `max_frames_in_batch` in config
   - Increase `accum_grad`
   - Disable AMP (`--use_amp`)

2. **CUDA Not Available**:
   - Check `nvidia-smi` on host
   - Verify NVIDIA Container Toolkit installed
   - Check `runtime: nvidia` in docker-compose.yml

3. **Data Not Found**:
   - Verify volume mounts in docker-compose.yml
   - Check Windows path format (forward slashes: `C:/...`)
   - Ensure data preprocessing completed

4. **Import Errors**:
   - Verify `PYTHONPATH` includes `third_party/Matcha-TTS`
   - Check requirements.txt installed
   - Rebuild Docker image

### Log Locations

**Training Logs**:
- `exp/cosyvoice2_emotional_sft/{llm,flow,hifigan}/train.log`

**TensorBoard Logs**:
- `tensorboard/cosyvoice2_emotional_sft/{llm,flow,hifigan}/`

**Container Logs**:
```bash
docker logs cosyvoice-finetune
```

---

## Backup and Recovery

### Critical Files to Backup

**Configuration**:
- `conf/cosyvoice2_emotional_sft.yaml`
- `docker/Dockerfile.train`
- `docker/docker-compose.train.yml`

**Scripts**:
- `scripts/*.sh`
- `local/prepare_emotional_data.py`

**Trained Models** (after training):
- `exp/cosyvoice2_emotional_sft/*/epoch_*.pt`
- `pretrained_models/CosyVoice2-Emotional-SFT/`

**Data** (preprocessed):
- `data/emotional_speech/*/parquet/`
- `data/emotional_*.data.list`

### Disaster Recovery

**If Docker container crashes**:
```bash
docker-compose -f docker/docker-compose.train.yml down
docker-compose -f docker/docker-compose.train.yml up -d
```

**If training crashes**:
- Checkpoints saved every epoch
- Resume from latest checkpoint
- Check logs for error cause

**If disk full**:
- Remove old checkpoints (keep best 5)
- Clear TensorBoard logs
- Remove intermediate preprocessing files

---

## Version Control

**Git Status**:
- Branch: `main`
- Submodules: `third_party/Matcha-TTS` (initialized)

**Untracked Files** (intentional):
- `.claude/` (local configuration)
- `conf/` (custom configs)
- `docker/` (modified Docker setup)
- `local/` (custom scripts)
- `scripts/` (training scripts)
- `docs/` (documentation)
- `exp/` (training outputs)
- `tensorboard/` (logs)

**Tracked Files** (original repo):
- Core source code (`cosyvoice/`)
- Tools (`tools/`)
- Examples (`examples/`)
- README, LICENSE

---

## Maintenance

### Regular Tasks

**Daily** (during training):
- Monitor GPU temperature (`nvidia-smi`)
- Check disk space (`df -h`)
- Review training logs for errors

**Weekly**:
- Backup important checkpoints
- Clean up old experiments
- Update documentation

**Monthly**:
- Update Docker images
- Update Python packages
- Review and optimize configs

### Updates and Patches

**To update CosyVoice**:
```bash
git pull origin main
git submodule update --remote
docker-compose -f docker/docker-compose.train.yml build --no-cache
```

**To update Python packages**:
```bash
# Edit requirements.txt
docker-compose -f docker/docker-compose.train.yml build --no-cache
```

---

## Contact and Support

**Official Repository**: https://github.com/FunAudioLLM/CosyVoice
**Issues**: https://github.com/FunAudioLLM/CosyVoice/issues
**Documentation**: See `CLAUDE.md` for project-specific guidance

**Local Documentation**:
- `CLAUDE.md` - Project overview and best practices
- `TRAINING_GUIDE.md` - Detailed training instructions
- `PARALLEL_TRAINING_GUIDE.md` - Multi-GPU setup
- `QUICK_PARALLEL_SETUP.md` - Quick reference
- `docs/SYSTEM_CONFIGURATION.md` - This file

---

## Change Log

**2025-12-01**:
- Initial system configuration documentation
- Fixed Dockerfile.train to install requirements.txt
- Created emotional fine-tuning scripts
- Created parallel training guides
- Configured SFT with emotion conditioning

---

## Notes

This configuration is optimized for:
- RTX 5090 GPU (32GB VRAM)
- Windows host with Docker Desktop
- CosyVoice 2.0 (0.5B model)
- Emotional speech fine-tuning
- Single-GPU training (can scale to multi-GPU)

For multi-machine parallel training, see `PARALLEL_TRAINING_GUIDE.md`.
