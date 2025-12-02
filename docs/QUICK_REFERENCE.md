# Quick Reference Guide

**Last Updated**: 2025-12-01
**Purpose**: Fast lookup for common commands and configurations

---

## Table of Contents

- [System Info](#system-info)
- [Docker Commands](#docker-commands)
- [Training Commands](#training-commands)
- [Monitoring](#monitoring)
- [Common Paths](#common-paths)
- [Configuration Quick Edits](#configuration-quick-edits)
- [Troubleshooting Commands](#troubleshooting-commands)

---

## System Info

### Hardware
```bash
# GPU info
nvidia-smi

# Detailed GPU
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu --format=csv

# Continuous monitoring
watch -n 1 nvidia-smi
```

### Disk Space
```bash
# Check free space
df -h

# Check directory sizes
du -sh data/
du -sh pretrained_models/
du -sh exp/
```

---

## Docker Commands

### Build & Start
```bash
# Build image
docker-compose -f docker/docker-compose.train.yml build

# Start container
docker-compose -f docker/docker-compose.train.yml up -d

# Access container
docker exec -it cosyvoice-finetune bash

# Stop container
docker-compose -f docker/docker-compose.train.yml down
```

### Container Management
```bash
# View logs
docker logs cosyvoice-finetune

# Check container status
docker ps

# Restart container
docker restart cosyvoice-finetune

# Remove container and image
docker-compose -f docker/docker-compose.train.yml down
docker rmi cosyvoice-train:latest
```

---

## Training Commands

### Test First (Recommended!)
```bash
# 1. Test preprocessing with small subset (~5-10 minutes)
bash scripts/test_preprocessing.sh

# 2. Test training with small subset (~2-3 minutes)
bash scripts/test_training.sh

# If tests pass, proceed with full pipeline
```

### Full Pipeline (Inside Container)
```bash
# 1. Data preprocessing (3-5 hours)
bash scripts/prepare_data.sh

# 2. LLM training (5-6 hours)
bash scripts/train_emotional_sft.sh

# 3. Flow training (4-5 hours) - optional
bash scripts/train_emotional_flow.sh

# 4. HiFiGAN training (6-8 hours) - optional
bash scripts/train_emotional_hifigan.sh
```

### Manual Training Command
```bash
# LLM training (manual)
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice2_emotional_sft.yaml \
  --train_data data/emotional_train.data.list \
  --cv_data data/emotional_val.data.list \
  --qwen_pretrain_path /workspace/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --model llm \
  --checkpoint /workspace/pretrained_models/CosyVoice2-0.5B/llm.pt \
  --model_dir exp/cosyvoice2_emotional_sft/llm \
  --tensorboard_dir tensorboard/cosyvoice2_emotional_sft/llm \
  --num_workers 4 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
```

### Model Averaging
```bash
# Average best 5 checkpoints
python cosyvoice/bin/average_model.py \
  --dst_model exp/cosyvoice2_emotional_sft/llm/llm.pt \
  --src_path exp/cosyvoice2_emotional_sft/llm \
  --num 5 \
  --val_best
```

---

## Monitoring

### Training Logs
```bash
# Watch training log (inside container)
tail -f exp/cosyvoice2_emotional_sft/llm/train.log

# Check last 50 lines
tail -50 exp/cosyvoice2_emotional_sft/llm/train.log

# Search for errors
grep -i "error" exp/cosyvoice2_emotional_sft/llm/train.log
grep -i "nan" exp/cosyvoice2_emotional_sft/llm/train.log
```

### TensorBoard
```bash
# On host machine
tensorboard --logdir tensorboard/cosyvoice2_emotional_sft/llm --port 6006

# Open in browser
http://localhost:6006

# All components
tensorboard --logdir tensorboard/cosyvoice2_emotional_sft --port 6006
```

### GPU Monitoring
```bash
# Inside container
watch -n 1 nvidia-smi

# Check specific metrics
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv -l 1
```

---

## Common Paths

### Host (Windows)
```
C:\Users\akshith\Desktop\voxe-cosy-finetune\CosyVoice\
├── data\                          # Dataset
├── pretrained_models\             # Pretrained models
├── exp\                           # Training outputs
├── tensorboard\                   # TensorBoard logs
├── conf\                          # Configs
├── scripts\                       # Training scripts
└── docker\                        # Docker files
```

### Container (Linux)
```
/workspace/CosyVoice/
├── data/                          # Dataset (read-only)
├── pretrained_models/             # Pretrained models (read-only)
├── exp/                           # Training outputs (read-write)
├── tensorboard/                   # TensorBoard logs (read-write)
├── conf/                          # Configs
├── scripts/                       # Training scripts
└── docker/                        # Docker files
```

### Key Files
```bash
# Config
conf/cosyvoice2_emotional_sft.yaml

# Data preparation
scripts/prepare_data.sh
local/prepare_emotional_data.py

# Training
scripts/train_emotional_sft.sh
scripts/train_emotional_flow.sh
scripts/train_emotional_hifigan.sh

# Docker
docker/Dockerfile.train
docker/docker-compose.train.yml

# Documentation
docs/SYSTEM_CONFIGURATION.md
docs/DATASET_DOCUMENTATION.md
docs/QUICK_REFERENCE.md
PARALLEL_TRAINING_GUIDE.md
```

---

## Configuration Quick Edits

### Reduce Batch Size (If OOM)
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
batch:
  max_frames_in_batch: 2000  # Reduce from 3000
```

### Increase Batch Size (For RTX 5090)
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
batch:
  max_frames_in_batch: 4000  # Increase from 3000
```

### Reduce Learning Rate
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
train_conf:
  optim_conf:
    lr: 1e-6  # Reduce from 5e-6
```

### Increase Gradient Accumulation
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
train_conf:
  accum_grad: 4  # Increase from 2
```

### Change Number of Epochs
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
train_conf:
  max_epoch: 50  # Increase from 30
```

---

## Troubleshooting Commands

### Verify Environment
```bash
# Inside container
python --version                    # Should be 3.10.x
echo $PYTHONPATH                    # Should include Matcha-TTS
which python                        # Should be conda env
conda env list                      # Should show cosyvoice
```

### Check Python Packages
```bash
# Inside container
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import cosyvoice; print('OK')"
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

### Verify Data
```bash
# Check manifest counts
wc -l data/full/manifests/*.jsonl

# Check processed data
wc -l data/emotional_train.data.list
wc -l data/emotional_val.data.list

# Verify parquet files
ls -lh data/emotional_speech/train/parquet/*.tar | wc -l

# Check Kaldi files
wc -l data/emotional_speech/train/{wav.scp,text,utt2spk,spk2utt}

# Verify emotion prefixes
head -3 data/emotional_speech/train/text
grep "<angry>" data/emotional_speech/train/text | head -3
```

### Check Pretrained Models
```bash
# Verify files exist
ls -lh pretrained_models/CosyVoice2-0.5B/

# Check file sizes
du -sh pretrained_models/CosyVoice2-0.5B/*.pt

# Verify ONNX models
ls -lh pretrained_models/CosyVoice2-0.5B/*.onnx
```

### Check Training Progress
```bash
# Count checkpoints
ls exp/cosyvoice2_emotional_sft/llm/epoch_*.pt | wc -l

# Check latest checkpoint
ls -lht exp/cosyvoice2_emotional_sft/llm/epoch_*.pt | head -5

# Verify training log exists
ls -lh exp/cosyvoice2_emotional_sft/llm/train.log

# Check loss values
grep "Epoch" exp/cosyvoice2_emotional_sft/llm/train.log | tail -20
```

### Disk Space Issues
```bash
# Find large files
find exp/ -type f -size +1G

# Remove old checkpoints (keep best 5)
# CAREFUL: This deletes files!
cd exp/cosyvoice2_emotional_sft/llm/
ls epoch_*.pt | sort | head -n -5 | xargs rm -f

# Clean up parquet files (can regenerate)
rm -rf data/emotional_speech/*/parquet/*.tar
```

### Network Issues
```bash
# Test internet connection
ping -c 3 google.com

# Test Docker networking
docker exec cosyvoice-finetune ping -c 3 google.com
```

---

## Inference Testing

### Quick Test (Inside Container)
```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# Load model
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

# Load reference audio
prompt_speech = load_wav('data/full/wav24k/esd/0011/Angry/0011_000351.wav', 16000)

# Generate with emotion
text = '<angry> I cannot believe you did this!'
for i, j in enumerate(cosyvoice.inference_sft(text, stream=False)):
    torchaudio.save('test_angry.wav', j['tts_speech'], 24000)
print('Generated: test_angry.wav')
```

### Test Fine-Tuned Model
```python
# Same as above, but change model path
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-Emotional-SFT')
```

---

## Performance Benchmarks

### Expected Times (RTX 5090)

| Task | Time | Notes |
|------|------|-------|
| Docker build | 15-30 min | First time only |
| Data preprocessing | 3-5 hours | Run once, reuse |
| LLM training (30 epochs) | 5-6 hours | ~10-12 min/epoch |
| Flow training (30 epochs) | 4-5 hours | ~8-10 min/epoch |
| HiFiGAN training (30 epochs) | 6-8 hours | ~12-16 min/epoch |
| Model averaging | 1-2 min | Combine checkpoints |
| Inference (1 sentence) | 2-5 sec | Real-time factor ~0.1 |

### VRAM Usage

| Component | VRAM Used | Config |
|-----------|-----------|--------|
| LLM | 18-22 GB | batch_size=3000 |
| Flow | 14-18 GB | batch_size=3000 |
| HiFiGAN | 20-24 GB | batch_size=3000 |
| LLM (RTX 5090) | 22-26 GB | batch_size=4000 |

---

## Useful One-Liners

```bash
# Quick status check
docker ps && nvidia-smi && df -h | grep -E '/$|CosyVoice'

# Check if training is running
ps aux | grep train.py

# Kill stuck training
pkill -9 -f train.py

# Count total samples
cat data/emotional_train.data.list | wc -l

# Find latest checkpoint
ls -t exp/cosyvoice2_emotional_sft/llm/epoch_*.pt | head -1

# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# Estimate training time remaining
# (Total epochs - Current epoch) * Avg time per epoch

# Check if loss is decreasing
grep "loss=" exp/cosyvoice2_emotional_sft/llm/train.log | tail -20

# Verify emotion prefixes in data
cut -d' ' -f2 data/emotional_speech/train/text | sort | uniq -c
```

---

## Git Commands (Version Control)

```bash
# Check status
git status

# Add custom files
git add conf/ scripts/ local/ docs/

# Commit changes
git commit -m "Add emotional fine-tuning configuration"

# Create branch for experiments
git checkout -b experiment-lr-1e6

# View changes
git diff

# Ignore large files (already in .gitignore)
# - exp/
# - tensorboard/
# - data/
# - pretrained_models/
```

---

## Emergency Commands

### Training Crashed - Resume
```bash
# Find latest checkpoint
ls -t exp/cosyvoice2_emotional_sft/llm/epoch_*.pt | head -1

# Edit training script to use latest checkpoint
# Change --checkpoint to point to latest epoch

# Restart training
bash scripts/train_emotional_sft.sh
```

### Out of Disk Space
```bash
# Find space hogs
du -sh exp/* | sort -h
du -sh data/* | sort -h

# Delete old experiments
rm -rf exp/old_experiment_*/

# Delete preprocessed intermediate files (can regenerate)
rm data/emotional_speech/*/utt2embedding.pt
rm data/emotional_speech/*/utt2speech_token.pt
```

### Docker Issues
```bash
# Restart Docker Desktop
# (GUI: Right-click → Restart)

# Or via command
docker restart cosyvoice-finetune

# Nuclear option: rebuild everything
docker-compose -f docker/docker-compose.train.yml down
docker system prune -a
docker-compose -f docker/docker-compose.train.yml build --no-cache
```

---

## Shortcuts

### Aliases (Add to ~/.bashrc inside container)
```bash
alias ll='ls -lh'
alias lt='ls -lht'
alias tlog='tail -f exp/cosyvoice2_emotional_sft/llm/train.log'
alias gpu='watch -n 1 nvidia-smi'
alias trainllm='bash scripts/train_emotional_sft.sh'
```

### Environment Variables
```bash
# Add to ~/.bashrc for convenience
export EXP_DIR=/workspace/CosyVoice/exp/cosyvoice2_emotional_sft
export MODEL_DIR=/workspace/pretrained_models/CosyVoice2-0.5B
export DATA_DIR=/workspace/data

# Then use
cd $EXP_DIR/llm
ls $MODEL_DIR
```

---

## Documentation Cross-Reference

- **Full system details**: `docs/SYSTEM_CONFIGURATION.md`
- **Dataset info**: `docs/DATASET_DOCUMENTATION.md`
- **Multi-GPU training**: `PARALLEL_TRAINING_GUIDE.md`
- **Quick parallel setup**: `QUICK_PARALLEL_SETUP.md`
- **Project guidelines**: `CLAUDE.md`
- **This guide**: `docs/QUICK_REFERENCE.md`

---

**Tip**: Bookmark this page for quick lookups during training!
