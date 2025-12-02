# Parallel Training Guide: Running LLM + Flow + HiFiGAN on Multiple RTX 5090 Machines

## Overview

This guide shows you how to train all three CosyVoice2 components (LLM, Flow, HiFiGAN) in parallel across multiple machines with RTX 5090 GPUs.

**Key Advantage**: Since the components are independent, you can train them simultaneously on different machines to get a fully fine-tuned model 3x faster.

## Difficulty Level: EASY

**Time to setup each machine**: 5-10 minutes
**Why it's easy**:
- Same Docker image works on all machines
- Preprocessed data is shared (only process once!)
- Training scripts differ by ONE parameter (`--model` flag)
- No complex networking or distributed training setup needed

## Architecture

```
Machine 1 (Current/RTX 4090)  →  LLM Training      (5-6 hours)  →  llm.pt
Machine 2 (RTX 5090)          →  Flow Training     (4-5 hours)  →  flow.pt
Machine 3 (RTX 5090)          →  HiFiGAN Training  (6-8 hours)  →  hift.pt
                                                                     ↓
                                        Combined Model: CosyVoice2-Emotional-SFT
```

## Prerequisites

### On Each Machine:
- Docker Desktop with NVIDIA Container Toolkit
- RTX 5090 GPU (32GB VRAM recommended)
- 50GB free disk space
- Network access to share preprocessed data

### Shared Resources:
- **Preprocessed data** (run once, share to all machines)
- **Pretrained models** (CosyVoice2-0.5B)
- **Docker setup** (same on all machines)

## Setup Instructions

### Step 1: Prepare Data (Run ONCE on Machine 1)

```bash
# On Machine 1
cd C:\Users\akshith\Desktop\voxe-cosy-finetune\CosyVoice
docker-compose -f docker/docker-compose.train.yml up -d
docker exec -it cosyvoice-finetune bash

# Inside container
bash scripts/prepare_data.sh
```

**Output**: `data/emotional_speech/` directory with parquet files (~20GB)

**Time**: 3-5 hours (only do this once!)

### Step 2: Share Preprocessed Data

**Option A: Network Share (Recommended)**
```bash
# On Machine 1 (Windows)
# Share the data folder via network (\\Machine1\CosyVoice\data)

# On Machine 2 & 3
# Map network drive or copy via rsync/robocopy
robocopy \\Machine1\CosyVoice\data C:\Users\user\CosyVoice\data /E /Z
```

**Option B: Cloud Storage**
```bash
# Upload from Machine 1
aws s3 sync data/ s3://your-bucket/cosyvoice-data/

# Download on Machine 2 & 3
aws s3 sync s3://your-bucket/cosyvoice-data/ data/
```

**Option C: External Drive**
- Copy `data/emotional_speech/` folder to external SSD
- Connect to each machine and copy locally

### Step 3: Setup Each Machine

**On Machine 2 (Flow Training)**:
```bash
cd C:\Users\user\CosyVoice
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Copy your modified files
# - docker/Dockerfile.train (with requirements.txt installation)
# - scripts/train_emotional_flow.sh
# - conf/cosyvoice2_emotional_sft.yaml
# - local/prepare_emotional_data.py

# Copy or sync preprocessed data
# data/emotional_speech/ → from Machine 1

# Download pretrained models
# pretrained_models/CosyVoice2-0.5B/ → download or copy from Machine 1

# Build Docker
docker-compose -f docker/docker-compose.train.yml build

# Start container
docker-compose -f docker/docker-compose.train.yml up -d
docker exec -it cosyvoice-finetune bash
```

**On Machine 3 (HiFiGAN Training)**: Same as Machine 2

### Step 4: Start Training on All Machines

**Machine 1 (LLM)**:
```bash
docker exec -it cosyvoice-finetune bash
bash scripts/train_emotional_sft.sh
```

**Machine 2 (Flow)**:
```bash
docker exec -it cosyvoice-finetune bash
bash scripts/train_emotional_flow.sh
```

**Machine 3 (HiFiGAN)**:
```bash
docker exec -it cosyvoice-finetune bash
bash scripts/train_emotional_hifigan.sh
```

### Step 5: Monitor Training

**On each machine**, open another terminal:

```bash
# Watch training logs
docker exec -it cosyvoice-finetune bash
tail -f exp/cosyvoice2_emotional_sft/{llm,flow,hifigan}/train.log

# Or use TensorBoard
tensorboard --logdir tensorboard/cosyvoice2_emotional_sft --port 6006
```

Open browser: http://localhost:6006

### Step 6: Collect Trained Models

**After training completes on each machine:**

**From Machine 1** (LLM):
```bash
# Average best checkpoints
python cosyvoice/bin/average_model.py \
  --dst_model exp/cosyvoice2_emotional_sft/llm/llm.pt \
  --src_path exp/cosyvoice2_emotional_sft/llm \
  --num 5 --val_best

# Copy to shared location
cp exp/cosyvoice2_emotional_sft/llm/llm.pt /shared/llm.pt
```

**From Machine 2** (Flow):
```bash
python cosyvoice/bin/average_model.py \
  --dst_model exp/cosyvoice2_emotional_sft/flow/flow.pt \
  --src_path exp/cosyvoice2_emotional_sft/flow \
  --num 5 --val_best

cp exp/cosyvoice2_emotional_sft/flow/flow.pt /shared/flow.pt
```

**From Machine 3** (HiFiGAN):
```bash
python cosyvoice/bin/average_model.py \
  --dst_model exp/cosyvoice2_emotional_sft/hifigan/hift.pt \
  --src_path exp/cosyvoice2_emotional_sft/hifigan \
  --num 5 --val_best

cp exp/cosyvoice2_emotional_sft/hifigan/hift.pt /shared/hift.pt
```

### Step 7: Combine into Final Model

**On any machine** (or Machine 1):
```bash
mkdir -p pretrained_models/CosyVoice2-Emotional-SFT

# Copy fine-tuned models
cp /shared/llm.pt pretrained_models/CosyVoice2-Emotional-SFT/
cp /shared/flow.pt pretrained_models/CosyVoice2-Emotional-SFT/
cp /shared/hift.pt pretrained_models/CosyVoice2-Emotional-SFT/

# Copy supporting files (unchanged)
cp pretrained_models/CosyVoice2-0.5B/*.onnx pretrained_models/CosyVoice2-Emotional-SFT/
cp pretrained_models/CosyVoice2-0.5B/cosyvoice2.yaml pretrained_models/CosyVoice2-Emotional-SFT/
cp -r pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN pretrained_models/CosyVoice2-Emotional-SFT/
```

### Step 8: Test Combined Model

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# Load fully fine-tuned model
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-Emotional-SFT')

# Test emotional synthesis
prompt_speech = load_wav('data/full/wav24k/esd/0011/Angry/0011_000351.wav', 16000)

for emotion in ['angry', 'happy', 'sad', 'neutral']:
    text = f'<{emotion}> This is a test of the emotional voice synthesis.'
    for i, j in enumerate(cosyvoice.inference_sft(text, stream=False)):
        torchaudio.save(f'test_{emotion}.wav', j['tts_speech'], 24000)
    print(f'Generated {emotion} audio')
```

## Training Time Comparison

### Sequential Training (Single Machine):
- LLM: 5-6 hours
- Flow: 4-5 hours
- HiFiGAN: 6-8 hours
- **Total: 15-19 hours**

### Parallel Training (3 Machines):
- All components: max(5-6, 4-5, 6-8) hours
- **Total: 6-8 hours** (3x faster!)

## Resource Requirements Per Machine

| Component | VRAM Usage | RAM | Disk | Training Time (RTX 5090) |
|-----------|-----------|-----|------|-------------------------|
| LLM | 18-22GB | 32GB | 20GB | 5-6 hours |
| Flow | 14-18GB | 32GB | 20GB | 4-5 hours |
| HiFiGAN | 20-24GB | 32GB | 20GB | 6-8 hours |

## Key Configuration Differences

### All three scripts are nearly identical, differing ONLY in:

**train_emotional_sft.sh** (LLM):
```bash
--model llm \
--checkpoint $pretrained_model_dir/llm.pt \
--model_dir exp/cosyvoice2_emotional_sft/llm
```

**train_emotional_flow.sh** (Flow):
```bash
--model flow \
--checkpoint $pretrained_model_dir/flow.pt \
--model_dir exp/cosyvoice2_emotional_sft/flow
```

**train_emotional_hifigan.sh** (HiFiGAN):
```bash
--model hifigan \
--checkpoint $pretrained_model_dir/hift.pt \
--model_dir exp/cosyvoice2_emotional_sft/hifigan
```

## Important Notes

### HiFiGAN Special Requirements

HiFiGAN uses GAN training, which has specific constraints:

**In config file** (conf/cosyvoice2_emotional_sft.yaml):
```yaml
train_conf_gan:
  accum_grad: 1  # MUST be 1 for GAN training (no gradient accumulation)
```

The training script automatically uses `train_conf_gan` when `--model hifigan` is specified.

### Data Sharing Best Practices

1. **Share preprocessed data**, not raw JSONL (saves 3-5 hours per machine)
2. **Verify checksums** after copying to ensure data integrity
3. **Use fast network/SSD** for transfers (20GB of parquet files)

### Monitoring All Machines

**Use tmux/screen** to keep training running:
```bash
# On each machine
tmux new -s training
bash scripts/train_emotional_{llm,flow,hifigan}.sh
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

**Centralized monitoring** (optional):
- Set up shared TensorBoard server
- Aggregate logs from all machines
- Use monitoring tools (Prometheus/Grafana)

## Troubleshooting

### Issue: OOM on RTX 5090 (32GB VRAM)

**Unlikely** - RTX 5090 has more VRAM than RTX 4090. If it happens:

```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
max_frames_in_batch: 2500  # Reduce from 3000
accum_grad: 3              # Increase from 2
```

### Issue: Different training speeds

**Expected** - Each component trains at different speeds:
- Flow: Fastest (4-5 hours)
- LLM: Medium (5-6 hours)
- HiFiGAN: Slowest (6-8 hours) - GAN training is complex

### Issue: Data sync problems

**Verify data integrity**:
```bash
# On Machine 1 (source)
find data/emotional_speech -name "*.tar" | sort | md5sum > checksums.txt

# On Machine 2 & 3 (destinations)
md5sum -c checksums.txt
```

### Issue: Training diverges on one component

**Possible causes**:
- Bad data copy (verify checksums)
- Different config file versions (ensure all machines use same config)
- GPU issues (check `nvidia-smi` for errors)

**Solution**: Restart training with lower learning rate or different checkpoint.

## Advanced: Running All on Single RTX 5090

**If you have only ONE RTX 5090**, you can still train all components faster:

**Option A: Sequential (Standard)**
- Train LLM → Flow → HiFiGAN sequentially
- Time: 15-19 hours

**Option B: Sequential with Prioritization**
- Train LLM first (most important for emotion conditioning)
- Test with pretrained Flow/HiFiGAN
- Then fine-tune Flow and HiFiGAN if needed

**Option C: Multi-GPU Single Machine**
- If you have multiple GPUs, train components in parallel:
```bash
# Terminal 1: LLM on GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/train_emotional_sft.sh

# Terminal 2: Flow on GPU 1
CUDA_VISIBLE_DEVICES=1 bash scripts/train_emotional_flow.sh

# Terminal 3: HiFiGAN on GPU 2
CUDA_VISIBLE_DEVICES=2 bash scripts/train_emotional_hifigan.sh
```

## Files Created

I've created these new training scripts for you:

1. **scripts/train_emotional_flow.sh** - Flow matching model training
2. **scripts/train_emotional_hifigan.sh** - HiFiGAN vocoder training
3. **PARALLEL_TRAINING_GUIDE.md** - This guide

All scripts follow the same pattern as `train_emotional_sft.sh`, differing only in the `--model` parameter.

## Quick Start Checklist

### Machine 1 (LLM + Data Prep):
- [ ] Build Docker image with fixed Dockerfile
- [ ] Run `scripts/prepare_data.sh` (3-5 hours, ONCE)
- [ ] Share `data/emotional_speech/` to other machines
- [ ] Run `scripts/train_emotional_sft.sh`

### Machine 2 (Flow):
- [ ] Copy/clone CosyVoice repository
- [ ] Copy modified Dockerfile and scripts
- [ ] Sync preprocessed data from Machine 1
- [ ] Download or copy pretrained models
- [ ] Build Docker image
- [ ] Run `scripts/train_emotional_flow.sh`

### Machine 3 (HiFiGAN):
- [ ] Same as Machine 2
- [ ] Run `scripts/train_emotional_hifigan.sh`

### After All Training Completes:
- [ ] Average checkpoints on each machine
- [ ] Collect all three model files (llm.pt, flow.pt, hift.pt)
- [ ] Combine into CosyVoice2-Emotional-SFT directory
- [ ] Test emotional inference

## Summary

**Difficulty**: ⭐⭐☆☆☆ (Easy)
**Time Savings**: 3x faster with parallel training
**Complexity**: Minimal - same setup, different `--model` flag
**Recommended**: Yes, if you have multiple RTX 5090 machines available

The hardest part is the initial data preprocessing (3-5 hours), but you only do it ONCE and share the results. After that, training is fully independent and parallelizable.
