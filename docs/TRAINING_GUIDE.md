# CosyVoice2 Emotional Speech Fine-Tuning Guide

## Quick Start: Complete Training Pipeline

This guide provides step-by-step instructions for fine-tuning CosyVoice2-0.5B on emotional speech (ESD + RAVDESS datasets).

**Hardware:** RTX 5090 (32GB VRAM)
**Dataset:** 15,080 samples (ESD: 14,000, RAVDESS: 1,080)
**Emotions:** angry, happy, neutral, sad, surprise
**Training Time:** ~10-12 hours for 30 epochs

---

## Phase 1: Environment Setup (1-2 hours)

### Step 1.1: Download Pretrained Model

**IMPORTANT:** Download the model BEFORE building Docker.

```powershell
# On Windows host (PowerShell)
cd C:\Users\akshith\Desktop\voxe-cosy-finetune\CosyVoice

# Create directory
mkdir pretrained_models

# Download using Python
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
```

**Verify download:**
```powershell
dir pretrained_models\CosyVoice2-0.5B
```

You should see:
- `llm.pt` (fine-tuning checkpoint)
- `flow.pt`, `hift.pt` (frozen models)
- `campplus.onnx` (speaker embeddings)
- `speech_tokenizer_v2.onnx` (speech tokens)
- `CosyVoice-BlankEN/` (tokenizer)

### Step 1.2: Build and Start Docker Container

```powershell
# Build Docker image (takes ~15-20 minutes)
cd docker
docker-compose -f docker-compose.train.yml build

# Start container
docker-compose -f docker-compose.train.yml up -d

# Verify GPU access
docker exec -it cosyvoice-finetune nvidia-smi
```

**Expected output:** Should show RTX 5090 with 32GB VRAM

### Step 1.3: Install Python Dependencies

```powershell
# Enter container
docker exec -it cosyvoice-finetune /bin/bash
```

Inside container:
```bash
# Activate conda environment
conda activate cosyvoice

# Install dependencies (takes ~10-15 minutes)
pip install -r /workspace/CosyVoice/requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Phase 2: Data Preparation (2-3 hours)

All commands below are run **inside the Docker container**.

### Step 2.1: Run Complete Data Pipeline

```bash
cd /workspace/CosyVoice

# Make scripts executable
chmod +x scripts/*.sh

# Run data preparation (automated)
bash scripts/prepare_data.sh
```

This script will:
1. Convert JSONL to Kaldi format (filters ESD + RAVDESS only)
2. Extract speaker embeddings using CAM++ (15-20 min)
3. Extract speech tokens (30-45 min)
4. Create parquet files for training (20-30 min)

**Expected output:**
```
Train data: 16 parquet files
Val data: 3 parquet files
```

**Verify data:**
```bash
# Check train/val splits
ls -lh data/emotional_speech/train/parquet/*.tar
ls -lh data/emotional_speech/val/parquet/*.tar

# Check data lists
cat data/emotional_train.data.list
cat data/emotional_val.data.list
```

---

## Phase 3: Training (10-12 hours)

### Step 3.1: Launch Training

```bash
cd /workspace/CosyVoice

# Start training
bash scripts/train_emotional_sft.sh
```

**Training parameters:**
- Batch size: ~6-8 samples (dynamic batching)
- Gradient accumulation: 2 (effective batch: 12-16)
- Learning rate: 5e-6
- Epochs: 30
- GPU memory: ~28GB / 32GB

### Step 3.2: Monitor Training

**Option A: Watch logs**
```bash
# In another terminal
docker exec -it cosyvoice-finetune /bin/bash
cd /workspace/CosyVoice
tail -f exp/cosyvoice2_emotional_sft/llm/train.log
```

**Option B: TensorBoard** (recommended)
```bash
# In another terminal inside container
conda activate cosyvoice
tensorboard --logdir tensorboard/cosyvoice2_emotional_sft --host 0.0.0.0 --port 6006
```

Access in browser: `http://localhost:6006`

**Metrics to watch:**
- `train_loss`: Should decrease from ~6.0 to ~2.5-3.0
- `train_acc`: Should increase from ~0.2 to ~0.65-0.70
- `cv_loss`: Validation loss (watch for overfitting if it diverges)

**Expected training log:**
```
Epoch 0/30, Step 100/29400, Loss: 5.234, Acc: 0.245
Epoch 0/30, Step 200/29400, Loss: 4.891, Acc: 0.289
...
Epoch 5/30, CV Loss: 3.456, CV Acc: 0.512
Saving checkpoint: epoch-5.pt
...
Epoch 29/30, CV Loss: 2.678, CV Acc: 0.687
Training completed!
```

**Checkpoints saved to:** `exp/cosyvoice2_emotional_sft/llm/`
- `epoch-0.pt` through `epoch-29.pt`
- `latest.pt` (symlink to most recent)

---

## Phase 4: Model Export (30 minutes)

### Step 4.1: Average Best Checkpoints

```bash
cd /workspace/CosyVoice

# Average top 5 checkpoints by validation loss
bash scripts/average_checkpoints.sh
```

This creates `exp/cosyvoice2_emotional_sft/llm/llm.pt` (averaged model).

### Step 4.2: Assemble Final Model

```bash
# Combine fine-tuned LLM with pretrained Flow/HiFT
bash scripts/assemble_model.sh
```

This creates `exp/cosyvoice2_emotional_final/` containing:
- `llm.pt` (fine-tuned)
- `flow.pt`, `hift.pt` (pretrained)
- `CosyVoice-BlankEN/` (tokenizer)
- `*.onnx` (embeddings/tokens)

---

## Phase 5: Testing (30 minutes)

### Step 5.1: Run Inference Test

```bash
cd /workspace/CosyVoice

# Generate test samples for all 5 emotions
python scripts/test_emotional_inference.py
```

**Output:** `exp/test_outputs/`
- `angry_output.wav`
- `happy_output.wav`
- `neutral_output.wav`
- `sad_output.wav`
- `surprise_output.wav`

### Step 5.2: Listen to Results

Copy audio files to Windows:
```powershell
# On Windows host
docker cp cosyvoice-finetune:/workspace/CosyVoice/exp/test_outputs C:\Users\akshith\Desktop\
```

Listen to each file and verify emotional variation.

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Edit `conf/cosyvoice2_emotional_sft.yaml`:
   ```yaml
   max_frames_in_batch: 2000  # Reduce from 3000
   ```
2. Or reduce gradient accumulation:
   ```yaml
   accum_grad: 1  # Reduce from 2
   ```
3. Restart training:
   ```bash
   bash scripts/train_emotional_sft.sh
   ```

### Issue: Training Loss Not Decreasing

**Symptoms:** Loss stuck at ~6.0 after 5+ epochs

**Solutions:**
1. Check data quality:
   ```bash
   # Verify parquet files exist
   ls -lh data/emotional_speech/train/parquet/
   ```
2. Lower learning rate in `conf/cosyvoice2_emotional_sft.yaml`:
   ```yaml
   lr: 2e-6  # Reduce from 5e-6
   ```
3. Check if pretrained model loaded correctly:
   ```bash
   grep "loading checkpoint" exp/cosyvoice2_emotional_sft/llm/train.log
   ```

### Issue: Poor Emotion Transfer

**Symptoms:** Generated audio doesn't match target emotion

**Solutions:**
1. Verify emotion prefixes in training data:
   ```bash
   # Check text file
   head data/emotional_speech/train/text
   # Should show: <angry> text, <happy> text, etc.
   ```
2. Increase training epochs to 50:
   ```yaml
   max_epoch: 50  # In conf/cosyvoice2_emotional_sft.yaml
   ```
3. Use composite speaker IDs (should already be enabled):
   ```python
   # In local/prepare_emotional_data.py
   --composite_speaker  # Should be True
   ```

### Issue: Pretrained Model Not Found

**Symptoms:** `FileNotFoundError: llm.pt`

**Solution:**
```bash
# Re-download model
cd /workspace
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

# Verify files
ls -lh pretrained_models/CosyVoice2-0.5B/
```

### Issue: Docker Volume Mount Failed

**Symptoms:** `No such file or directory` in container

**Solutions:**
1. Use forward slashes in `docker-compose.train.yml`:
   ```yaml
   volumes:
     - C:/Users/akshith/Desktop/voxe-cosy-finetune/CosyVoice:/workspace/CosyVoice
   ```
2. Enable file sharing in Docker Desktop:
   - Settings → Resources → File Sharing
   - Add `C:\Users\akshith\Desktop`
3. Restart Docker Desktop and container

---

## Advanced: Resume Training

If training is interrupted:

```bash
cd /workspace/CosyVoice

# Find latest checkpoint
ls -lh exp/cosyvoice2_emotional_sft/llm/epoch-*.pt

# Edit train script to resume from checkpoint
# Modify scripts/train_emotional_sft.sh:
# Change: --checkpoint $pretrained_model_dir/llm.pt
# To: --checkpoint exp/cosyvoice2_emotional_sft/llm/epoch-10.pt  # Use your latest

# Resume training
bash scripts/train_emotional_sft.sh
```

---

## Complete File Structure

After completing all steps:

```
CosyVoice/
├── conf/
│   └── cosyvoice2_emotional_sft.yaml      # Training config
├── data/
│   ├── emotional_speech/
│   │   ├── train/                          # Kaldi format + embeddings
│   │   ├── val/
│   │   └── test/
│   ├── emotional_train.data.list           # Parquet file list
│   └── emotional_val.data.list
├── docker/
│   ├── Dockerfile.train                    # Training container
│   └── docker-compose.train.yml
├── exp/
│   ├── cosyvoice2_emotional_sft/llm/      # Training checkpoints
│   ├── cosyvoice2_emotional_final/         # Final assembled model
│   └── test_outputs/                       # Generated audio samples
├── local/
│   └── prepare_emotional_data.py           # Data conversion script
├── pretrained_models/
│   └── CosyVoice2-0.5B/                   # Downloaded pretrained model
├── scripts/
│   ├── prepare_data.sh                     # Data pipeline
│   ├── train_emotional_sft.sh             # Training launcher
│   ├── average_checkpoints.sh             # Model averaging
│   ├── assemble_model.sh                  # Final model assembly
│   └── test_emotional_inference.py        # Inference testing
└── tensorboard/
    └── cosyvoice2_emotional_sft/          # TensorBoard logs
```

---

## Summary of Commands

**Full pipeline (run in order):**

```bash
# 1. Download pretrained model (on Windows)
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

# 2. Build and start Docker (on Windows)
cd docker
docker-compose -f docker-compose.train.yml build
docker-compose -f docker-compose.train.yml up -d

# 3. Install dependencies (inside container)
docker exec -it cosyvoice-finetune /bin/bash
conda activate cosyvoice
pip install -r /workspace/CosyVoice/requirements.txt

# 4. Prepare data
cd /workspace/CosyVoice
chmod +x scripts/*.sh
bash scripts/prepare_data.sh

# 5. Train model
bash scripts/train_emotional_sft.sh

# 6. Average checkpoints
bash scripts/average_checkpoints.sh

# 7. Assemble final model
bash scripts/assemble_model.sh

# 8. Test inference
python scripts/test_emotional_inference.py
```

**Estimated total time:** 14-16 hours

---

## Next Steps

After successful training:

1. **Deploy for production:**
   ```bash
   # Use the final model for inference
   python3 webui.py --port 50000 --model_dir exp/cosyvoice2_emotional_final
   ```
   Access at `http://localhost:50000`

2. **Further fine-tuning:**
   - Increase epochs to 50 if emotional control is weak
   - Train Flow model for better acoustic quality (requires additional 6-8 hours)
   - Add more data from custom recordings

3. **Export for deployment:**
   ```bash
   # Export to JIT/ONNX for faster inference
   python cosyvoice/bin/export_jit.py --model_dir exp/cosyvoice2_emotional_final
   ```

---

## Dataset Information

**Filtered datasets (ESD + RAVDESS):**
- Total samples: 15,080 (train: ~11,800, val: ~1,700, test: ~1,600)
- ESD: 14,000 samples, 10 speakers, 5 emotions
- RAVDESS: 1,080 samples, 24 actors, well-balanced emotions
- CREMA-D: Excluded (noisy)

**Emotion distribution:**
- Angry: ~3,600 samples
- Happy: ~3,000 samples
- Neutral: ~3,000 samples
- Sad: ~3,000 samples
- Surprise: ~2,500 samples

---

## Support

If you encounter issues:

1. Check TensorBoard for training metrics
2. Review training log: `exp/cosyvoice2_emotional_sft/llm/train.log`
3. Verify GPU usage: `nvidia-smi` inside container
4. Check disk space: `df -h`
5. Refer to FAQ.md in the repository

For detailed CosyVoice documentation, see: `CLAUDE.md`
