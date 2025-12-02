# Common Issues and Solutions

**Last Updated**: 2025-12-02
**Purpose**: Quick troubleshooting guide for common problems

---

## Issue: Script Line Ending Errors

### Symptoms
```bash
bash scripts/prepare_data.sh
# Error: line 5: \r': command not found
# Error: syntax error near unexpected token `do\r''
```

### Cause
Windows line endings (CRLF) instead of Unix line endings (LF) in bash scripts.

### Solution

**Quick Fix** (Run in container):
```bash
dos2unix scripts/*.sh
```

**Or manually for each script**:
```bash
dos2unix scripts/prepare_data.sh
dos2unix scripts/train_emotional_sft.sh
dos2unix scripts/train_emotional_flow.sh
dos2unix scripts/train_emotional_hifigan.sh
```

**Verify it worked**:
```bash
bash scripts/prepare_data.sh
# Should now run without line ending errors
```

### Prevention
When creating/editing scripts on Windows:
1. Use an editor that supports Unix line endings (VS Code, Notepad++)
2. Set line ending to LF (not CRLF)
3. In VS Code: Bottom-right corner → Click "CRLF" → Select "LF"

---

## Issue: Docker Container Not Starting

### Symptoms
```bash
docker-compose -f docker/docker-compose.train.yml up -d
# Error: container exits immediately
```

### Solutions

**1. Check Docker logs**:
```bash
docker logs cosyvoice-finetune
```

**2. Check GPU access**:
```bash
nvidia-smi
# Should show GPU
```

**3. Rebuild container**:
```bash
docker-compose -f docker/docker-compose.train.yml down
docker-compose -f docker/docker-compose.train.yml build --no-cache
docker-compose -f docker/docker-compose.train.yml up -d
```

**4. Check NVIDIA Container Toolkit**:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# Should show GPU
```

---

## Issue: CUDA Out of Memory (OOM)

### Symptoms
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

### Solutions

**1. Reduce batch size** (recommended):
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
batch:
  max_frames_in_batch: 2000  # Reduce from 3000
```

**2. Increase gradient accumulation**:
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
train_conf:
  accum_grad: 4  # Increase from 2 (effective batch stays same)
```

**3. Disable AMP**:
```bash
# Edit training script, remove --use_amp flag
# Or comment out in scripts/train_emotional_sft.sh
```

**4. Check GPU memory**:
```bash
nvidia-smi
# Ensure nothing else is using GPU
```

**5. Kill other processes**:
```bash
# If something is using GPU
nvidia-smi
# Find PID, then:
kill -9 <PID>
```

---

## Issue: Data Files Not Found

### Symptoms
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/emotional_train.data.list'
```

### Solutions

**1. Run data preprocessing first**:
```bash
bash scripts/prepare_data.sh
```

**2. Verify data exists**:
```bash
ls -la data/full/manifests/
wc -l data/full/manifests/*.jsonl
```

**3. Check volume mounts** (in docker-compose.train.yml):
```yaml
volumes:
  - C:/Users/akshith/.../CosyVoice/data:/workspace/data:ro
```

**4. Verify inside container**:
```bash
docker exec -it cosyvoice-finetune bash
ls -la /workspace/data/full/manifests/
```

---

## Issue: Module Import Errors

### Symptoms
```python
ImportError: No module named 'cosyvoice'
ModuleNotFoundError: No module named 'matcha'
```

### Solutions

**1. Check PYTHONPATH**:
```bash
echo $PYTHONPATH
# Should include: /workspace/CosyVoice/third_party/Matcha-TTS
```

**2. Set PYTHONPATH manually**:
```bash
export PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
```

**3. Verify submodule initialized**:
```bash
ls -la third_party/Matcha-TTS/
# Should have files, not be empty
```

**4. Initialize submodule** (if empty):
```bash
# On host
git submodule update --init --recursive
```

**5. Rebuild Docker image**:
```bash
docker-compose -f docker/docker-compose.train.yml build --no-cache
```

---

## Issue: Training Loss Not Decreasing

### Symptoms
```
Epoch 1, loss=2.5
Epoch 5, loss=2.5
Epoch 10, loss=2.5  # Not decreasing
```

### Solutions

**1. Check data loading**:
```bash
# Verify emotion prefixes exist
head -10 data/emotional_speech/train/text
# Should see: <angry> text, <happy> text, etc.
```

**2. Reduce learning rate**:
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
train_conf:
  optim_conf:
    lr: 1e-6  # Reduce from 5e-6
```

**3. Verify checkpoint loading**:
```bash
# Check training log
grep "Loading checkpoint" exp/cosyvoice2_emotional_sft/llm/train.log
```

**4. Check for NaN/Inf**:
```bash
grep -i "nan\|inf" exp/cosyvoice2_emotional_sft/llm/train.log
```

---

## Issue: Training Diverges (NaN Loss)

### Symptoms
```
Epoch 3, loss=NaN
RuntimeError: Function 'PowBackward0' returned nan values
```

### Solutions

**1. Reduce learning rate significantly**:
```yaml
train_conf:
  optim_conf:
    lr: 1e-7  # Much lower
```

**2. Enable gradient clipping** (should already be enabled):
```yaml
train_conf:
  grad_clip: 5
```

**3. Disable AMP**:
```bash
# Remove --use_amp from training script
```

**4. Check data for corruption**:
```bash
# Re-run data preprocessing
bash scripts/prepare_data.sh
```

**5. Start from scratch**:
```bash
# Remove checkpoints and restart
rm -rf exp/cosyvoice2_emotional_sft/llm/*
bash scripts/train_emotional_sft.sh
```

---

## Issue: Slow Training

### Symptoms
Training is significantly slower than expected (>20 min/epoch on RTX 5090).

### Solutions

**1. Check GPU utilization**:
```bash
watch -n 1 nvidia-smi
# GPU-Util should be 80-95%
```

**2. Increase num_workers**:
```bash
# Edit scripts/train_emotional_sft.sh
--num_workers 8  # Increase from 4
```

**3. Increase prefetch**:
```bash
--prefetch 200  # Increase from 100
```

**4. Check disk I/O**:
```bash
# If data on slow drive, move to faster SSD
```

**5. Verify AMP enabled**:
```bash
# Should have --use_amp flag in training script
```

**6. Check data pipeline**:
```yaml
# Edit conf/cosyvoice2_emotional_sft.yaml
shuffle_size: 500  # Reduce from 1000 if slow
```

---

## Issue: Permission Errors

### Symptoms
```
PermissionError: [Errno 13] Permission denied: 'exp/...'
```

### Solutions

**1. Fix permissions** (inside container):
```bash
chmod -R 777 /workspace/CosyVoice/exp
chmod -R 777 /workspace/CosyVoice/tensorboard
```

**2. Fix ownership** (if needed):
```bash
chown -R root:root /workspace/CosyVoice/exp
```

**3. Check volume mounts**:
```yaml
# In docker-compose.train.yml
# Ensure exp/ and tensorboard/ are NOT read-only
volumes:
  - .../exp:/workspace/CosyVoice/exp  # NOT :ro
```

---

## Issue: Python Package Conflicts

### Symptoms
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

### Solutions

**1. Rebuild Docker with --no-cache**:
```bash
docker-compose -f docker/docker-compose.train.yml build --no-cache
```

**2. Check requirements.txt**:
```bash
cat requirements.txt | grep -i <package>
```

**3. Install specific versions**:
```bash
# Inside container
pip install torch==2.3.1 --force-reinstall
```

---

## Issue: Checkpoint Loading Fails

### Symptoms
```
RuntimeError: Error(s) in loading state_dict for Model
```

### Solutions

**1. Verify checkpoint exists**:
```bash
ls -lh pretrained_models/CosyVoice2-0.5B/llm.pt
```

**2. Check checkpoint integrity**:
```python
import torch
ckpt = torch.load('pretrained_models/CosyVoice2-0.5B/llm.pt', map_location='cpu')
print(ckpt.keys())
```

**3. Re-download pretrained model**:
```python
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
```

---

## Issue: TensorBoard Not Showing Data

### Symptoms
TensorBoard opens but shows no data.

### Solutions

**1. Verify logs exist**:
```bash
ls -la tensorboard/cosyvoice2_emotional_sft/llm/
```

**2. Correct logdir path**:
```bash
tensorboard --logdir tensorboard/cosyvoice2_emotional_sft/llm --port 6006
# NOT --logdir tensorboard/ (too broad)
```

**3. Check training log interval**:
```yaml
train_conf:
  log_interval: 50  # Logs every 50 steps
```

**4. Wait for first log**:
```bash
# TensorBoard updates every 30 seconds
# Give it a minute after training starts
```

---

## Issue: Emotion Prefix Not Working

### Symptoms
Generated audio doesn't reflect emotion in text prefix.

### Solutions

**1. Verify training data has prefixes**:
```bash
grep "<angry>" data/emotional_speech/train/text | head -5
```

**2. Check inference format**:
```python
# Correct
text = '<angry> I cannot believe this!'

# Incorrect
text = 'I cannot believe this! [angry]'
text = 'angry: I cannot believe this!'
```

**3. Use correct inference method**:
```python
# For fine-tuned model
cosyvoice.inference_sft(text, stream=False)

# NOT inference_zero_shot or inference_cross_lingual
```

**4. Verify model fine-tuned**:
```bash
# Should use fine-tuned model, not pretrained
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-Emotional-SFT')
```

---

## Issue: Preprocessing Takes Too Long

### Symptoms
Stage 2 (speech tokens) taking >3 hours.

### Solutions

**1. Check GPU usage**:
```bash
nvidia-smi
# Speech tokenizer should use GPU
```

**2. Reduce thread count** (paradoxically can help):
```bash
# Edit scripts/prepare_data.sh
--num_thread 4  # Reduce from 8
```

**3. Check ONNX Runtime**:
```python
import onnxruntime
print(onnxruntime.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

**4. Process in batches**:
```bash
# Split data into smaller chunks
# Process each chunk separately
```

---

## Quick Diagnostic Commands

### Check Everything Is OK
```bash
# GPU
nvidia-smi

# Python environment
python --version
echo $PYTHONPATH
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

# Data
ls -la data/full/manifests/
wc -l data/emotional_*.data.list

# Models
ls -lh pretrained_models/CosyVoice2-0.5B/*.pt

# Disk space
df -h
```

### Check Training Status
```bash
# Latest checkpoint
ls -t exp/cosyvoice2_emotional_sft/llm/epoch_*.pt | head -1

# Training log
tail -20 exp/cosyvoice2_emotional_sft/llm/train.log

# Loss trend
grep "loss=" exp/cosyvoice2_emotional_sft/llm/train.log | tail -20
```

---

## Getting Help

If issues persist:

1. **Check logs**:
   ```bash
   tail -100 exp/cosyvoice2_emotional_sft/llm/train.log
   docker logs cosyvoice-finetune
   ```

2. **Review documentation**:
   - [System Configuration](SYSTEM_CONFIGURATION.md)
   - [Quick Reference](QUICK_REFERENCE.md)
   - [Dataset Documentation](DATASET_DOCUMENTATION.md)

3. **Search GitHub issues**:
   - https://github.com/FunAudioLLM/CosyVoice/issues

4. **Open new issue** with:
   - Error message (full traceback)
   - System specs (GPU, Docker version)
   - Steps to reproduce
   - Relevant log files

---

## Emergency Reset

If everything is broken and you need to start fresh:

```bash
# 1. Stop container
docker-compose -f docker/docker-compose.train.yml down

# 2. Remove container and image
docker rmi cosyvoice-train:latest

# 3. Clean up (CAREFUL - deletes training outputs)
rm -rf exp/*
rm -rf tensorboard/*
rm -rf data/emotional_speech/*

# 4. Rebuild from scratch
docker-compose -f docker/docker-compose.train.yml build --no-cache
docker-compose -f docker/docker-compose.train.yml up -d

# 5. Re-run data prep
docker exec -it cosyvoice-finetune bash
bash scripts/prepare_data.sh

# 6. Start training
bash scripts/train_emotional_sft.sh
```

---

**Document Status**: Living document - updated as new issues discovered
**Last Updated**: 2025-12-02
**Contributions**: Add new issues as you encounter them!
