# 🎯 Training Readiness Checklist

## ✅ Prerequisites Complete

### 1. Data Preprocessing ✓
- **Train**: 15,075 samples across 16 parquet files
- **Validation**: 1,822 samples across 2 parquet files
- **Test**: 2,038 samples across 3 parquet files
- **Speaker embeddings**: Extracted with CAM++ model
- **Speech tokens**: Extracted with speech_tokenizer_v2
- **Location**: `data/full/parquet/{train,val,test}/`

### 2. Model Files ✓
- **Pretrained Model**: CosyVoice2-0.5B in `pretrained_models/`
  - `llm.pt` - Language model
  - `flow.pt` - Flow matching model
  - `hifigan.pt` - Vocoder model
  - `campplus.onnx` - Speaker encoder
  - `speech_tokenizer_v2.onnx` - Speech tokenizer

### 3. Configuration ✓
- **Main Config**: `conf/cosyvoice2_emotional_sft.yaml`
  - Learning rate: 5e-6 (LLM/Flow), 0.0002 (HiFiGAN)
  - Max epochs: 50 (LLM/Flow), 200 (HiFiGAN)
  - Gradient accumulation: 2
  - W&B tracking: Configured

- **Test Config**: `conf/cosyvoice2_test.yaml` (auto-generated)
  - Max epochs: 1
  - Same as main config but faster for testing

### 4. W&B Integration ✓
- **Project**: `rr-sjsu/cosy-voxe`
- **Entity**: `rr-sjsu`
- **API Key**: Configured in YAML
- **Features**: Real-time loss tracking, learning curves, system metrics

### 5. Training Scripts ✓
- **Test Script**: `scripts/test_training.sh` - 1 epoch verification
- **Full Training**: `scripts/train_with_wandb.sh` - Production training
- **Checkpoint Finder**: `scripts/find_best_checkpoint.py` - Post-training analysis

---

## 🧪 Test Training (Recommended First Step)

Before committing to 8-30 hour training runs, verify everything works:

### Quick Test (Single Model)
```bash
# Test LLM only (~5-10 minutes)
bash scripts/test_training.sh llm

# Test Flow only (~5-10 minutes)
bash scripts/test_training.sh flow

# Test HiFiGAN only (~5-10 minutes)
bash scripts/test_training.sh hifigan
```

### Complete Test (All Models)
```bash
# Test all three models sequentially (~20-30 minutes)
bash scripts/test_training.sh all
```

**What the test does:**
1. Creates mini dataset (first parquet file only, ~1000 samples)
2. Runs 1 complete epoch
3. Verifies checkpoint creation
4. Confirms data loading, model forward/backward, optimization work
5. W&B runs in offline mode (no network required)

**Expected output:**
```
✅ llm test PASSED
   Checkpoint: exp/test_run/llm/
✅ flow test PASSED
   Checkpoint: exp/test_run/flow/
✅ hifigan test PASSED
   Checkpoint: exp/test_run/hifigan/

🎉 ALL TESTS PASSED! Ready for full training.
```

---

## 🚀 Full Training

Once tests pass, start full training:

### Option 1: Train LLM Only (Recommended)
Fastest path to usable model (8-10 hours):
```bash
export PYTHONPATH=third_party/Matcha-TTS
bash scripts/train_with_wandb.sh llm
```

**Monitor:**
- W&B dashboard: https://wandb.ai/rr-sjsu/cosy-voxe
- Local TensorBoard: `tensorboard --logdir tensorboard/emotional_sft/llm --port 6006`
- Checkpoints: `exp/emotional_sft/llm/epoch_*_whole.pt`

### Option 2: Train LLM + Flow (Better Quality)
Improves speech quality (20-35 hours total):
```bash
# 1. Train LLM first (8-10 hours)
bash scripts/train_with_wandb.sh llm

# 2. Wait for completion, then train Flow (12-25 hours)
bash scripts/train_with_wandb.sh flow
```

### Option 3: Full Pipeline (Best Quality)
Complete fine-tuning (36-62 hours total):
```bash
# 1. Train LLM (8-10 hours)
bash scripts/train_with_wandb.sh llm

# 2. Train Flow (12-25 hours)
bash scripts/train_with_wandb.sh flow

# 3. Train HiFiGAN (16-27 hours)
bash scripts/train_with_wandb.sh hifigan
```

---

## 📊 Training Time Estimates

| Component | Epochs | Time/Epoch | Total Time | When to Use |
|-----------|--------|------------|------------|-------------|
| **LLM** | 50 | 10-12 min | 8-10 hours | Minimum for usable model |
| **Flow** | 50 | 15-20 min | 12-25 hours | Improves naturalness |
| **HiFiGAN** | 200 | 5-8 min | 16-27 hours | Best audio quality |

**Hardware**: RTX 5090 (32GB VRAM)
**Effective batch size**: 2 × batch_size (gradient accumulation=2)

---

## 🔍 Monitoring Training

### Real-Time Metrics

**W&B Dashboard** (https://wandb.ai/rr-sjsu/cosy-voxe):
- ✅ Training loss (per step)
- ✅ Validation (CV) loss (per epoch)
- ✅ Learning rate schedule
- ✅ Gradient norms
- ✅ GPU utilization
- ✅ Training curves

**TensorBoard** (local):
```bash
tensorboard --logdir tensorboard/emotional_sft/llm --port 6006
# Open: http://localhost:6006
```

### What to Watch For

**Healthy Training:**
- ✅ Loss decreases smoothly
- ✅ CV loss tracks training loss (small gap)
- ✅ Grad norm stable (0.1 - 5.0 range)
- ✅ No NaN/Inf values

**Warning Signs:**
- ⚠️ CV loss increasing while train loss decreases → **Overfitting** (stop early)
- ⚠️ Grad norm exploding (>10) → **Gradient explosion** (reduce LR)
- ⚠️ Loss plateauing early → May need more epochs or higher LR

---

## 🛑 Early Stopping

**Important**: No automatic early stopping exists!

Monitor CV loss in W&B and manually stop if:
1. CV loss stops decreasing for 5+ epochs
2. CV loss starts increasing (overfitting)
3. Loss plateaus and you're satisfied with quality

**Stop training**:
```bash
# Find the training process
ps aux | grep train.py

# Kill it
kill <PID>
```

**Find best checkpoint**:
```bash
python scripts/find_best_checkpoint.py \
  --model_dir exp/emotional_sft/llm \
  --top_k 5
```

This shows which epoch had lowest CV loss.

---

## 📦 After Training

### 1. Find Best Checkpoint

```bash
# For LLM
python scripts/find_best_checkpoint.py --model_dir exp/emotional_sft/llm

# For Flow
python scripts/find_best_checkpoint.py --model_dir exp/emotional_sft/flow

# For HiFiGAN
python scripts/find_best_checkpoint.py --model_dir exp/emotional_sft/hifigan
```

Output example:
```
Top 5 checkpoints:
1. Epoch  23 | CV Loss: 4.2315 | exp/emotional_sft/llm/epoch_23_whole.pt
2. Epoch  24 | CV Loss: 4.2401 | exp/emotional_sft/llm/epoch_24_whole.pt
3. Epoch  22 | CV Loss: 4.2534 | exp/emotional_sft/llm/epoch_22_whole.pt

Best checkpoint recommendation:
  Epoch: 23
  CV Loss: 4.2315
  Path: exp/emotional_sft/llm/epoch_23_whole.pt

To use this checkpoint:
  cp exp/emotional_sft/llm/epoch_23_whole.pt pretrained_models/CosyVoice2-0.5B/llm.pt
```

### 2. Copy Best Checkpoints to Model Directory

```bash
# Copy best LLM
cp exp/emotional_sft/llm/epoch_<best>_whole.pt pretrained_models/CosyVoice2-0.5B/llm.pt

# Copy best Flow (if trained)
cp exp/emotional_sft/flow/epoch_<best>_whole.pt pretrained_models/CosyVoice2-0.5B/flow.pt

# Copy best HiFiGAN (if trained)
cp exp/emotional_sft/hifigan/epoch_<best>_whole.pt pretrained_models/CosyVoice2-0.5B/hifigan.pt
```

### 3. Test Inference

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# Load your fine-tuned model
model = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

# Test with emotional prompt
prompt_speech = load_wav('path/to/emotional_sample.wav', 16000)
text = "I am so excited to see you!"

for i, output in enumerate(model.inference_zero_shot(
    text,
    "I am so excited!",
    prompt_speech,
    stream=False
)):
    torchaudio.save(f'output_emotional_{i}.wav',
                   output['tts_speech'],
                   model.sample_rate)
```

---

## 🎯 Quick Reference

### Test Before Training
```bash
bash scripts/test_training.sh all  # ~20-30 min
```

### Start Training
```bash
bash scripts/train_with_wandb.sh llm  # Recommended first
```

### Monitor
- W&B: https://wandb.ai/rr-sjsu/cosy-voxe
- TensorBoard: `tensorboard --logdir tensorboard/emotional_sft/llm`

### Find Best Model
```bash
python scripts/find_best_checkpoint.py --model_dir exp/emotional_sft/llm
```

### Use Fine-Tuned Model
```bash
cp exp/emotional_sft/llm/epoch_<best>_whole.pt pretrained_models/CosyVoice2-0.5B/llm.pt
```

---

## 📚 Additional Resources

- **W&B Integration Guide**: `WANDB_INTEGRATION.md`
- **General Training Guide**: `TRAINING_GUIDE.md`
- **Model Comparison**: `MODEL_COMPARISON_GUIDE.md`
- **Inference Guide**: `INFERENCE_GUIDE.md`

---

## ✅ Pre-Flight Checklist

Before starting full training, verify:

- [ ] Test training completed successfully (`bash scripts/test_training.sh all`)
- [ ] W&B dashboard accessible (https://wandb.ai/rr-sjsu/cosy-voxe)
- [ ] Sufficient disk space (~50GB for checkpoints + logs)
- [ ] GPU available and idle (`nvidia-smi`)
- [ ] No other training jobs running
- [ ] Internet connection stable (for W&B logging)
- [ ] You understand how to monitor and stop training
- [ ] You know how to find the best checkpoint afterward

If all checked, you're ready to train! 🚀

**Recommended workflow:**
1. Run test: `bash scripts/test_training.sh llm` (5-10 min)
2. If passed, start LLM training: `bash scripts/train_with_wandb.sh llm` (8-10 hours)
3. Monitor on W&B and stop if overfitting
4. Find best checkpoint and evaluate
5. Optionally train Flow and HiFiGAN for better quality
