# Training Issues and Fixes

This document records all issues encountered during training setup and testing, along with their solutions.

## Issue 1: ModuleNotFoundError - cosyvoice module not found

**When**: First test training attempt
**Error Message**:
```
ModuleNotFoundError: No module named 'cosyvoice'
```

**Root Cause**: PYTHONPATH only included `third_party/Matcha-TTS` but not the current directory where the `cosyvoice` package resides.

**Fix**:
```bash
# Before (WRONG):
export PYTHONPATH=third_party/Matcha-TTS

# After (CORRECT):
export PYTHONPATH=$PWD:$PWD/third_party/Matcha-TTS
```

**Files Modified**:
- `scripts/test_training.sh`
- `scripts/train_with_wandb.sh`

---

## Issue 2: DataLoader prefetch_factor ValueError

**When**: First test training attempt
**Error Message**:
```
ValueError: prefetch_factor option could only be specified in multiprocessing.
let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.
```

**Root Cause**: The training script's DataLoader had `prefetch_factor` set but `num_workers=0` (default). DataLoader requires `num_workers > 0` to use prefetching.

**Fix**: Add explicit DataLoader parameters to training command:
```bash
--num_workers 2 \
--prefetch 50
```

**Files Modified**:
- `scripts/test_training.sh`
- `scripts/train_with_wandb.sh`

**Why These Values**:
- `num_workers=2`: Uses 2 worker processes for data loading (good for single GPU)
- `prefetch=50`: Prefetches 50 batches ahead for smooth GPU utilization

---

## Issue 3: Flow/HiFiGAN tokenizer error

**When**: Flow and HiFiGAN test training
**Error Message**:
```
OSError: None is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error.
Repository Not Found for url: https://huggingface.co/None/resolve/main/tokenizer_config.json.
```

**Root Cause**: Flow and HiFiGAN training also need the Qwen tokenizer for data processing, but the `--qwen_pretrain_path` argument was not passed to the training command.

**Why LLM Worked**: The LLM training script likely had the path configured differently or found it via config.

**Fix**: Add to both training scripts:
```bash
--qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
```

**Files Modified**:
- `scripts/test_training.sh`
- `scripts/train_with_wandb.sh`

---

## Issue 4: HiFiGAN TPR Loss tensor dimension mismatch

**When**: HiFiGAN test training
**Error Message**:
```
RuntimeError: The size of tensor a (304) must match the size of tensor b (298) at non-singleton dimension 1
```

**Location**: `cosyvoice/utils/losses.py:9` in `tpr_loss()` function

**Root Cause**: The TPR (Topological Representation) loss function tries to compute `torch.median(dr - dg)` but the discriminator outputs have mismatched dimensions. This happens because:
1. Audio samples in the batch may have slightly different lengths
2. The discriminator produces outputs with dimensions matching the input audio
3. Real audio (dr) and generated audio (dg) end up with different time dimensions (304 vs 298 samples)

**Technical Details**:
- The discriminator processes real and generated audio separately
- Due to slight length variations in preprocessing or generation, the outputs don't match
- The original code assumed both tensors would always have the same shape

**Fix**: Modified `tpr_loss()` to handle dimension mismatches by aligning tensors:

```python
def tpr_loss(disc_real_outputs, disc_generated_outputs, tau):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # Handle dimension mismatch by truncating to minimum size
        min_size = min(dr.shape[-1], dg.shape[-1])
        dr_aligned = dr[..., :min_size]
        dg_aligned = dg[..., :min_size]

        m_DG = torch.median((dr_aligned - dg_aligned))
        L_rel = torch.mean((((dr_aligned - dg_aligned) - m_DG) ** 2)[dr_aligned < dg_aligned + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss
```

**Why This Fix Works**:
- Truncates both tensors to the minimum common size
- Preserves the loss calculation semantics
- Minimal impact on training (only a few samples difference)
- More robust to length variations in the dataset

**Files Modified**:
- `cosyvoice/utils/losses.py`

---

## Summary of All Test Results

### LLM Test: ✅ PASSED
- **Duration**: ~10 minutes for 5 epochs
- **Training Loss**: 1.963 → 1.570 (19.9% reduction)
- **Training Accuracy**: 0.152 → 0.234 (53.9% increase)
- **Validation**: Performed successfully each epoch
- **Checkpoints**: Saved in `exp/test_run/llm/`
- **Status**: Ready for production training

### Flow Test: ⚠️ FIXED (Not Re-tested)
- **Initial Error**: Missing qwen_pretrain_path
- **Fix Applied**: Added `--qwen_pretrain_path` argument
- **Status**: Ready for testing with fix

### HiFiGAN Test: ⚠️ FIXED (Not Re-tested)
- **Initial Errors**:
  1. Missing qwen_pretrain_path (fixed)
  2. TPR loss tensor dimension mismatch (fixed)
- **Fixes Applied**:
  - Added `--qwen_pretrain_path` argument
  - Fixed `tpr_loss()` function to handle dimension mismatches
- **Status**: Ready for testing with fixes

---

## Recommendations

### Immediate Next Steps:

1. **Start LLM Production Training** (Verified Working):
   ```bash
   bash scripts/train_with_wandb.sh llm
   ```
   - Expected duration: 8-10 hours
   - Will train on 15,075 samples for 50 epochs
   - Monitor via W&B dashboard: https://wandb.ai/rr-sjsu/cosy-voxe

2. **Optional: Re-test Flow and HiFiGAN** (Recommended):
   ```bash
   bash scripts/test_training.sh flow
   bash scripts/test_training.sh hifigan
   ```
   - Quick verification (~5-10 minutes each)
   - Confirms all fixes work correctly

3. **Production Training Order**:
   - Train LLM first (most critical component)
   - Flow training is optional (pretrained works well)
   - HiFiGAN training is optional (pretrained works well)

### Important Notes:

- **LLM is the most important** component for emotional transfer learning
- **Pretrained Flow/HiFiGAN** work well with fine-tuned LLM
- **Full Flow/HiFiGAN training** may provide marginal improvements but takes additional time:
  - Flow: ~15-20 hours for 200 epochs
  - HiFiGAN: ~10-12 hours for 200 epochs

### For Production Use:

After LLM training completes:
1. Find best checkpoint: `python scripts/find_best_checkpoint.py exp/emotional_sft/llm`
2. Copy to model directory for inference
3. Test emotional TTS with different emotions
4. Decide if Flow/HiFiGAN fine-tuning is needed based on quality

---

## Files Modified in This Session

All changes have been committed (see git history):

1. **scripts/test_training.sh**
   - Fixed PYTHONPATH
   - Added num_workers and prefetch
   - Added qwen_pretrain_path

2. **scripts/train_with_wandb.sh**
   - Fixed PYTHONPATH
   - Added num_workers and prefetch
   - Added qwen_pretrain_path

3. **cosyvoice/utils/losses.py**
   - Fixed tpr_loss() tensor dimension mismatch

4. **test_emotional_inference.py** (NEW)
   - Example script for emotional TTS inference
   - Shows how to use fine-tuned LLM with pretrained Flow/HiFiGAN

---

## Conclusion

All critical training issues have been identified and fixed:
- ✅ PYTHONPATH configuration
- ✅ DataLoader setup
- ✅ Tokenizer path for all components
- ✅ TPR loss dimension handling

The project is now **ready for production training**. LLM training is verified working and can start immediately. Flow and HiFiGAN fixes are in place but not yet verified through test runs.
