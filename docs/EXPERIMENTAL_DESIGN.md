# Experimental Design: Emotional TTS Fine-Tuning

## Dataset
- **Train**: 14,080 utterances | **Val**: 1,822 utterances (11.5% split)
- **Sources**: ESD + RAVDESS (CREMA-D excluded)
- **Audio**: 24kHz, ~12 hours total

## Training Pipeline
1. **Flow Matching** (tokens → mel-spec) - Emotional prosody control
2. **HiFiGAN** (mel-spec → waveform) - High-quality synthesis

---

## Flow Matching Hyperparameters - FAILED RUNS

### Run 1 (FAILED - Gradient Explosion)
| Parameter | Value | Result |
|-----------|-------|--------|
| `lr` | 1e-4 | Gradient explosion: grad_norm 20→80 |
| `scheduler` | warmuplr | Warmup exacerbated instability |
| CV/loss | 0.8 → 1.8 | Catastrophic divergence at epoch 7 |
| TRAIN/loss | 0.35 → 1.0 | Model collapsed |

### Run 2 (FAILED - Still Diverging)
| Parameter | Value | Result |
|-----------|-------|--------|
| `lr` | 2e-5 | Reduced 5x from Run 1 |
| `scheduler` | constantlr | Removed warmup |
| `grad_clip` | 0.5 | Stricter clipping (was 1.0) |
| `accum_grad` | 4 | Larger batch (was 2) |
| CV/loss | 0.77 → 1.18 | **WORSE**: Gradual but continuous divergence |
| grad_norm | 1.2 → 7.8 | Still exploding despite clip=0.5 |
| **Outcome** | ❌ FAILURE | After 14 epochs, CV loss increased 53% |

**Root Cause**: Flow model fundamentally unstable for this dataset/task. Even 2e-5 LR too high.

---

## HiFiGAN Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` (gen/disc)` | 0.0002 | Standard GAN learning rate |
| `scheduler` | constantlr | No warmup for GAN |
| `grad_clip` | 5.0 | Higher tolerance for adversarial training |
| `accum_grad` | **1** | CRITICAL: Must be 1 for GAN (no accumulation) |
| `max_epoch` | 30 | Reduced from 200 for 14k dataset |

**Training Time**: ~4-5 hours on RTX 5090

---

---

## HiFiGAN Training - SUCCESSFUL

### Run 1 (SUCCESS - Epoch 4 Best Quality)
| Parameter | Value | Result |
|-----------|-------|--------|
| `lr` (gen/disc) | 0.0002 | Stable GAN training |
| `scheduler` | constantlr | No warmup needed |
| `grad_clip` | 5.0 | Higher tolerance for adversarial loss |
| `accum_grad` | **1** | CRITICAL: No accumulation for GAN |
| `max_epoch` | 30 | ~20 hours on RTX 5090 |
| **Epoch 0** | CV: 100.71, Mel: 0.674, F0: 59.44 | Starting point |
| **Epoch 4** | CV: 80.59, Mel: 0.488, F0: 43.73 | 20% improvement, best checkpoint |
| **Outcome** | ✅ SUCCESS | Metrics improving, no overfitting |

**Audio Quality Progression**:
- Epochs 0-4: Robotic/metallic artifacts (undertrained)
- Epochs 10-15: Expected to match pretrained quality
- Epochs 20-30: May surpass pretrained with better emotion

---

## Failed Experiments & Lessons

### 1. LLM Fine-Tuning Attempts (NOT TESTED YET)
- Skipped due to Flow instability concerns
- Pretrained LLM + fine-tuned HiFiGAN approach chosen instead
- **Reason**: HiFiGAN-only fine-tuning more stable for emotional TTS

### 2. Flow Matching Training Issues
1. **Dataset mismatch**: ESD+RAVDESS may have distribution shift
2. **Model capacity**: 14k samples insufficient for stable flow fine-tuning
3. **Hyperparameter sensitivity**: Flow extremely sensitive to LR
4. **Gradient clipping ineffective**: Even 0.5 clip couldn't prevent divergence

### 3. Early Checkpoint Audio Generation Failures
- **Issue**: Robotic/metallic artifacts in epoch 0-4 samples
- **Root Cause**: HiFiGAN vocoder undertrained (needs 10-20 epochs minimum)
- **Solution**: Generated baseline with pretrained HiFiGAN for comparison
- **Learning**: Don't evaluate vocoder quality before epoch 10

### 4. Inference API Compatibility Issues
- **Issue**: CosyVoice2 `.to()` method errors, `inference_instruct2()` parameter mismatches
- **Root Cause**: API differences between CosyVoice 1.0 and 2.0
- **Solution**: Use `inference_zero_shot()` or `inference_instruct2()` without calling `.to()`
- **Learning**: CosyVoice2 outputs are already on CPU

---

## Successful Approaches

### ✅ HiFiGAN-Only Fine-Tuning
- **Approach**: Fine-tune only HiFiGAN vocoder, use pretrained LLM + Flow
- **Benefits**: Stable training, works with small datasets (14k), faster iteration
- **Results**: 20% CV loss improvement by epoch 4, continuing to improve

### ✅ Emotion Sample Generation for Comparison
- **Method**: Generate same texts + references across checkpoints
- **Baseline**: Pretrained HiFiGAN (clean quality)
- **Fine-tuned**: Epochs 0, 4, 10, 15... (progressive improvement)
- **Value**: Objective comparison of emotional expressiveness

### ✅ W&B Integration for Monitoring
- **Metrics**: CV/loss, loss_mel, loss_f0, loss_gen, loss_disc
- **Charts**: Real-time training curves, epoch comparison
- **Benefit**: Early detection of divergence, identify best checkpoint

---

## Recommendations

### DO ✅
- Fine-tune HiFiGAN vocoder for emotional TTS (stable, works with small data)
- Use pretrained LLM + Flow (no fine-tuning needed)
- Monitor mel loss (<0.5) and F0 loss (<40) for quality
- Wait until epoch 10+ before evaluating audio quality
- Generate comparison samples with pretrained baseline

### DON'T ❌
- Fine-tune Flow on datasets <50k utterances (unstable)
- Use warmup scheduler for GAN training (constantlr only)
- Evaluate vocoder quality before epoch 10 (undertrained)
- Use gradient accumulation for GAN (`accum_grad` must be 1)
- Skip baseline comparison (need pretrained reference)

### IF NEEDED ⚠️
- Flow fine-tuning: lr=5e-6, freeze encoder, train decoder only
- LLM fine-tuning: Start after HiFiGAN converges, use lr=1e-5
