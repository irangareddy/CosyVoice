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

## Lessons Learned

### Flow Matching Training Issues
1. **Dataset mismatch**: ESD+RAVDESS may have distribution shift between train/val
2. **Model capacity**: 14k samples insufficient for stable flow matching fine-tuning
3. **Hyperparameter sensitivity**: Flow extremely sensitive to LR (1e-4 explodes, 2e-5 still diverges)
4. **Gradient clipping ineffective**: Even 0.5 clip couldn't prevent divergence

### Recommendations
- ❌ **DO NOT** fine-tune Flow on small emotional datasets (<50k utterances)
- ✅ **DO** fine-tune HiFiGAN vocoder instead (more stable, GAN losses handle small data better)
- ✅ **DO** use pretrained Flow with fine-tuned LLM + HiFiGAN
- ⚠️ If Flow fine-tuning required: Try lr=5e-6, freeze encoder layers, train decoder only
