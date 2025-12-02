# Experimental Design: Emotional TTS Fine-Tuning

## Dataset
- **Train**: 14,080 utterances | **Val**: 1,822 utterances (11.5% split)
- **Sources**: ESD + RAVDESS (CREMA-D excluded)
- **Audio**: 24kHz, ~12 hours total

## Training Pipeline
1. **Flow Matching** (tokens → mel-spec) - Emotional prosody control
2. **HiFiGAN** (mel-spec → waveform) - High-quality synthesis

---

## Flow Matching Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | 2e-5 | Initial 1e-4 caused gradient explosion (grad_norm 20→80) |
| `scheduler` | constantlr | Warmup made training unstable; constant LR more stable |
| `grad_clip` | 0.5 | Strict clipping prevents divergence |
| `accum_grad` | 4 | Larger effective batch = smoother gradients |
| `max_epoch` | 15 | Lower LR needs more epochs |
| `weight_decay` | 0.01 | L2 regularization |

**Training Time**: ~3.5-4 hours on RTX 5090

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
