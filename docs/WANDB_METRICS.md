# W&B Training Metrics Guide

## Overview
This document explains the metrics logged to Weights & Biases during HiFiGAN vocoder fine-tuning for emotional TTS.

**W&B Project**: `rr-sjsu/cosy-voxe`
**Run**: `emotional-hifigan`
**Dashboard**: https://wandb.ai/rr-sjsu/cosy-voxe/runs/610jaivb

---

## Key Metrics

### 1. CV/loss (Validation Loss)
**What it measures**: Overall validation performance (lower = better)

**Formula**: Combined mel-spectrogram, F0 (pitch), and generator losses on validation set

**Target**: < 80 for good quality

**Chart interpretation**:
- **Decreasing trend**: Model improving
- **Plateau**: Model converged
- **Increasing**: Overfitting (use earlier checkpoint)

**Example values**:
- Epoch 0: 100.71 (baseline)
- Epoch 4: 80.59 (20% improvement) ✅

---

### 2. CV/loss_mel (Mel-Spectrogram Loss)
**What it measures**: Audio quality and clarity (lower = better)

**Formula**: L1 distance between generated and ground-truth mel-spectrograms

**Target**: < 0.5 for production quality

**Chart interpretation**:
- Critical metric for audio quality
- Directly correlates with clarity and detail
- Lower = less robotic/metallic artifacts

**Example values**:
- Epoch 0: 0.674 (robotic artifacts)
- Epoch 4: 0.488 (approaching target) ✅

---

### 3. CV/loss_f0 (Pitch/F0 Loss)
**What it measures**: Prosody and naturalness (lower = better)

**Formula**: L1 distance between generated and ground-truth pitch contours

**Target**: < 30 for natural prosody

**Chart interpretation**:
- Higher F0 loss = monotone, unnatural speech
- Lower F0 loss = natural pitch variation and emotion
- Critical for emotional expressiveness

**Example values**:
- Epoch 0: 59.44 (monotone)
- Epoch 4: 43.73 (improving) ⚠️ (still needs work)

---

### 4. TRAIN/loss_gen (Generator Training Loss)
**What it measures**: How well generator fools discriminator (lower = better)

**Formula**: GAN adversarial loss + feature matching loss

**Target**: 2.0-3.0 for stable training

**Chart interpretation**:
- Should decrease gradually
- Too low (<1.0) = mode collapse
- Too high (>5.0) = generator not learning
- Oscillates normally in GAN training

**Example values**:
- Epoch 0: 3.06
- Epoch 4: 2.95 ✅

---

### 5. TRAIN/loss_disc (Discriminator Training Loss)
**What it measures**: How well discriminator distinguishes real vs. fake (lower = better)

**Formula**: Binary cross-entropy on real/fake classification

**Target**: 3.0-4.0 for stable training

**Chart interpretation**:
- Should stay relatively stable
- Too low (<2.0) = discriminator too strong
- Too high (>6.0) = discriminator too weak
- Balance with loss_gen is critical

**Example values**:
- Epoch 0: 3.44
- Epoch 4: 3.31 ✅

---

### 6. TRAIN/loss_fm (Feature Matching Loss)
**What it measures**: Intermediate feature similarity (lower = better)

**Formula**: L1 distance between discriminator features for real/fake

**Target**: 3.5-6.0 for stable training

**Chart interpretation**:
- Helps stabilize GAN training
- Prevents mode collapse
- Encourages diverse, realistic outputs

**Example values**:
- Epoch 0: 3.95
- Epoch 4: 5.81

---

### 7. TRAIN/grad_norm (Gradient Norm)
**What it measures**: Magnitude of gradients during training

**Formula**: L2 norm of all parameter gradients

**Target**: < 10 for stable training (clipped at 5.0)

**Chart interpretation**:
- Sudden spikes = gradient explosion
- Consistently high (>20) = instability
- Low (<1) = learning stagnation
- Should be relatively stable

**Example values**:
- Usually 4-15 (healthy range) ✅

---

### 8. TRAIN/lr (Learning Rate)
**What it measures**: Current optimizer learning rate

**Value**: 0.0002 (constant)

**Chart interpretation**:
- Flat line for constantlr scheduler ✅
- Would show decay for other schedulers

---

## Chart Patterns to Watch

### ✅ Healthy Training
- CV/loss steadily decreasing
- loss_mel approaching < 0.5
- loss_f0 gradually decreasing
- loss_gen and loss_disc balanced (within 0.5 of each other)
- grad_norm stable (no spikes)

### ⚠️ Warning Signs
- CV/loss plateau (model converged, stop training)
- loss_gen and loss_disc diverging (>2.0 gap)
- grad_norm spiking (>50)

### ❌ Critical Issues
- CV/loss increasing (overfitting, use earlier checkpoint)
- loss_gen << loss_disc (<2.0 gap) = mode collapse
- loss_disc << loss_gen (>3.0 gap) = discriminator dominating
- grad_norm exploding (>100) = training unstable

---

## How to Find Best Checkpoint

1. **Sort by CV/loss** (lowest = best overall)
2. **Check loss_mel** (must be < 0.6 minimum)
3. **Check loss_f0** (lower = more natural prosody)
4. **Verify no overfitting** (CV/loss not increasing)

**Current best**: Epoch 4 (CV: 80.59, Mel: 0.488, F0: 43.73)

---

## Comparison Script Usage

```bash
# Inside Docker container
docker exec cosyvoice-finetune python /workspace/CosyVoice/compare_checkpoints.py

# Output: Ranked table of all epochs by metrics
```

This will show:
- All epochs sorted by CV loss
- Best checkpoint for each metric
- Recommended checkpoint to use

---

## Metric Relationships

```
CV/loss = weighted_sum(loss_mel, loss_f0, loss_gen, loss_fm, loss_tpr)
              ↓
        Audio Quality

loss_mel    →  Clarity, detail
loss_f0     →  Prosody, emotion
loss_gen    →  Naturalness
loss_fm     →  Stability
```

All metrics must improve together for best results. Optimizing only one can hurt others.
