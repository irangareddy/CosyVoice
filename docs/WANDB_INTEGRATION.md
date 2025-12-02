# Weights & Biases (W&B) Integration

## Overview

W&B tracking has been fully integrated into CosyVoice2 training pipeline to monitor:
- Training loss (per step)
- Validation (CV) loss (per epoch)
- Learning rate
- Gradient norms
- Training curves

## Configuration

W&B is configured in `conf/cosyvoice2_emotional_sft.yaml`:

```yaml
train_conf:
    # ... other settings ...
    wandb_project: "rr-sjsu/cosy-voxe"
    wandb_entity: "rr-sjsu"
    wandb_api_key: "fa1dd39746b8c27d133e3ac6d49ce74207706ff5"
    wandb_run_name: "emotional-sft-llm"  # Auto-generated if not specified
```

## Usage

### Method 1: Direct Training (Automatic)

W&B will initialize automatically if configured in the YAML:

```bash
export PYTHONPATH=third_party/Matcha-TTS
torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_id=100 --rdzv_backend='c10d' --rdzv_endpoint='localhost:29400' \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice2_emotional_sft.yaml \
  --train_data data/full/parquet/train/data.list \
  --cv_data data/full/parquet/val/data.list \
  --model llm \
  --checkpoint pretrained_models/CosyVoice2-0.5B/llm.pt \
  --model_dir exp/emotional_sft/llm \
  --tensorboard_dir tensorboard/emotional_sft/llm
```

### Method 2: Using Wrapper Script

Use the provided script for easier setup:

```bash
bash scripts/train_with_wandb.sh llm      # Train LLM
bash scripts/train_with_wandb.sh flow     # Train Flow
bash scripts/train_with_wandb.sh hifigan  # Train HiFiGAN
```

### Method 3: Environment Variables (Override Config)

You can override config settings with environment variables:

```bash
export WANDB_API_KEY="your-api-key"
export WANDB_PROJECT="your-entity/your-project"
export WANDB_RUN_NAME="custom-run-name-$(date +%Y%m%d-%H%M%S)"
export WANDB_MODE="online"  # or "offline" for no internet
export WANDB_DIR="./wandb_logs"

# Then run training normally
torchrun ... cosyvoice/bin/train.py ...
```

## Implementation Details

### Files Modified

1. **`cosyvoice/utils/train_utils.py`**:
   - Added `import wandb` with availability check
   - Added `init_wandb()` function to initialize W&B
   - Modified `log_per_step()` to log training metrics to W&B
   - Modified `log_per_save()` to log validation metrics to W&B

2. **`cosyvoice/bin/train.py`**:
   - Imported `init_wandb` function
   - Added `wandb_run = init_wandb(args, configs)` after TensorBoard init

3. **`conf/cosyvoice2_emotional_sft.yaml`**:
   - Added W&B configuration settings under `train_conf`

### Logged Metrics

**Training Metrics (per step)**:
- `TRAIN/epoch`: Current epoch number
- `TRAIN/lr`: Learning rate
- `TRAIN/grad_norm`: Gradient norm (for monitoring exploding/vanishing gradients)
- `TRAIN/loss`: Total loss
- `TRAIN/<component_loss>`: Individual loss components (e.g., `llm_loss`, `flow_loss`)
- `global_step`: Global training step

**Validation Metrics (per epoch)**:
- `CV/epoch`: Current epoch
- `CV/lr`: Learning rate at validation
- `CV/loss`: Total validation loss
- `CV/<component_loss>`: Individual validation loss components

## Dashboard Access

After training starts, you'll see:
```
W&B initialized: project=cosy-voxe, entity=rr-sjsu, run=emotional-sft-llm-20250315-120000
W&B dashboard: https://wandb.ai/rr-sjsu/cosy-voxe/runs/xxx
```

Visit the dashboard URL to monitor:
- Real-time training curves
- Loss comparisons across components
- System metrics (GPU usage, memory)
- Training duration and progress

## Troubleshooting

### W&B Not Initializing

**Issue**: Training runs but no W&B dashboard appears

**Solutions**:
1. Check wandb is installed: `pip install wandb`
2. Verify API key is correct in config or environment
3. Check internet connection (or set `WANDB_MODE=offline`)
4. Look for initialization message in logs:
   ```
   W&B initialized: project=...
   ```

### Offline Mode

If training without internet:

```bash
export WANDB_MODE="offline"
bash scripts/train_with_wandb.sh llm
```

Sync later when online:
```bash
wandb sync ./wandb_logs/wandb/run-xxx
```

### Multiple Runs with Same Name

W&B automatically resumes runs with the same name. To create a new run:
- Let script auto-generate name (includes timestamp)
- Or manually specify unique name:
  ```yaml
  wandb_run_name: "emotional-sft-llm-v2"
  ```

### Rank 0 Only Logging

W&B logging is automatically restricted to rank 0 (main process) in distributed training. Other ranks will skip W&B initialization to avoid duplicate runs.

## Integration with Existing Tools

### TensorBoard

W&B runs **alongside** TensorBoard (not replacing it):
- TensorBoard logs: `tensorboard/emotional_sft/{model}/`
- W&B logs: `./wandb_logs/`

Both receive the same metrics.

### Finding Best Checkpoint

Use `scripts/find_best_checkpoint.py` to identify best model by CV loss:

```bash
python scripts/find_best_checkpoint.py \
  --model_dir exp/emotional_sft/llm \
  --top_k 5
```

This reads TensorBoard logs and checkpoint files to rank by validation loss.

## Expected Training Monitoring

### LLM Training (50 epochs)
- **Duration**: 8-10 hours
- **Key Metrics**: `TRAIN/loss`, `CV/loss`, `TRAIN/lr`, `TRAIN/grad_norm`
- **What to Watch**:
  - Loss should decrease smoothly
  - CV loss should track training loss (gap indicates overfitting)
  - Grad norm should be stable (not exploding to >10 or vanishing to <0.01)

### Flow Training (50-100 epochs)
- **Duration**: 12-33 hours
- **Key Metrics**: `TRAIN/loss`, `CV/loss`
- **What to Watch**:
  - Slower convergence than LLM
  - May need more epochs if CV loss still decreasing

### HiFiGAN Training (200 epochs)
- **Duration**: 16-27 hours
- **Key Metrics**: `TRAIN/loss`, `TRAIN/d_loss` (discriminator), `CV/loss`
- **What to Watch**:
  - Generator and discriminator losses should balance
  - Discriminator loss ~0.5-1.0 (too low = overpowering generator)

## Advanced Configuration

### Custom Logging Frequency

Edit `conf/cosyvoice2_emotional_sft.yaml`:

```yaml
train_conf:
    log_interval: 50  # Log every 50 steps (default)
    save_per_step: -1  # -1 = save every epoch, >0 = save every N steps
```

### Tracking Additional Metrics

To add custom metrics, modify `log_per_step()` or `log_per_save()` in `cosyvoice/utils/train_utils.py`:

```python
# In log_per_step() or log_per_save():
if WANDB_AVAILABLE and wandb.run is not None and rank == 0:
    wandb_log = {
        # ... existing metrics ...
        'custom_metric': your_value,
    }
    wandb.log(wandb_log, step=step + 1)
```

## Benefits

1. **Remote Monitoring**: Check training progress from anywhere via web dashboard
2. **Comparison**: Compare different runs side-by-side
3. **System Metrics**: Automatic GPU/CPU/memory tracking
4. **Reproducibility**: All hyperparameters automatically logged
5. **Sharing**: Easy to share results with team via URL
6. **Alerts**: Set up email/Slack alerts for training completion or failures

## Notes

- W&B initialization is graceful: if unavailable or misconfigured, training continues with TensorBoard only
- API key in config is for convenience; remove before committing to public repos
- W&B run names include timestamp to avoid collisions
- All training config automatically logged to W&B for reproducibility
