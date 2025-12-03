#!/bin/bash
# Emotional Speech HiFiGAN Training with W&B
# Dataset: ESD + RAVDESS (filtered, no CREMA-D)
# Training: HiFiGAN vocoder (mel-spectrogram → waveform)
# Strategy: GAN training for emotional audio quality

set -e

# W&B Configuration
WANDB_API_KEY="fa1dd39746b8c27d133e3ac6d49ce74207706ff5"
WANDB_ENTITY="rr-sjsu"
WANDB_PROJECT="cosy-voxe"
WANDB_RUN_NAME="emotional-hifigan-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "HiFiGAN Training for Emotional TTS"
echo "=========================================="
echo "Model: CosyVoice2 HiFiGAN Vocoder (HiFT)"
echo "Task: Mel-spectrogram → High-quality waveform"
echo "W&B Entity: ${WANDB_ENTITY}"
echo "W&B Project: ${WANDB_PROJECT}"
echo "W&B Run: ${WANDB_RUN_NAME}"
echo ""

# Set W&B environment variables
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"
export WANDB_MODE="online"
export WANDB_DIR="./wandb_logs"
export PYTHONPATH=$PWD:$PWD/third_party/Matcha-TTS

# Create directories
mkdir -p ${WANDB_DIR}
mkdir -p exp/emotional_hifigan
mkdir -p tensorboard/emotional_hifigan

# Verify pretrained checkpoint exists
if [ ! -f "/workspace/pretrained_models/CosyVoice2-0.5B/hift.pt" ]; then
    echo "ERROR: Pretrained HiFiGAN checkpoint not found!"
    echo "Expected: /workspace/pretrained_models/CosyVoice2-0.5B/hift.pt"
    exit 1
fi

# Verify data files exist
if [ ! -f "/workspace/data/full/parquet/train/data.list" ]; then
    echo "ERROR: Training data not found!"
    echo "Expected: /workspace/data/full/parquet/train/data.list"
    exit 1
fi

if [ ! -f "/workspace/data/full/parquet/val/data.list" ]; then
    echo "ERROR: Validation data not found!"
    echo "Expected: /workspace/data/full/parquet/val/data.list"
    exit 1
fi

echo "Starting HiFiGAN GAN training..."
echo ""
echo "IMPORTANT NOTES:"
echo "- GAN training alternates generator and discriminator updates"
echo "- Expect loss oscillations (this is normal for GAN)"
echo "- Monitor: loss_gen, loss_disc, loss_mel, loss_f0"
echo "- Training time: ~4-5 hours (30 epochs)"
echo ""

# Train HiFiGAN vocoder
torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_id=102 --rdzv_backend='c10d' --rdzv_endpoint='localhost:29402' \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice2_emotional_hifigan.yaml \
  --train_data /workspace/data/full/parquet/train/data.list \
  --cv_data /workspace/data/full/parquet/val/data.list \
  --model hifigan \
  --checkpoint /workspace/pretrained_models/CosyVoice2-0.5B/hift.pt \
  --model_dir exp/emotional_hifigan \
  --tensorboard_dir tensorboard/emotional_hifigan \
  --qwen_pretrain_path /workspace/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --num_workers 0 \
  --use_amp

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Checkpoints: exp/emotional_hifigan/"
echo "TensorBoard: tensorboard --logdir tensorboard/emotional_hifigan"
echo "W&B dashboard: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "=========================================="
