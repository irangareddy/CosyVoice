#!/bin/bash
# Emotional Speech Flow Matching Training with W&B
# Dataset: ESD + RAVDESS (filtered, no CREMA-D)
# Training: Flow model only (token → mel-spectrogram)
# Strategy: Optimized for emotional prosody learning

set -e

# W&B Configuration
WANDB_API_KEY="fa1dd39746b8c27d133e3ac6d49ce74207706ff5"
WANDB_ENTITY="rr-sjsu"
WANDB_PROJECT="cosy-voxe"
WANDB_RUN_NAME="emotional-flow-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "Flow Matching Training for Emotional TTS"
echo "=========================================="
echo "Model: CosyVoice2-0.5B Flow (Causal Masked Diffusion)"
echo "Task: Speech token → Mel-spectrogram with emotion"
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
mkdir -p exp/emotional_flow
mkdir -p tensorboard/emotional_flow

# Verify pretrained checkpoint exists
if [ ! -f "/workspace/pretrained_models/CosyVoice2-0.5B/flow.pt" ]; then
    echo "ERROR: Pretrained Flow checkpoint not found!"
    echo "Expected: /workspace/pretrained_models/CosyVoice2-0.5B/flow.pt"
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

echo "Starting Flow Matching training..."
echo ""

# Train Flow model
torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_id=100 --rdzv_backend='c10d' --rdzv_endpoint='localhost:29401' \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice2_emotional_flow.yaml \
  --train_data /workspace/data/full/parquet/train/data.list \
  --cv_data /workspace/data/full/parquet/val/data.list \
  --model flow \
  --checkpoint /workspace/pretrained_models/CosyVoice2-0.5B/flow.pt \
  --model_dir exp/emotional_flow \
  --tensorboard_dir tensorboard/emotional_flow \
  --qwen_pretrain_path /workspace/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --num_workers 2 \
  --prefetch 50 \
  --use_amp

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Checkpoints: exp/emotional_flow/"
echo "TensorBoard: tensorboard --logdir tensorboard/emotional_flow"
echo "W&B dashboard: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "=========================================="
