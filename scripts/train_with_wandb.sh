#!/bin/bash
# Training script with W&B integration
# Usage: bash scripts/train_with_wandb.sh [llm|flow|hifigan]

set -e

MODEL_TYPE=${1:-llm}
WANDB_API_KEY="fa1dd39746b8c27d133e3ac6d49ce74207706ff5"
WANDB_PROJECT="rr-sjsu/cosy-voxe"
WANDB_RUN_NAME="emotional-sft-${MODEL_TYPE}-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "Training CosyVoice2 ${MODEL_TYPE} with W&B"
echo "=========================================="
echo "W&B Project: ${WANDB_PROJECT}"
echo "W&B Run: ${WANDB_RUN_NAME}"
echo ""

# Set W&B environment variables
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"
export WANDB_MODE="online"  # Use "offline" if no internet
export WANDB_DIR="./wandb_logs"
export PYTHONPATH=$PWD:$PWD/third_party/Matcha-TTS

# Create wandb directory
mkdir -p ${WANDB_DIR}

# Start training with W&B
torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_id=100 --rdzv_backend='c10d' --rdzv_endpoint='localhost:29400' \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice2_emotional_sft.yaml \
  --train_data data/full/parquet/train/data.list \
  --cv_data data/full/parquet/val/data.list \
  --model ${MODEL_TYPE} \
  --checkpoint pretrained_models/CosyVoice2-0.5B/${MODEL_TYPE}.pt \
  --model_dir exp/emotional_sft/${MODEL_TYPE} \
  --tensorboard_dir tensorboard/emotional_sft/${MODEL_TYPE} \
  --num_workers 2 \
  --prefetch 50

echo ""
echo "=========================================="
echo "Training completed!"
echo "View logs: tensorboard --logdir tensorboard/emotional_sft/${MODEL_TYPE}"
echo "W&B dashboard: https://wandb.ai/${WANDB_PROJECT}"
echo "=========================================="
