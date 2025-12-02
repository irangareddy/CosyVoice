#!/bin/bash
# Emotional Speech SFT Training for CosyVoice2-0.5B Flow
# Dataset: ESD + RAVDESS (filtered, no CREMA-D)
# Training: Flow matching model fine-tuning

set -e

# Configuration
export PYTHONPATH=third_party/Matcha-TTS:$PWD
export CUDA_VISIBLE_DEVICES="0"
num_gpus=1
job_id=2025
dist_backend="nccl"
num_workers=4
prefetch=100
train_engine=torch_ddp

# Paths
pretrained_model_dir=/workspace/pretrained_models/CosyVoice2-0.5B
config=conf/cosyvoice2_emotional_sft.yaml
train_data=data/emotional_train.data.list
cv_data=data/emotional_val.data.list
model_dir=exp/cosyvoice2_emotional_sft/flow
tensorboard_dir=tensorboard/cosyvoice2_emotional_sft/flow

# Create output directories
mkdir -p $model_dir
mkdir -p $tensorboard_dir

echo "========================================="
echo "Training CosyVoice2-0.5B Flow for Emotional Speech"
echo "========================================="
echo "Dataset: ESD + RAVDESS (filtered)"
echo "Config: $config"
echo "Train data: $train_data ($(wc -l < $train_data 2>/dev/null || echo 'N/A') files)"
echo "Val data: $cv_data ($(wc -l < $cv_data 2>/dev/null || echo 'N/A') files)"
echo "Pretrained checkpoint: $pretrained_model_dir/flow.pt"
echo "Output directory: $model_dir"
echo "TensorBoard: $tensorboard_dir"
echo "========================================="

# Check if pretrained model exists
if [ ! -f "$pretrained_model_dir/flow.pt" ]; then
    echo "ERROR: Pretrained Flow model not found at $pretrained_model_dir/flow.pt"
    exit 1
fi

# Check if data files exist
if [ ! -f "$train_data" ]; then
    echo "ERROR: Training data list not found at $train_data"
    echo "Please run data preparation first"
    exit 1
fi

if [ ! -f "$cv_data" ]; then
    echo "ERROR: Validation data list not found at $cv_data"
    echo "Please run data preparation first"
    exit 1
fi

echo "Starting training..."
echo ""

# Train Flow model
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
  cosyvoice/bin/train.py \
  --train_engine $train_engine \
  --config $config \
  --train_data $train_data \
  --cv_data $cv_data \
  --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
  --model flow \
  --checkpoint $pretrained_model_dir/flow.pt \
  --model_dir $model_dir \
  --tensorboard_dir $tensorboard_dir \
  --ddp.dist_backend $dist_backend \
  --num_workers $num_workers \
  --prefetch $prefetch \
  --pin_memory \
  --use_amp 2>&1 | tee $model_dir/train.log

echo ""
echo "========================================="
echo "Training completed!"
echo "Checkpoints saved to: $model_dir"
echo "Training log: $model_dir/train.log"
echo "========================================="
