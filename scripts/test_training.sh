#!/bin/bash
# Test training with small subset (5 epochs, ~2-3 minutes)
# Verifies training setup works before running full 5-6 hour training

set -e

echo "========================================="
echo "TEST: Training Pipeline (5 epochs)"
echo "========================================="
echo "This will train for 5 epochs (~2-3 min) to verify setup"
echo ""

# Configuration
export PYTHONPATH=third_party/Matcha-TTS:$PWD
export CUDA_VISIBLE_DEVICES="0"
num_gpus=1
job_id=9999
dist_backend="nccl"
num_workers=2
prefetch=50
train_engine=torch_ddp

# Paths
pretrained_model_dir=/workspace/pretrained_models/CosyVoice2-0.5B
config=conf/cosyvoice2_test.yaml
train_data=data/test_emotional_train.data.list
cv_data=data/test_emotional_val.data.list
model_dir=exp/test_cosyvoice2_emotional_sft/llm
tensorboard_dir=tensorboard/test_cosyvoice2_emotional_sft/llm

# Create output directories
mkdir -p $model_dir
mkdir -p $tensorboard_dir

echo "Checking prerequisites..."
if [ ! -f "$train_data" ]; then
    echo "ERROR: Test data not found at $train_data"
    echo "Please run: bash scripts/test_preprocessing.sh first"
    exit 1
fi

if [ ! -f "$pretrained_model_dir/llm.pt" ]; then
    echo "ERROR: Pretrained LLM model not found"
    exit 1
fi

echo "✓ Test data found: $(wc -l < $train_data) parquet files"
echo "✓ Pretrained model found"
echo ""

echo "Starting TEST training (5 epochs, ~2-3 minutes)..."
echo ""

# Train LLM only for 5 epochs
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  cosyvoice/bin/train.py \
  --train_engine $train_engine \
  --config $config \
  --train_data $train_data \
  --cv_data $cv_data \
  --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
  --model llm \
  --checkpoint $pretrained_model_dir/llm.pt \
  --model_dir $model_dir \
  --tensorboard_dir $tensorboard_dir \
  --ddp.dist_backend $dist_backend \
  --num_workers $num_workers \
  --prefetch $prefetch \
  --pin_memory \
  --use_amp 2>&1 | tee $model_dir/train.log

echo ""
echo "========================================="
echo "TEST TRAINING COMPLETED!"
echo "========================================="

# Check if training succeeded
if [ -f "$model_dir/epoch_5.pt" ]; then
    echo "✓ Training completed successfully!"
    echo "✓ Checkpoint saved: epoch_5.pt"
    echo ""
    echo "Checkpoints created:"
    ls -lh $model_dir/epoch_*.pt
    echo ""
    echo "Training log:"
    echo "  $model_dir/train.log"
    echo ""
    echo "Check loss values:"
    grep "Epoch" $model_dir/train.log | tail -10
    echo ""
    echo "========================================="
    echo "All systems verified! You can now run:"
    echo "  bash scripts/train_emotional_sft.sh"
    echo "========================================="
else
    echo "⚠ Training may have issues. Check logs:"
    tail -50 $model_dir/train.log
fi
