#!/bin/bash
# Quick test training script - 1 epoch with limited data
# Verifies all three models (LLM, Flow, HiFiGAN) work before full training
# Usage: bash scripts/test_training.sh [llm|flow|hifigan|all]

set -e

MODEL_TYPE=${1:-llm}

echo "=========================================="
echo "TEST RUN: Training ${MODEL_TYPE} for 1 epoch"
echo "=========================================="
echo "This will train for 1 epoch (~5-10 min) to verify setup"
echo ""

# Configuration
export PYTHONPATH=$PWD:$PWD/third_party/Matcha-TTS
export WANDB_MODE="offline"  # Offline mode for test run
export WANDB_DIR="./wandb_logs"

# Create small test dataset (first parquet file only)
echo "Step 1: Creating small test dataset..."
mkdir -p data/test_run
head -n 1 data/full/parquet/train/data.list > data/test_run/train_mini.list
head -n 1 data/full/parquet/val/data.list > data/test_run/val_mini.list

echo "✓ Test datasets created:"
echo "  Train: 1 parquet file (~1000 samples)"
echo "  Val: 1 parquet file (~100 samples)"
echo ""

# Create test config if it doesn't exist
if [ ! -f "conf/cosyvoice2_test.yaml" ]; then
    echo "Step 2: Creating test config..."
    cp conf/cosyvoice2_emotional_sft.yaml conf/cosyvoice2_test.yaml
    # Modify max_epoch to 1 for quick test
    sed -i 's/max_epoch: 50/max_epoch: 1/g' conf/cosyvoice2_test.yaml
    sed -i 's/max_epoch: 200/max_epoch: 1/g' conf/cosyvoice2_test.yaml
    # Disable W&B for test
    sed -i 's/wandb_project:.*/wandb_project: ""/' conf/cosyvoice2_test.yaml
    echo "✓ Test config created (1 epoch only)"
else
    echo "Step 2: Using existing test config"
fi
echo ""

# Function to test a single model
test_model() {
    local model=$1
    echo "=========================================="
    echo "Testing ${model} model..."
    echo "=========================================="

    torchrun --nnodes=1 --nproc_per_node=1 \
      --rdzv_id=999 --rdzv_backend='c10d' --rdzv_endpoint='localhost:29401' \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --config conf/cosyvoice2_test.yaml \
      --train_data data/test_run/train_mini.list \
      --cv_data data/test_run/val_mini.list \
      --model ${model} \
      --checkpoint pretrained_models/CosyVoice2-0.5B/${model}.pt \
      --model_dir exp/test_run/${model} \
      --tensorboard_dir tensorboard/test_run/${model} \
      --num_workers 2 \
      --prefetch 50

    # Check if epoch completed
    if [ -f "exp/test_run/${model}/epoch_0_whole.pt" ] || [ -f "exp/test_run/${model}/epoch_1_whole.pt" ]; then
        echo ""
        echo "✅ ${model} test PASSED"
        echo "   Checkpoint: exp/test_run/${model}/"
        return 0
    else
        echo ""
        echo "❌ ${model} test FAILED - no checkpoint created"
        return 1
    fi
}

# Run tests
if [ "$MODEL_TYPE" == "all" ]; then
    echo "Testing ALL models sequentially..."
    echo ""
    test_model "llm" && LLM_OK=1 || LLM_OK=0
    echo ""
    test_model "flow" && FLOW_OK=1 || FLOW_OK=0
    echo ""
    test_model "hifigan" && HIFIGAN_OK=1 || HIFIGAN_OK=0

    echo ""
    echo "=========================================="
    echo "TEST RESULTS SUMMARY"
    echo "=========================================="
    [ $LLM_OK -eq 1 ] && echo "✅ LLM: PASSED" || echo "❌ LLM: FAILED"
    [ $FLOW_OK -eq 1 ] && echo "✅ Flow: PASSED" || echo "❌ Flow: FAILED"
    [ $HIFIGAN_OK -eq 1 ] && echo "✅ HiFiGAN: PASSED" || echo "❌ HiFiGAN: FAILED"
    echo ""

    if [ $LLM_OK -eq 1 ] && [ $FLOW_OK -eq 1 ] && [ $HIFIGAN_OK -eq 1 ]; then
        echo "🎉 ALL TESTS PASSED! Ready for full training."
        echo ""
        echo "Start full training with:"
        echo "  bash scripts/train_with_wandb.sh llm      # Train LLM"
        echo "  bash scripts/train_with_wandb.sh flow     # Train Flow"
        echo "  bash scripts/train_with_wandb.sh hifigan  # Train HiFiGAN"
    else
        echo "⚠️  Some tests failed. Check logs in exp/test_run/*/"
    fi
else
    test_model "$MODEL_TYPE"
    echo ""
    echo "=========================================="
    echo "Single model test complete!"
    echo "=========================================="
    echo ""
    echo "To test all models:"
    echo "  bash scripts/test_training.sh all"
    echo ""
    echo "To start full training:"
    echo "  bash scripts/train_with_wandb.sh ${MODEL_TYPE}"
fi

echo "=========================================="
