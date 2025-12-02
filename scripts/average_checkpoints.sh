#!/bin/bash
# Average Best Checkpoints for Final Model
# Averages top 5 checkpoints by validation loss

set -e

echo "========================================="
echo "Averaging Model Checkpoints"
echo "========================================="

# Configuration
checkpoint_dir=exp/cosyvoice2_emotional_sft/llm
output_model=$checkpoint_dir/llm.pt
num_checkpoints=5

# Check if checkpoint directory exists
if [ ! -d "$checkpoint_dir" ]; then
    echo "ERROR: Checkpoint directory not found: $checkpoint_dir"
    echo "Please complete training first"
    exit 1
fi

# Count available checkpoints
num_available=$(ls $checkpoint_dir/epoch-*.pt 2>/dev/null | wc -l)
echo "Available checkpoints: $num_available"

if [ $num_available -eq 0 ]; then
    echo "ERROR: No epoch checkpoints found in $checkpoint_dir"
    echo "Training may not have completed successfully"
    exit 1
fi

if [ $num_available -lt $num_checkpoints ]; then
    echo "WARNING: Only $num_available checkpoints available, using all of them"
    num_checkpoints=$num_available
fi

echo "Averaging top $num_checkpoints checkpoints by validation loss..."
echo ""

python cosyvoice/bin/average_model.py \
    --dst_model $output_model \
    --src_path $checkpoint_dir \
    --num $num_checkpoints \
    --val_best

if [ -f "$output_model" ]; then
    echo ""
    echo "========================================="
    echo "Model Averaging Completed!"
    echo "========================================="
    echo "Averaged model saved to: $output_model"
    echo ""
    echo "Next steps:"
    echo "  1. Assemble final model: bash scripts/assemble_model.sh"
    echo "  2. Test inference: python scripts/test_emotional_inference.py"
    echo "========================================="
else
    echo "ERROR: Model averaging failed"
    exit 1
fi
