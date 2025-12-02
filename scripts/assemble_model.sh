#!/bin/bash
# Assemble Fine-Tuned Model for Inference
# Combines fine-tuned LLM with pretrained Flow and HiFT

set -e

echo "========================================="
echo "Assembling Fine-Tuned Model"
echo "========================================="

# Configuration
pretrained_dir=/workspace/pretrained_models/CosyVoice2-0.5B
llm_checkpoint=exp/cosyvoice2_emotional_sft/llm/llm.pt
output_dir=exp/cosyvoice2_emotional_final

# Check if fine-tuned LLM exists
if [ ! -f "$llm_checkpoint" ]; then
    echo "ERROR: Fine-tuned LLM not found at $llm_checkpoint"
    echo "Please run model averaging first:"
    echo "  python cosyvoice/bin/average_model.py --dst_model $llm_checkpoint --src_path exp/cosyvoice2_emotional_sft/llm --num 5 --val_best"
    exit 1
fi

# Create output directory
mkdir -p $output_dir

echo ""
echo "Copying fine-tuned LLM..."
cp $llm_checkpoint $output_dir/
echo "  ✓ llm.pt"

echo ""
echo "Copying pretrained Flow and HiFT (not fine-tuned)..."
cp $pretrained_dir/flow.pt $output_dir/
echo "  ✓ flow.pt"
cp $pretrained_dir/hift.pt $output_dir/
echo "  ✓ hift.pt"

echo ""
echo "Copying tokenizer and ONNX models..."
cp -r $pretrained_dir/CosyVoice-BlankEN $output_dir/
echo "  ✓ CosyVoice-BlankEN/"
cp $pretrained_dir/*.onnx $output_dir/ 2>/dev/null || echo "  (no ONNX files)"
cp $pretrained_dir/*.yaml $output_dir/ 2>/dev/null || echo "  (no YAML files)"

echo ""
echo "========================================="
echo "Model Assembly Completed!"
echo "========================================="
echo "Final model directory: $output_dir"
echo ""
echo "Contents:"
ls -lh $output_dir/*.pt 2>/dev/null || true
echo ""
echo "You can now run inference with:"
echo "  python scripts/test_emotional_inference.py"
echo "========================================="
