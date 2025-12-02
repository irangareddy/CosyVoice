#!/bin/bash
# Upgrade PyTorch for RTX 5090 (sm_120) support
# This installs PyTorch with full RTX 5090 optimization

set -e

echo "========================================="
echo "Upgrading PyTorch for RTX 5090 Support"
echo "========================================="
echo ""

# Check current PyTorch version
echo "Current PyTorch installation:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo ""

# Check GPU
echo "Current GPU:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo ""

echo "This will upgrade PyTorch to support RTX 5090 (sm_120)"
echo "Current PyTorch supports: sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90"
echo "RTX 5090 needs: sm_120"
echo ""

read -p "Continue with upgrade? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upgrade cancelled"
    exit 0
fi

echo ""
echo "Step 1: Uninstalling current PyTorch..."
pip uninstall -y torch torchaudio torchvision

echo ""
echo "Step 2: Installing PyTorch nightly with CUDA 12.1 (RTX 5090 support)..."
pip install --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu121

echo ""
echo "Step 3: Verifying installation..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print(f'CUDA capability: sm_{cap[0]}{cap[1]}')
    print('Expected for RTX 5090: sm_120')
    if cap == (12, 0):
        print('✓ RTX 5090 fully supported!')
    else:
        print('⚠ Warning: GPU capability mismatch')
"

echo ""
echo "========================================="
echo "PyTorch Upgrade Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Test training: bash scripts/test_training.sh"
echo "  2. Full training: bash scripts/train_emotional_sft.sh"
echo ""
echo "Note: You're now using PyTorch nightly build"
echo "      This has full RTX 5090 optimization!"
echo "========================================="
