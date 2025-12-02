#!/bin/bash
# Fix RTX 5090 PyTorch compatibility and verify Matcha-TTS submodule
# Run this inside the Docker container

set -e

echo "========================================="
echo "RTX 5090 PyTorch Fix Script"
echo "========================================="
echo ""

# Step 1: Check current PyTorch version
echo "[Step 1/5] Checking current PyTorch and CUDA versions..."
python -c "import torch; print(f'Current PyTorch: {torch.__version__}'); print(f'Current CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
echo ""

# Step 2: Verify Matcha-TTS submodule
echo "[Step 2/5] Verifying Matcha-TTS submodule..."
if [ -f "third_party/Matcha-TTS/matcha/utils/audio.py" ]; then
    echo "✓ Matcha-TTS submodule exists and populated"
    ls -lh third_party/Matcha-TTS/matcha/utils/audio.py
else
    echo "✗ Matcha-TTS submodule missing - initializing..."
    git submodule update --init --recursive
    echo "✓ Matcha-TTS submodule initialized"
fi
echo ""

# Step 3: Backup current PyTorch version
echo "[Step 3/5] Creating backup of requirements..."
pip freeze > /tmp/requirements_backup_$(date +%Y%m%d_%H%M%S).txt
echo "✓ Backup saved to /tmp/requirements_backup_*.txt"
echo ""

# Step 4: Upgrade PyTorch for RTX 5090
echo "[Step 4/5] Upgrading PyTorch to support RTX 5090 (CUDA 12.6)..."
echo "This will take a few minutes..."
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
echo ""

# Step 5: Verify new PyTorch installation
echo "[Step 5/5] Verifying upgraded PyTorch..."
python -c "import torch; print(f'New PyTorch: {torch.__version__}'); print(f'New CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "========================================="
echo "Upgrade Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Test if GPU is detected: python -c 'import torch; print(torch.cuda.get_device_name(0))'"
echo "2. Run training test: bash scripts/test_training.sh"
echo ""
