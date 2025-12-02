#!/bin/bash
# Docker entrypoint script for CosyVoice training
# Ensures Matcha-TTS submodule is initialized before running commands

set -e

echo "========================================="
echo "CosyVoice Training Container"
echo "========================================="

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate cosyvoice

# Check and initialize Matcha-TTS submodule if needed
echo "Checking Matcha-TTS submodule..."
if [ ! -f "third_party/Matcha-TTS/matcha/utils/audio.py" ]; then
    echo "Matcha-TTS submodule not found. Initializing..."
    git submodule update --init --recursive
    if [ -f "third_party/Matcha-TTS/matcha/utils/audio.py" ]; then
        echo "✓ Matcha-TTS submodule initialized successfully"
    else
        echo "ERROR: Failed to initialize Matcha-TTS submodule!"
        exit 1
    fi
else
    echo "✓ Matcha-TTS submodule already initialized"
fi

# Verify PyTorch and CUDA
echo ""
echo "Environment Info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
echo ""
echo "Ready for training!"
echo "========================================="
echo ""

# Execute the command passed to docker run
exec "$@"
