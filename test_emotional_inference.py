#!/usr/bin/env python3
"""
Test emotional TTS inference with fine-tuned LLM checkpoint.

This script demonstrates how to use the test-trained LLM checkpoint
with pretrained Flow/HiFiGAN for emotional speech synthesis.

Usage:
    python test_emotional_inference.py
"""

import sys
import os
from pathlib import Path
import shutil

# Add Matcha-TTS to path
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

def setup_test_model_dir():
    """
    Create a temporary model directory combining:
    - Fine-tuned LLM from test training
    - Pretrained Flow and HiFiGAN
    """
    test_llm_dir = Path('exp/test_run/llm')
    pretrained_dir = Path('pretrained_models/CosyVoice2-0.5B')
    temp_model_dir = Path('pretrained_models/test_emotional_model')

    # Check if test LLM exists
    if not test_llm_dir.exists():
        print(f"❌ Error: Test LLM directory not found: {test_llm_dir}")
        print("   Run test training first: bash scripts/test_training.sh llm")
        return None

    # Find best LLM checkpoint (last epoch)
    llm_checkpoints = sorted(test_llm_dir.glob('epoch_*_whole.pt'))
    if not llm_checkpoints:
        print(f"❌ Error: No LLM checkpoints found in {test_llm_dir}")
        return None

    best_llm_ckpt = llm_checkpoints[-1]  # Use last epoch
    print(f"Using LLM checkpoint: {best_llm_ckpt}")

    # Create temp model directory structure
    temp_model_dir.mkdir(parents=True, exist_ok=True)

    # Copy fine-tuned LLM
    print(f"Copying fine-tuned LLM to {temp_model_dir}/llm.pt")
    shutil.copy2(best_llm_ckpt, temp_model_dir / 'llm.pt')

    # Copy pretrained Flow and HiFiGAN
    print(f"Copying pretrained Flow from {pretrained_dir}/flow.pt")
    shutil.copy2(pretrained_dir / 'flow.pt', temp_model_dir / 'flow.pt')

    print(f"Copying pretrained HiFiGAN from {pretrained_dir}/hift.pt")
    shutil.copy2(pretrained_dir / 'hift.pt', temp_model_dir / 'hift.pt')

    # Copy other necessary files
    for file in ['speech_tokenizer_v1.onnx', 'campplus.onnx']:
        src = pretrained_dir / file
        if src.exists():
            shutil.copy2(src, temp_model_dir / file)

    # Copy CosyVoice-BlankEN directory (tokenizer)
    blank_en_src = pretrained_dir / 'CosyVoice-BlankEN'
    blank_en_dst = temp_model_dir / 'CosyVoice-BlankEN'
    if blank_en_src.exists():
        if blank_en_dst.exists():
            shutil.rmtree(blank_en_dst)
        shutil.copytree(blank_en_src, blank_en_dst)

    print(f"✅ Test model directory created: {temp_model_dir}")
    return temp_model_dir

def test_emotional_inference():
    """Run emotional TTS inference test."""

    print("=" * 60)
    print("Testing Emotional TTS Inference")
    print("=" * 60)
    print()

    # Setup model directory
    model_dir = setup_test_model_dir()
    if model_dir is None:
        return

    print()
    print("Loading model (this may take a minute)...")

    # Load model with fine-tuned LLM + pretrained Flow/HiFiGAN
    try:
        cosyvoice = CosyVoice2(
            str(model_dir),
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print()
    print("=" * 60)
    print("Running Inference Tests")
    print("=" * 60)
    print()

    # Create output directory
    output_dir = Path('generated_audio/test_emotional')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test sentences with different emotions
    test_cases = [
        {
            'emotion': 'happy',
            'text': 'I am so excited to see you today!',
            'prompt_text': 'This makes me really happy.',
            'reference': 'Find a happy audio sample from your dataset'
        },
        {
            'emotion': 'sad',
            'text': 'I feel really disappointed about this.',
            'prompt_text': 'This makes me so sad.',
            'reference': 'Find a sad audio sample from your dataset'
        },
        {
            'emotion': 'angry',
            'text': 'This is absolutely unacceptable!',
            'prompt_text': 'I am very angry about this.',
            'reference': 'Find an angry audio sample from your dataset'
        }
    ]

    print("⚠️  NOTE: You need to provide reference audio for each emotion")
    print("    Reference audio should be from your ESD/RAVDESS dataset")
    print()
    print("Example reference audio paths:")
    print("  Happy: data/raw/esd/0011/Angry/train/0011_000001.wav")
    print("  Sad: data/raw/esd/0011/Sad/train/0011_000011.wav")
    print()

    # For demonstration, we'll just show the code pattern
    print("Inference code pattern:")
    print("-" * 60)
    print("""
# Load emotional reference audio
prompt_speech_16k = load_wav('path/to/happy_sample.wav', 16000)

# Generate emotional speech
for i, output in enumerate(cosyvoice.inference_zero_shot(
    text='I am so excited to see you today!',
    prompt_text='This makes me really happy.',
    prompt_speech_16k=prompt_speech_16k,
    stream=False
)):
    output_path = 'generated_audio/test_emotional/happy_test.wav'
    torchaudio.save(
        output_path,
        output['tts_speech'],
        cosyvoice.sample_rate
    )
    print(f'✅ Generated: {output_path}')
    """)
    print("-" * 60)
    print()

    print("To actually run inference:")
    print("1. Find reference audio samples from your dataset:")
    print("   - data/raw/esd/[speaker]/[emotion]/train/[file].wav")
    print()
    print("2. Modify this script to add actual reference paths")
    print()
    print("3. Run inference and listen to the results")
    print()

    print("=" * 60)
    print("Important Notes")
    print("=" * 60)
    print()
    print("⚠️  Test checkpoint quality:")
    print("   - Only trained on ~1000 samples for 5 epochs")
    print("   - May not have strong emotional transfer yet")
    print("   - Quality will improve after full training (50 epochs, 15k samples)")
    print()
    print("✅ For production-quality results:")
    print("   1. Run full LLM training: bash scripts/train_with_wandb.sh llm")
    print("   2. Wait 8-10 hours for 50 epochs")
    print("   3. Use best checkpoint for inference")
    print()
    print("🚀 Current test checkpoint is just for verification!")
    print()

if __name__ == '__main__':
    test_emotional_inference()
