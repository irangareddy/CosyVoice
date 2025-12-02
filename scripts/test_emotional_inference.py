#!/usr/bin/env python3
"""
Test emotional voice generation with fine-tuned CosyVoice2-0.5B.
Generates audio samples for all 5 emotions using the same text.
"""
import sys
import os

# Add CosyVoice to path
sys.path.insert(0, '/workspace/CosyVoice')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


def main():
    # Configuration
    model_dir = 'exp/cosyvoice2_emotional_final'
    output_dir = 'exp/test_outputs'

    print("="*60)
    print("Emotional Voice Generation Test")
    print("="*60)
    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print("")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    try:
        cosyvoice = CosyVoice2(
            model_dir,
            load_jit=False,  # Use PyTorch models
            load_trt=False,  # No TensorRT
            load_vllm=False,  # No vLLM
            fp16=True  # Use FP16 for faster inference
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nMake sure you have:")
        print("  1. Completed training")
        print("  2. Assembled final model: bash scripts/assemble_model.sh")
        return

    # Test parameters
    emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
    test_text = "Hello, this is a test of emotional voice synthesis."

    # Reference speaker audio (use a neutral sample from validation set)
    reference_wav = '/workspace/data/full/wav24k/esd/0011/Neutral/0011_000001.wav'

    if not os.path.exists(reference_wav):
        print(f"Warning: Reference audio not found at {reference_wav}")
        print("Using alternative reference...")
        # Try to find any ESD neutral audio
        import glob
        candidates = glob.glob('/workspace/data/full/wav24k/esd/*/Neutral/*.wav')
        if candidates:
            reference_wav = candidates[0]
            print(f"Using: {reference_wav}")
        else:
            print("Error: No reference audio found")
            return

    print(f"\nReference audio: {reference_wav}")
    reference_audio = load_wav(reference_wav, 16000)

    print("\n" + "="*60)
    print("Generating Emotional Speech Samples")
    print("="*60)

    # Generate audio for each emotion
    for emotion in emotions:
        # Prepend emotion prefix (same as training)
        emotional_text = f"<{emotion}> {test_text}"

        print(f"\n[{emotion.upper()}]")
        print(f"  Text: {emotional_text}")

        try:
            # Generate audio (zero-shot with emotion prefix)
            for i, output in enumerate(cosyvoice.inference_zero_shot(
                tts_text=emotional_text,
                prompt_text="<neutral> Hello.",  # Neutral reference prompt
                prompt_speech_16k=reference_audio,
                stream=False
            )):
                output_path = f'{output_dir}/{emotion}_output.wav'
                torchaudio.save(output_path, output['tts_speech'], cosyvoice.sample_rate)
                print(f"  ✓ Saved: {output_path}")
                break  # Only take first output
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*60)
    print("Inference Test Completed!")
    print("="*60)
    print(f"Audio files saved to: {output_dir}")
    print("\nListen to the outputs to verify emotional variation:")
    for emotion in emotions:
        print(f"  - {emotion}_output.wav")
    print("="*60)


if __name__ == '__main__':
    main()
