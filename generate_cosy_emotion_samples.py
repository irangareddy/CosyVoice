#!/usr/bin/env python3
"""
Generate Emotion Samples using CosyVoice2 with Fine-tuned HiFiGAN
Uses same texts and reference audios as F5TTS for fair model comparison.

Usage:
    python generate_cosy_emotion_samples.py [--checkpoint exp/emotional_hifigan/epoch_4_whole.pt]
"""

import sys
sys.path.insert(0, '/workspace/CosyVoice')
sys.path.insert(0, '/workspace/CosyVoice/third_party/Matcha-TTS')

import os
import re
import json
import csv
import time
import argparse
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Same emotion texts as F5TTS for fair comparison
EMOTION_SAMPLES = {
    "neutral": "The meeting starts at 10 am in Room 204.",
    "angry": "Seriously? You ignored the plan and broke everything.",
    "happy": "Yes! We nailed it—this feels amazing!",
    "sad": "I'm trying my best, but today feels heavy.",
    "surprise": "Whoa—wait, what? I didn't see that coming!",
}

# Same speakers as F5TTS (ESD dataset)
REFERENCE_SPEAKERS = {
    "female": "0019",
    "male": "0014",
}

# Emotion folder mapping
EMOTION_MAP = {
    "neutral": "Neutral",
    "angry": "Angry",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
}


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text for use in filename (same as F5TTS script)."""
    words = text.split()[:4]
    filename = "_".join(words).lower()
    filename = re.sub(r'[^a-z0-9_]', '', filename)
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename


def load_hifigan_checkpoint(cosyvoice: CosyVoice2, checkpoint_path: str):
    """Load custom HiFiGAN checkpoint into CosyVoice model."""
    print(f"Loading HiFiGAN checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=cosyvoice.model.device)

    # Filter only generator weights (exclude discriminator, epoch, step)
    hift_state_dict = {}
    for k, v in checkpoint.items():
        # Skip discriminator weights and metadata
        if k.startswith('discriminator.') or k in ['epoch', 'step']:
            continue
        # Remove 'generator.' prefix if present
        new_key = k.replace('generator.', '')
        hift_state_dict[new_key] = v

    # Load into model
    cosyvoice.model.hift.load_state_dict(hift_state_dict, strict=False)
    cosyvoice.model.hift.eval()

    print(f"✓ Custom HiFiGAN checkpoint loaded ({len(hift_state_dict)} parameters)")


def find_reference_audio(
    emotion: str,
    speaker_id: str,
    dataset_root: str = '/workspace/data/full/wav24k/esd'
) -> str:
    """Find first reference audio for emotion and speaker from ESD dataset."""
    emotion_key = EMOTION_MAP[emotion]
    audio_dir = Path(dataset_root) / speaker_id / emotion_key

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    return str(wav_files[0])


def generate_sample(
    cosyvoice: CosyVoice2,
    text: str,
    emotion: str,
    ref_audio_path: str,
    output_path: str
) -> Tuple[bool, str]:
    """Generate a single emotion sample using CosyVoice2.

    Returns:
        (success: bool, error_message: str)
    """
    try:
        # Load reference audio (automatically resampled to 16kHz)
        prompt_speech_16k = load_wav(ref_audio_path, 16000)

        # Create instruction text for emotional synthesis
        instruct_text = f"Speak with a {emotion} emotion, clear and expressive"

        # Generate audio using instruct mode
        for audio_data in cosyvoice.inference_instruct2(
            tts_text=text,
            instruct_text=instruct_text,
            prompt_speech_16k=prompt_speech_16k,
            stream=False,
            speed=1.0
        ):
            # Extract audio tensor [1, num_samples]
            output_audio = audio_data['tts_speech']

            # Save using soundfile
            sf.write(
                output_path,
                output_audio.squeeze(0).cpu().numpy(),  # [num_samples]
                cosyvoice.sample_rate
            )

            return True, ""

    except Exception as e:
        error_msg = str(e)
        return False, error_msg


def generate_emotion_samples(
    checkpoint_path: str,
    output_dir: str = '/workspace/CosyVoice/test_outputs/cosy_epoch4_samples',
    base_model_path: str = '/workspace/pretrained_models/CosyVoice2-0.5B'
) -> Dict[str, int]:
    """Generate all emotion samples for model comparison.

    Args:
        checkpoint_path: Path to HiFiGAN checkpoint
        output_dir: Output directory for samples
        base_model_path: Path to CosyVoice2 base model

    Returns:
        Statistics dict with total, success, failed counts
    """
    # Initialize statistics
    stats = {"total": 0, "success": 0, "failed": 0}
    total_samples = len(EMOTION_SAMPLES) * len(REFERENCE_SPEAKERS)
    start_time = time.time()

    # Metadata collection
    samples_metadata: List[Dict] = []

    # Output directory setup
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    error_log_file = output_base / "error_log.txt"

    print(f"\n{'='*70}")
    print(f"CosyVoice2 + HiFiGAN Emotion Sample Generation")
    print(f"{'='*70}")
    print(f"Base Model: {base_model_path}")
    print(f"HiFiGAN Checkpoint: {checkpoint_path}")
    print(f"Output Dir: {output_base}")
    print(f"Emotions: {len(EMOTION_SAMPLES)}")
    print(f"Speakers: {', '.join(REFERENCE_SPEAKERS.keys())}")
    print(f"Total Samples: {len(EMOTION_SAMPLES)} × {len(REFERENCE_SPEAKERS)} = {total_samples}")
    print(f"{'='*70}\n")

    # Load CosyVoice2 model
    print("Loading CosyVoice2 base model...")
    try:
        cosyvoice = CosyVoice2(
            base_model_path,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
        print(f"✓ CosyVoice2 loaded (sample rate: {cosyvoice.sample_rate} Hz)")
    except Exception as e:
        print(f"❌ Failed to load CosyVoice2: {e}")
        return stats

    # Load custom HiFiGAN checkpoint
    try:
        load_hifigan_checkpoint(cosyvoice, checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to load HiFiGAN checkpoint: {e}")
        return stats

    print()

    # Create progress bar
    progress_bar = tqdm(
        total=total_samples,
        desc="Generating",
        unit="sample",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ncols=100,
        colour="green"
    )

    try:
        for emotion_idx, (emotion, prompt) in enumerate(EMOTION_SAMPLES.items(), 1):
            emotion_status = f"[{emotion_idx}/{len(EMOTION_SAMPLES)}] {emotion.upper()}"

            for voice_type in REFERENCE_SPEAKERS.keys():
                speaker_id = REFERENCE_SPEAKERS[voice_type]
                current_task = f"{emotion_status} | {voice_type.capitalize()} ({speaker_id})"

                # Update progress bar description
                progress_bar.set_description(current_task)

                try:
                    # Find reference audio from ESD dataset
                    ref_audio_path = find_reference_audio(emotion, speaker_id)
                except FileNotFoundError as e:
                    progress_bar.write(f"  ❌ {e}")
                    stats["failed"] += 1
                    stats["total"] += 1
                    progress_bar.update(1)
                    continue

                # Create output directory structure
                output_emotion_dir = output_base / emotion / speaker_id
                output_emotion_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename
                text_snippet = sanitize_filename(prompt)
                filename = f"{speaker_id}_{emotion}_{text_snippet}.wav"
                output_file = output_emotion_dir / filename

                stats["total"] += 1

                # Generate audio
                success, error_msg = generate_sample(
                    cosyvoice=cosyvoice,
                    text=prompt,
                    emotion=emotion,
                    ref_audio_path=ref_audio_path,
                    output_path=str(output_file)
                )

                # Calculate relative file path
                file_path_rel = f"{emotion}/{speaker_id}/{filename}"

                # Collect metadata
                sample_meta = {
                    "emotion": emotion,
                    "speaker_id": speaker_id,
                    "speaker_type": voice_type.capitalize(),
                    "file_path": file_path_rel,
                    "file_name": filename,
                    "text": prompt,
                    "ref_audio": ref_audio_path,
                    "emotion_index": emotion_idx,
                    "speaker_index": list(REFERENCE_SPEAKERS.keys()).index(voice_type) + 1,
                    "success": success,
                }

                if success:
                    stats["success"] += 1
                    progress_bar.write(f"  ✅ {emotion.upper()} | {voice_type.capitalize()}: {filename}")
                else:
                    stats["failed"] += 1
                    progress_bar.write(f"  ❌ {emotion.upper()} | {voice_type.capitalize()}: Failed")

                    if error_msg:
                        # Show truncated error
                        error_first_line = error_msg.split('\n')[0]
                        if len(error_first_line) > 150:
                            error_first_line = error_first_line[:150] + "..."
                        progress_bar.write(f"      Error: {error_first_line}")

                        # Log full error to file
                        with open(error_log_file, "a", encoding="utf-8") as f:
                            f.write(f"\n{'='*70}\n")
                            f.write(f"Failed: {emotion.upper()} | {voice_type.capitalize()}\n")
                            f.write(f"Output: {output_file}\n")
                            f.write(f"Reference: {ref_audio_path}\n")
                            f.write(f"Error:\n{error_msg}\n")
                            f.write(f"{'='*70}\n")

                samples_metadata.append(sample_meta)

                # Update progress
                progress_bar.update(1)

                # Update postfix with stats
                elapsed = time.time() - start_time
                success_rate = (100 * stats['success'] / stats['total']) if stats['total'] > 0 else 0
                progress_bar.set_postfix({
                    "✅": stats['success'],
                    "❌": stats['failed'],
                    "Rate": f"{success_rate:.1f}%"
                })

    finally:
        progress_bar.close()

    # Generate CSV metadata file
    csv_file = output_base / "text.csv"
    if samples_metadata:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "emotion", "speaker_id", "speaker_type", "file_path",
                "text", "ref_audio"
            ])
            writer.writeheader()
            for meta in samples_metadata:
                writer.writerow({
                    "emotion": meta["emotion"],
                    "speaker_id": meta["speaker_id"],
                    "speaker_type": meta["speaker_type"],
                    "file_path": meta["file_path"],
                    "text": meta["text"],
                    "ref_audio": meta["ref_audio"],
                })

    # Generate JSON metadata file (only successful samples)
    json_file = output_base / "samples.json"
    successful_samples = [meta for meta in samples_metadata if meta.get("success", False)]
    if successful_samples:
        json_data = []
        for meta in successful_samples:
            json_entry = {k: v for k, v in meta.items() if k != "success"}
            json_data.append(json_entry)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Print summary
    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Generation Complete!")
    print(f"{'='*70}")
    print(f"Total:    {stats['total']}")
    print(f"✅ Success: {stats['success']}")
    print(f"❌ Failed:  {stats['failed']}")
    if stats['total'] > 0:
        print(f"Rate:     {100 * stats['success'] / stats['total']:.1f}%")
    print(f"Time:     {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Output:   {output_base}")
    if stats['failed'] > 0:
        print(f"\n⚠️  Full error details saved to: {error_log_file}")
    if samples_metadata:
        print(f"\n📄 Documentation files:")
        print(f"   CSV:  {csv_file}")
        print(f"   JSON: {json_file}")
    print(f"{'='*70}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate emotion samples using CosyVoice2 with fine-tuned HiFiGAN"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/CosyVoice/exp/emotional_hifigan/epoch_4_whole.pt",
        help="Path to HiFiGAN checkpoint (default: epoch 4)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/CosyVoice/test_outputs/cosy_epoch4_samples",
        help="Output directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/workspace/pretrained_models/CosyVoice2-0.5B",
        help="Path to CosyVoice2 base model"
    )

    args = parser.parse_args()

    try:
        stats = generate_emotion_samples(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            base_model_path=args.base_model
        )
        sys.exit(0 if stats["failed"] == 0 else 1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
