#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Split Audit Script for CosyVoice Emotional TTS
Analyzes train/val distribution to detect potential issues
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def load_json_files(directory, pattern):
    """Load and merge all JSON files matching pattern"""
    data = {}
    dir_path = Path(directory)
    for json_file in sorted(dir_path.glob(pattern)):
        with open(json_file, 'r') as f:
            data.update(json.load(f))
    return data

def analyze_parquet_list(data_list_path):
    """Count total utterances from data.list"""
    with open(data_list_path, 'r') as f:
        return len(f.readlines())

def main():
    print("="*60)
    print("CosyVoice Data Split Audit")
    print("="*60)
    print()

    # Paths
    train_dir = "data/full/parquet/train"
    val_dir = "data/full/parquet/val"

    # Load speaker mappings
    print("📂 Loading speaker data...")
    try:
        train_spk = load_json_files(train_dir, "spk2parquet_*.json")
        val_spk = load_json_files(val_dir, "spk2parquet_*.json")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find speaker mapping files")
        print(f"   {e}")
        sys.exit(1)

    # Load utterance counts
    try:
        train_utt_count = analyze_parquet_list(f"{train_dir}/data.list")
        val_utt_count = analyze_parquet_list(f"{val_dir}/data.list")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find data.list files")
        print(f"   {e}")
        sys.exit(1)

    # Analyze speakers
    train_speakers = set(train_spk.keys())
    val_speakers = set(val_spk.keys())
    overlap_speakers = train_speakers & val_speakers

    print("="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print()
    print(f"Train utterances: {train_utt_count:,}")
    print(f"Val utterances:   {val_utt_count:,}")
    print(f"Total utterances: {train_utt_count + val_utt_count:,}")
    print(f"Val ratio:        {val_utt_count/(train_utt_count + val_utt_count)*100:.1f}%")
    print()

    print("="*60)
    print("👥 SPEAKER DISTRIBUTION")
    print("="*60)
    print()
    print(f"Train speakers: {len(train_speakers)}")
    print(f"Val speakers:   {len(val_speakers)}")
    print(f"Overlap:        {len(overlap_speakers)}")
    print()

    # Check for data leakage or distribution issues
    print("="*60)
    print("🔍 DISTRIBUTION ANALYSIS")
    print("="*60)
    print()

    if len(overlap_speakers) == 0:
        print("✅ ZERO-SHOT SETUP: No speaker overlap (good for generalization)")
        print("   → Model must learn emotion independent of speaker identity")
    elif len(overlap_speakers) == len(train_speakers) == len(val_speakers):
        print("✅ SEEN-SPEAKER SETUP: All speakers in both sets")
        print("   → Model learns speaker-specific emotional patterns")
    else:
        print("⚠️  MIXED SETUP: Partial speaker overlap")
        print(f"   → {len(overlap_speakers)} speakers in both")
        print(f"   → {len(train_speakers - val_speakers)} speakers only in train")
        print(f"   → {len(val_speakers - train_speakers)} speakers only in val")
        print("   This might cause validation metrics to be unstable!")

    print()

    # Check validation set size
    if val_utt_count < 100:
        print("⚠️  WARNING: Validation set is very small (<100 utterances)")
        print("   → Metrics may be noisy and unreliable")
    elif val_utt_count < 500:
        print("⚠️  WARNING: Validation set is small (<500 utterances)")
        print("   → Consider increasing validation set size")
    else:
        print(f"✅ Validation set size is reasonable ({val_utt_count} utterances)")

    print()

    # Check train/val ratio
    val_ratio = val_utt_count / (train_utt_count + val_utt_count)
    if val_ratio < 0.05:
        print(f"⚠️  WARNING: Validation set is <5% of total data ({val_ratio*100:.1f}%)")
        print("   → May not be representative of the full distribution")
    elif val_ratio > 0.30:
        print(f"⚠️  WARNING: Validation set is >30% of total data ({val_ratio*100:.1f}%)")
        print("   → You're not using enough data for training")
    else:
        print(f"✅ Train/val ratio looks good ({val_ratio*100:.1f}% validation)")

    print()
    print("="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    print()

    # Generate recommendations based on analysis
    recommendations = []

    if len(overlap_speakers) > 0 and len(overlap_speakers) < len(train_speakers):
        recommendations.append(
            "Consider creating a stratified split where EITHER:\n"
            "   a) All speakers appear in both train and val (seen-speaker)\n"
            "   b) No speakers overlap (zero-shot generalization)"
        )

    if val_utt_count < 500:
        recommendations.append(
            "Increase validation set size to 500-1000 utterances for stable metrics"
        )

    if train_utt_count < 1000:
        recommendations.append(
            "Training set is small (<1000 utterances).\n"
            "   → Use aggressive regularization (weight_decay=0.01, dropout)\n"
            "   → Reduce max_epoch to 3-5 to prevent overfitting"
        )

    if not recommendations:
        recommendations.append("Data split looks reasonable! Proceed with training.")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
        print()

    print("="*60)
    print("📋 SPEAKER OVERLAP DETAILS (first 10)")
    print("="*60)
    if overlap_speakers:
        for i, spk in enumerate(sorted(overlap_speakers)[:10], 1):
            print(f"  {i}. {spk}")
        if len(overlap_speakers) > 10:
            print(f"  ... and {len(overlap_speakers) - 10} more")
    else:
        print("  (none)")
    print()

if __name__ == "__main__":
    main()
