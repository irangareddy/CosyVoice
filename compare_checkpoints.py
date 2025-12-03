#!/usr/bin/env python3
"""Compare all HiFiGAN training checkpoints to find the best epoch"""

import os
import yaml
import glob
from pathlib import Path
from typing import Dict, List
import pandas as pd

def extract_epoch_from_filename(filename: str) -> int:
    """Extract epoch number from checkpoint filename"""
    try:
        # e.g., epoch_0_whole.yaml -> 0
        basename = os.path.basename(filename)
        epoch_str = basename.split('_')[1]
        return int(epoch_str)
    except:
        return -1

def load_checkpoint_metrics(yaml_file: str) -> Dict:
    """Load validation metrics from checkpoint YAML file"""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            return data
    except Exception as e:
        print(f"Error loading {yaml_file}: {e}")
        return {}

def compare_checkpoints(exp_dir: str = 'exp/emotional_hifigan'):
    """Compare all checkpoint validation metrics"""

    print("="*80)
    print("HiFiGAN Checkpoint Comparison")
    print("="*80)

    # Find all checkpoint YAML files
    yaml_files = glob.glob(os.path.join(exp_dir, 'epoch_*_whole.yaml'))

    if not yaml_files:
        print(f"\n❌ No checkpoint YAML files found in {exp_dir}")
        print("Make sure training has completed at least one epoch with validation.")
        return None

    print(f"\nFound {len(yaml_files)} checkpoint(s)")

    # Extract metrics from each checkpoint
    results = []
    for yaml_file in sorted(yaml_files):
        epoch = extract_epoch_from_filename(yaml_file)
        if epoch < 0:
            continue

        metrics = load_checkpoint_metrics(yaml_file)

        if not metrics:
            continue

        # Extract key validation metrics
        result = {
            'Epoch': epoch,
            'CV Loss': metrics.get('cv_loss', float('inf')),
            'Mel Loss': metrics.get('loss_mel', float('inf')),
            'F0 Loss': metrics.get('loss_f0', float('inf')),
            'Gen Loss': metrics.get('loss_gen', float('inf')),
            'FM Loss': metrics.get('loss_fm', float('inf')),
            'Checkpoint': os.path.basename(yaml_file).replace('.yaml', '.pt')
        }
        results.append(result)

    if not results:
        print("❌ No valid metrics found in checkpoint files")
        return None

    # Create DataFrame for easy comparison
    df = pd.DataFrame(results)
    df = df.sort_values('Epoch')

    # Display all checkpoints
    print("\n" + "="*80)
    print("ALL CHECKPOINTS (sorted by epoch)")
    print("="*80)
    print(df.to_string(index=False))

    # Find best checkpoint by different metrics
    print("\n" + "="*80)
    print("BEST CHECKPOINTS BY METRIC")
    print("="*80)

    best_cv = df.loc[df['CV Loss'].idxmin()]
    best_mel = df.loc[df['Mel Loss'].idxmin()]
    best_f0 = df.loc[df['F0 Loss'].idxmin()]

    print(f"\n🏆 Best Overall (CV Loss): Epoch {int(best_cv['Epoch'])} - Loss: {best_cv['CV Loss']:.2f}")
    print(f"   Checkpoint: {best_cv['Checkpoint']}")

    print(f"\n🎵 Best Audio Quality (Mel Loss): Epoch {int(best_mel['Epoch'])} - Loss: {best_mel['Mel Loss']:.4f}")
    print(f"   Checkpoint: {best_mel['Checkpoint']}")

    print(f"\n🎤 Best Prosody (F0 Loss): Epoch {int(best_f0['Epoch'])} - Loss: {best_f0['F0 Loss']:.2f}")
    print(f"   Checkpoint: {best_f0['Checkpoint']}")

    # Recommend best checkpoint
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Generally, best overall CV loss is the safest choice
    recommended_epoch = int(best_cv['Epoch'])
    recommended_checkpoint = best_cv['Checkpoint']

    print(f"\n✅ Recommended Checkpoint: Epoch {recommended_epoch}")
    print(f"   File: exp/emotional_hifigan/{recommended_checkpoint}")
    print(f"\n   Reason: Lowest overall validation loss (CV Loss: {best_cv['CV Loss']:.2f})")
    print(f"   This checkpoint provides the best balance across all metrics.")

    # Check for potential overfitting
    if len(df) > 5:
        last_5 = df.tail(5)
        if last_5['CV Loss'].is_monotonic_increasing:
            print(f"\n⚠️  Warning: CV loss increasing in last 5 epochs - possible overfitting")
            print(f"   Consider using an earlier checkpoint around epoch {int(df.loc[df['CV Loss'].idxmin()]['Epoch'])}")

    print("\n" + "="*80)

    return df

def generate_comparison_plot(df, output_dir='test_outputs'):
    """Generate plots comparing metrics across epochs"""
    try:
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('HiFiGAN Training Progress', fontsize=16, fontweight='bold')

        # Plot CV Loss
        axes[0, 0].plot(df['Epoch'], df['CV Loss'], marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Overall Validation Loss (CV Loss)', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        best_epoch = df.loc[df['CV Loss'].idxmin(), 'Epoch']
        axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best: Epoch {int(best_epoch)}')
        axes[0, 0].legend()

        # Plot Mel Loss
        axes[0, 1].plot(df['Epoch'], df['Mel Loss'], marker='o', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_title('Mel-Spectrogram Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        best_epoch = df.loc[df['Mel Loss'].idxmin(), 'Epoch']
        axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best: Epoch {int(best_epoch)}')
        axes[0, 1].legend()

        # Plot F0 Loss
        axes[1, 0].plot(df['Epoch'], df['F0 Loss'], marker='o', linewidth=2, markersize=6, color='orange')
        axes[1, 0].set_title('F0/Pitch Loss', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        best_epoch = df.loc[df['F0 Loss'].idxmin(), 'Epoch']
        axes[1, 0].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best: Epoch {int(best_epoch)}')
        axes[1, 0].legend()

        # Plot Gen Loss
        axes[1, 1].plot(df['Epoch'], df['Gen Loss'], marker='o', linewidth=2, markersize=6, color='purple')
        axes[1, 1].set_title('Generator Loss', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        best_epoch = df.loc[df['Gen Loss'].idxmin(), 'Epoch']
        axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best: Epoch {int(best_epoch)}')
        axes[1, 1].legend()

        plt.tight_layout()

        output_file = os.path.join(output_dir, 'checkpoint_comparison.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved to: {output_file}")

    except ImportError:
        print("\n⚠️  matplotlib not available - skipping plot generation")
        print("   Install with: pip install matplotlib")

if __name__ == '__main__':
    import sys

    # Allow custom experiment directory
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else 'exp/emotional_hifigan'

    print(f"Analyzing checkpoints in: {exp_dir}\n")

    df = compare_checkpoints(exp_dir)

    if df is not None:
        # Try to generate plots
        generate_comparison_plot(df)

        # Save comparison to CSV
        output_csv = 'test_outputs/checkpoint_comparison.csv'
        os.makedirs('test_outputs', exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"📄 Comparison data saved to: {output_csv}")
