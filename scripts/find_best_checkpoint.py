#!/usr/bin/env python3
"""
Find the best checkpoint based on validation loss from TensorBoard logs
Usage: python scripts/find_best_checkpoint.py --model_dir exp/emotional_sft/llm
"""

import argparse
import os
import re
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import torch

def parse_tensorboard_logs(tensorboard_dir):
    """Parse TensorBoard event files and extract CV loss per epoch"""
    ea = event_accumulator.EventAccumulator(tensorboard_dir)
    ea.Reload()

    # Get all scalar tags
    tags = ea.Tags()['scalars']
    cv_tags = [t for t in tags if 'CV' in t and 'loss' in t.lower()]

    if not cv_tags:
        print(f"Warning: No CV loss tags found in {tensorboard_dir}")
        print(f"Available tags: {tags}")
        return {}

    # Extract CV loss per epoch
    epoch_losses = {}
    for tag in cv_tags:
        events = ea.Scalars(tag)
        for event in events:
            epoch = event.step
            loss = event.value
            if epoch not in epoch_losses:
                epoch_losses[epoch] = {}
            epoch_losses[epoch][tag] = loss

    return epoch_losses

def find_best_checkpoint_from_logs(model_dir, tensorboard_dir=None):
    """Find best checkpoint based on validation loss"""
    model_dir = Path(model_dir)

    # Find tensorboard directory
    if tensorboard_dir is None:
        tensorboard_dir = str(model_dir).replace('exp/', 'tensorboard/')

    print(f"Looking for TensorBoard logs in: {tensorboard_dir}")

    if not os.path.exists(tensorboard_dir):
        print(f"TensorBoard directory not found: {tensorboard_dir}")
        return None

    # Parse logs
    try:
        epoch_losses = parse_tensorboard_logs(tensorboard_dir)
    except Exception as e:
        print(f"Error parsing TensorBoard logs: {e}")
        epoch_losses = {}

    # If no logs, find checkpoints manually
    checkpoints = list(model_dir.glob('epoch_*_whole.pt'))

    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        return None

    # Try to get loss from checkpoint files
    checkpoint_info = []
    for ckpt_path in checkpoints:
        # Extract epoch from filename
        match = re.search(r'epoch_(\d+)_whole', ckpt_path.name)
        if not match:
            continue
        epoch = int(match.group(1))

        # Try to load checkpoint and get loss
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # Look for validation loss in checkpoint
            cv_loss = None
            if 'cv_loss' in ckpt:
                cv_loss = ckpt['cv_loss']
            elif 'loss_dict' in ckpt:
                # Get total loss
                cv_loss = sum(ckpt['loss_dict'].values()) if isinstance(ckpt['loss_dict'], dict) else ckpt['loss_dict']

            # Use TensorBoard log if available
            if epoch in epoch_losses:
                # Average all CV losses for this epoch
                cv_loss = sum(epoch_losses[epoch].values()) / len(epoch_losses[epoch])

            checkpoint_info.append({
                'epoch': epoch,
                'path': ckpt_path,
                'cv_loss': cv_loss
            })
        except Exception as e:
            print(f"Warning: Could not load {ckpt_path}: {e}")
            checkpoint_info.append({
                'epoch': epoch,
                'path': ckpt_path,
                'cv_loss': None
            })

    # Sort by CV loss (lowest first)
    valid_checkpoints = [c for c in checkpoint_info if c['cv_loss'] is not None]
    if valid_checkpoints:
        valid_checkpoints.sort(key=lambda x: x['cv_loss'])
        return valid_checkpoints
    else:
        # Sort by epoch (latest first) if no loss available
        checkpoint_info.sort(key=lambda x: x['epoch'], reverse=True)
        return checkpoint_info

def main():
    parser = argparse.ArgumentParser(description='Find best checkpoint based on validation loss')
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory (e.g., exp/emotional_sft/llm)')
    parser.add_argument('--tensorboard_dir', type=str, default=None, help='TensorBoard directory (auto-detected if not specified)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top checkpoints to display')
    args = parser.parse_args()

    print("=" * 60)
    print("Finding Best Checkpoints")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print()

    checkpoints = find_best_checkpoint_from_logs(args.model_dir, args.tensorboard_dir)

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print()
    print(f"Top {min(args.top_k, len(checkpoints))} checkpoints:")
    print("-" * 60)

    for i, ckpt in enumerate(checkpoints[:args.top_k]):
        rank = i + 1
        epoch = ckpt['epoch']
        cv_loss = ckpt['cv_loss']
        path = ckpt['path']

        if cv_loss is not None:
            print(f"{rank}. Epoch {epoch:3d} | CV Loss: {cv_loss:.4f} | {path}")
        else:
            print(f"{rank}. Epoch {epoch:3d} | CV Loss: N/A      | {path}")

    print()
    print("=" * 60)
    print("Best checkpoint recommendation:")
    best = checkpoints[0]
    print(f"  Epoch: {best['epoch']}")
    if best['cv_loss']:
        print(f"  CV Loss: {best['cv_loss']:.4f}")
    print(f"  Path: {best['path']}")
    print()
    print("To use this checkpoint:")
    print(f"  cp {best['path']} pretrained_models/CosyVoice2-0.5B/llm.pt")
    print("=" * 60)

if __name__ == '__main__':
    main()
