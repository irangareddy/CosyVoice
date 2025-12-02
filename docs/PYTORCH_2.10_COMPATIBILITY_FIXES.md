# PyTorch 2.10 Compatibility Fixes for CosyVoice

## Overview

This document describes the fixes applied to make CosyVoice compatible with PyTorch 2.10 (CUDA 13.0), specifically for RTX 5090 support.

## Issues Encountered and Solutions

### 1. Triton Compatibility Issue

**Problem:**
```
ImportError: cannot import name 'ir' from 'triton._C.libtriton'
```

**Solution:**
Upgraded Triton from 2.3.1 to 3.5.1:
```bash
pip install --upgrade triton
```

**Note:** This causes a conflict with openai-whisper which requires `triton<3`, but this doesn't affect training functionality.

### 2. TorchCodec/Audio Loading Issue

**Problem:**
```
ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.
```

PyTorch 2.10's torchaudio defaults to using TorchCodec for audio loading, which requires FFmpeg libraries that weren't installed in the Docker container.

**Solution:**
Modified `cosyvoice/dataset/processor.py` to use `soundfile` directly instead of torchaudio's default loader:

**Changes:**
- Added import: `import soundfile as sf` and `import numpy as np`
- Replaced `torchaudio.load()` with direct soundfile usage:
```python
# Use soundfile directly to avoid torchcodec dependency
audio_data, sample_rate = sf.read(BytesIO(sample['audio_data']))
# Convert to tensor and handle shape: soundfile returns (num_samples, channels) or (num_samples,)
audio_tensor = torch.from_numpy(audio_data).float()
if audio_tensor.dim() == 1:
    sample['speech'] = audio_tensor.unsqueeze(0)
else:
    sample['speech'] = audio_tensor.permute(1, 0)  # (channels, num_samples)
sample['sample_rate'] = sample_rate
```

### 3. PyTorch Distributed API Change

**Problem:**
```
AttributeError: 'torch._C._distributed_c10d.ProcessGroup' object has no attribute 'options'
```

In PyTorch 2.10, the ProcessGroup API changed and no longer has an `.options` attribute.

**Solution:**
Modified `cosyvoice/utils/train_utils.py` in the `cosyvoice_join()` function:

**Changes:**
```python
# PyTorch 2.10+ compatibility: use default timeout if options not available
timeout = getattr(group_join, 'options', None)
if timeout is not None and hasattr(timeout, '_timeout'):
    timeout = timeout._timeout
else:
    timeout = None  # Use default timeout
dist.monitored_barrier(group=group_join, timeout=timeout)
```

## Testing

After applying these fixes, training completed successfully:
- Test run: 5 epochs on 2 samples
- Loss decreased from ~4.2 to ~3.96
- Checkpoints saved correctly (1.9GB per epoch)
- All training pipeline stages working

## Deprecation Warnings (Non-Breaking)

The following warnings appear but don't affect functionality:
1. `torch.cuda.amp.GradScaler` - deprecated in favor of `torch.amp.GradScaler('cuda')`
2. `torch.cuda.amp.autocast` - deprecated in favor of `torch.amp.autocast('cuda')`
3. Tensor transpose `.T` usage on non-2D tensors - already fixed by using `.permute()`
4. `pin_memory(device)` argument deprecated

These can be addressed in future updates but don't impact training.

## Files Modified

1. `cosyvoice/dataset/processor.py` - Audio loading fix
2. `cosyvoice/utils/train_utils.py` - Distributed training API fix

## Environment

- PyTorch: 2.10.0.dev20251123+cu130
- CUDA: 13.0
- GPU: NVIDIA GeForce RTX 5090
- Triton: 3.5.1
- Python: 3.10
