# CosyVoice2 Inference Guide

## Generated Audio Samples

Successfully generated audio samples using CosyVoice2-0.5B with PyTorch 2.10 and RTX 5090:

### Available Audio Files

1. **output_zero_shot_0.wav** (511 KB)
   - English zero-shot voice cloning
   - Text: "Hello! This is a test of CosyVoice with PyTorch 2.10 and RTX 5090."
   - Uses emotional prompt from training data

2. **output_cross_lingual_0.wav** (156 KB)
   - Cross-lingual synthesis (Chinese text with English reference)
   - Text: "这是一个跨语言语音合成测试。"
   - Demonstrates voice transfer across languages

3. **demo_english.wav** (402 KB)
   - High-quality English synthesis
   - Text: "Hello! This is a test of CosyVoice voice cloning with PyTorch 2.10 and RTX 5090. The quality is amazing!"
   - Uses asset/zero_shot_prompt.wav as reference

## Running Inference

### Quick Start Script

Use the provided `quick_inference.py` script:

```bash
# Inside the Docker container
conda activate cosyvoice
cd /workspace/CosyVoice
python quick_inference.py
```

### Manual Inference Example

```python
import sys
sys.path.append('third_party/Matcha-TTS')

import soundfile as sf
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Load model
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False
)

# Load reference audio
prompt_speech = load_wav('asset/zero_shot_prompt.wav', 16000)

# Generate speech
target_text = "Your text here"
prompt_text = "Description of the voice"

for audio_data in cosyvoice.inference_zero_shot(
    target_text,
    prompt_text,
    prompt_speech,
    stream=False
):
    # Save using soundfile (PyTorch 2.10 compatibility)
    sf.write(
        'output.wav',
        audio_data['tts_speech'].squeeze(0).cpu().numpy(),
        cosyvoice.sample_rate
    )
```

## Inference Modes

### 1. Zero-Shot Voice Cloning
Clone any voice from a reference audio sample:
```python
cosyvoice.inference_zero_shot(target_text, prompt_text, prompt_speech, stream=False)
```

### 2. Cross-Lingual Synthesis
Generate speech in one language using a reference from another:
```python
cosyvoice.inference_cross_lingual(target_text, prompt_speech, stream=False)
```

### 3. Instruct Mode (if available)
Control emotion, speaking style, etc.:
```python
cosyvoice.inference_instruct(target_text, instruction, prompt_speech, stream=False)
```

## Performance

Real-time Factor (RTF) on RTX 5090:
- Zero-shot synthesis: ~0.54 (faster than real-time)
- Cross-lingual synthesis: ~0.47 (faster than real-time)

Lower RTF = faster generation (e.g., 0.5 RTF means 1 second of audio generated in 0.5 seconds)

## Known Issues with PyTorch 2.10

1. **Embedding Type Issue**: Some synthesis modes may fail with "Expected Long tensor but got Float" error. This is a PyTorch 2.10 compatibility issue being investigated.

2. **ONNX Runtime Warning**: CUDA provider fails to load with cuBLAS error. The model falls back to CPU for ONNX operations (speaker embedding, speech tokenization). This doesn't significantly impact performance as most compute is on GPU.

## Workarounds Applied

All audio loading/saving operations use `soundfile` directly instead of `torchaudio` to avoid the torchcodec dependency issue:

- Modified: `cosyvoice/utils/file_utils.py`
- Modified: `cosyvoice/dataset/processor.py`

## Tips

1. **Reference Audio Quality**: Use high-quality, clean reference audio (16kHz minimum) for best results
2. **Text Length**: Keep generated text under 100 words for stable generation
3. **Prompt Text**: Provide clear descriptions of the desired voice characteristics
4. **Streaming Mode**: Set `stream=True` for real-time generation (CosyVoice2 only)

## Copying Files from Docker

To copy generated audio files to your host machine:

```bash
# From host machine
docker cp cosyvoice-finetune:/workspace/CosyVoice/output.wav ./output.wav
```

Or mount a volume when running the container:
```bash
docker run -v /host/path:/container/path ...
```
