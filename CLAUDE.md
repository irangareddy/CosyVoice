# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CosyVoice is a multilingual text-to-speech (TTS) system that combines large language models with speech synthesis. The project supports multiple versions:
- **CosyVoice 1.0**: 300M parameter model with zero-shot voice cloning
- **CosyVoice 2.0**: 0.5B parameter model with streaming support and improved quality
- **CosyVoice 3.0**: Latest version with enhanced performance

Key capabilities: zero-shot voice cloning, cross-lingual synthesis, streaming inference, instruction-based synthesis (emotion/dialect control), and support for Chinese, English, Japanese, Korean, and Chinese dialects.

## Architecture

The system has three main components that work sequentially:

1. **LLM (Language Model)**: Converts text and speaker embeddings to discrete speech tokens
   - CosyVoice 1.0: Uses TransformerLM with Conformer text encoder
   - CosyVoice 2.0: Uses Qwen2-based architecture with causal modeling for streaming
   - Located in `cosyvoice/llm/llm.py`

2. **Flow (Flow Matching)**: Converts discrete tokens to mel-spectrograms
   - Uses conditional flow matching with diffusion-based decoder
   - CosyVoice 2.0 has causal flow decoder for streaming support
   - Located in `cosyvoice/flow/`

3. **HiFiGAN (Vocoder)**: Converts mel-spectrograms to waveforms
   - Uses HiFT (HiFiGAN with F0 predictor) for high-quality audio
   - Located in `cosyvoice/hifigan/`

Supporting components:
- **Frontend**: Text normalization and tokenization (`cosyvoice/cli/frontend.py`)
- **Speaker Encoder**: CAM++ ONNX model extracts speaker embeddings from reference audio
- **Speech Tokenizer**: ONNX model converts audio to discrete tokens for training

## Key File Locations

- Main inference API: `cosyvoice/cli/cosyvoice.py` (CosyVoice, CosyVoice2 classes)
- Model implementations: `cosyvoice/cli/model.py`
- Training script: `cosyvoice/bin/train.py`
- Model export: `cosyvoice/bin/export_jit.py`, `cosyvoice/bin/export_onnx.py`
- Data processing: `cosyvoice/dataset/processor.py`
- Configuration files: `examples/libritts/cosyvoice*/conf/*.yaml`

## Environment Setup

**Critical**: This project uses a git submodule (Matcha-TTS). Always set PYTHONPATH:
```bash
export PYTHONPATH=third_party/Matcha-TTS
```

Setup commands:
```bash
# Clone with submodules
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
git submodule update --init --recursive

# Create environment
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt

# Download models (using Python SDK)
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

## Running Inference

Basic usage pattern:
```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# For CosyVoice2 (recommended)
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# Zero-shot voice cloning
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物...', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('output.wav', j['tts_speech'], cosyvoice.sample_rate)
```

Web demo:
```bash
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

## Training Pipeline

Training scripts are in `examples/libritts/cosyvoice*/run.sh`. The pipeline has these stages:

**Data Preparation** (stages -1 to 3):
- Stage -1: Download LibriTTS dataset
- Stage 0: Prepare wav.scp/text/utt2spk/spk2utt files
- Stage 1: Extract speaker embeddings using `tools/extract_embedding.py`
- Stage 2: Extract speech tokens using `tools/extract_speech_token.py`
- Stage 3: Create parquet format data using `tools/make_parquet_list.py`

**Training** (stage 5):
```bash
# Train LLM, Flow, or HiFiGAN
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \  # or 'deepspeed'
  --config conf/cosyvoice.yaml \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --model llm \  # or 'flow' or 'hifigan'
  --checkpoint pretrained_models/CosyVoice-300M/llm.pt \
  --model_dir exp/cosyvoice/llm/torch_ddp \
  --tensorboard_dir tensorboard/cosyvoice/llm/torch_ddp
```

For CosyVoice2 training, use `conf/cosyvoice2.yaml` and add `--qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN`.

**Model Averaging** (stage 6):
```bash
python cosyvoice/bin/average_model.py \
  --dst_model exp/cosyvoice/llm/torch_ddp/llm.pt \
  --src_path exp/cosyvoice/llm/torch_ddp \
  --num 5 \
  --val_best
```

**Model Export** (stage 7):
```bash
python cosyvoice/bin/export_jit.py --model_dir pretrained_models/CosyVoice-300M
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/CosyVoice-300M
```

## Testing and Linting

Lint code before committing:
```bash
# Install linting tools
pip install flake8==3.8.2 flake8-bugbear flake8-comprehensions flake8-executable flake8-pyi==20.5.0 mccabe pycodestyle==2.6.0 pyflakes==2.2.0

# Run linter
flake8 --max-line-length 180 --ignore B006,B008,B905,C408,E402,E731,E741,W503,W504,F401,F403,F405,F841 --exclude ./third_party/,./runtime/python/grpc/cosyvoice_pb2*py
```

The CI also checks for tabs and trailing whitespace in code files.

## Deployment

**FastAPI Server**:
```bash
cd runtime/python/fastapi
python3 server.py --port 50000 --model_dir iic/CosyVoice-300M

# Client usage
python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```

**gRPC Server**:
```bash
cd runtime/python/grpc
python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M
python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```

**TensorRT-LLM Runtime** (for 4x LLM acceleration):
```bash
cd runtime/triton_trtllm
docker compose up -d
```

**VLLM Runtime** (CosyVoice2 only):
Requires `vllm==v0.9.0` and `transformers==4.51.3`. Create separate environment due to strict dependencies:
```bash
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
pip install vllm==v0.9.0 transformers==4.51.3
python vllm_example.py
```

## Configuration Files

Model configs use HyperPyYAML format (`examples/libritts/cosyvoice*/conf/*.yaml`):
- `sample_rate`: 22050 (CosyVoice 1.0) or 24000 (CosyVoice 2.0)
- `llm`, `flow`, `hift`: Model architecture definitions
- `data_pipeline`: Data preprocessing pipeline
- `train_conf`: Training hyperparameters (optimizer, scheduler, learning rate)

Key differences between CosyVoice 1.0 and 2.0:
- CosyVoice 2.0 uses Qwen2-based LLM instead of Transformer
- CosyVoice 2.0 has causal flow decoder for streaming (`chunk_size: 25`)
- CosyVoice 2.0 uses 25 Hz token frame rate vs 50 Hz in 1.0

## Important Notes

1. **Matcha-TTS dependency**: Always ensure `third_party/Matcha-TTS` is available and PYTHONPATH is set
2. **Text frontend**: Use `text_frontend=False` during inference to reproduce official demo results
3. **Model selection**: Use CosyVoice2-0.5B for best performance (lower latency, better quality)
4. **Training modes**:
   - Pretrain: Higher LR (0.001), warmup scheduler
   - SFT (supervised fine-tuning): Lower LR (1e-5), constant scheduler, enable `use_spk_embedding`
5. **DeepSpeed**: Has separate optimizer config in `conf/ds_stage2.json`
6. **Streaming inference**: Only supported in CosyVoice2 with causal models (set `stream=True`)
7. **Speed control**: Available via `speed` parameter in inference methods
8. **Zero-shot speakers**: Can be saved and reused via `add_zero_shot_spk()` and `save_spkinfo()`
9. **Training components separately**: Train llm, flow, and hifigan models independently
10. **GAN training**: Must use `accum_grad: 1` (no gradient accumulation)
