#!/bin/bash
# Data Preparation Pipeline for Emotional Speech Fine-Tuning
# Converts JSONL manifests to CosyVoice format
# Filters to ESD + RAVDESS only (excludes CREMA-D)

set -e

echo "========================================="
echo "Emotional Speech Data Preparation"
echo "========================================="
echo "Dataset: ESD + RAVDESS (filtered, no CREMA-D)"
echo ""

# Step 1: Convert JSONL to Kaldi format
echo "[Step 1/4] Converting JSONL manifests to Kaldi format..."
mkdir -p data/emotional_speech/{train,val,test}

for split in train val test; do
    echo "  Processing $split split..."
    python local/prepare_emotional_data.py \
        --jsonl_path /workspace/data/full/manifests/$split.jsonl \
        --output_dir data/emotional_speech/$split \
        --base_path /workspace \
        --add_emotion_prefix \
        --composite_speaker \
        --filter_datasets ESD,RAVDESS
done

echo ""
echo "[Step 2/4] Extracting speaker embeddings (CAM++)..."
for split in train val test; do
    echo "  Processing $split split..."
    tools/extract_embedding.py \
        --dir data/emotional_speech/$split \
        --onnx_path /workspace/pretrained_models/CosyVoice2-0.5B/campplus.onnx \
        --num_thread 8
done

echo ""
echo "[Step 3/4] Extracting speech tokens..."
for split in train val test; do
    echo "  Processing $split split..."
    tools/extract_speech_token.py \
        --dir data/emotional_speech/$split \
        --onnx_path /workspace/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \
        --num_thread 8
done

echo ""
echo "[Step 4/4] Creating parquet files..."
for split in train val test; do
    echo "  Processing $split split..."
    mkdir -p data/emotional_speech/$split/parquet
    tools/make_parquet_list.py \
        --num_utts_per_parquet 1000 \
        --num_processes 8 \
        --src_dir data/emotional_speech/$split \
        --des_dir data/emotional_speech/$split/parquet
done

# Create combined data lists for training
echo ""
echo "Creating data list files..."
cat data/emotional_speech/train/parquet/data.list > data/emotional_train.data.list
cat data/emotional_speech/val/parquet/data.list > data/emotional_val.data.list

echo ""
echo "========================================="
echo "Data preparation completed!"
echo "========================================="
echo "Train data: $(wc -l < data/emotional_train.data.list) parquet files"
echo "Val data: $(wc -l < data/emotional_val.data.list) parquet files"
echo ""
echo "You can now start training with:"
echo "  bash scripts/train_emotional_sft.sh"
echo "========================================="
