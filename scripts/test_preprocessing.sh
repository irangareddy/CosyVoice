#!/bin/bash
# Test preprocessing pipeline with small subset
# This verifies everything works before running full 3-5 hour preprocessing

set -e

echo "========================================="
echo "TEST: Preprocessing Pipeline (Small Subset)"
echo "========================================="
echo "This will process ~100 samples to verify everything works"
echo ""

# Create test data directory
mkdir -p data/test_emotional_speech/{train,val}

echo "[TEST Step 1/4] Creating test subset (100 samples)..."
# Take first 100 lines from train manifest
head -100 /workspace/data/full/manifests/train.jsonl > /tmp/test_train.jsonl

# Take first 20 lines from val manifest
head -20 /workspace/data/full/manifests/val.jsonl > /tmp/test_val.jsonl

echo "[TEST Step 1/4] Converting JSONL to Kaldi format..."
for split in train val; do
    input_file="/tmp/test_${split}.jsonl"
    output_dir="data/test_emotional_speech/${split}"

    echo "  Processing test ${split} split..."
    python local/prepare_emotional_data.py \
        --jsonl_path "$input_file" \
        --output_dir "$output_dir" \
        --base_path /workspace \
        --add_emotion_prefix \
        --composite_speaker \
        --filter_datasets ESD,RAVDESS
done

echo ""
echo "[TEST Step 2/4] Extracting speaker embeddings (CAM++)..."
for split in train val; do
    echo "  Processing test ${split} split..."
    python tools/extract_embedding.py \
        --dir data/test_emotional_speech/${split} \
        --onnx_path /workspace/pretrained_models/CosyVoice2-0.5B/campplus.onnx \
        --num_thread 4
done

echo ""
echo "[TEST Step 3/4] Extracting speech tokens..."
for split in train val; do
    echo "  Processing test ${split} split..."
    python tools/extract_speech_token.py \
        --dir data/test_emotional_speech/${split} \
        --onnx_path /workspace/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \
        --num_thread 4
done

echo ""
echo "[TEST Step 4/4] Creating parquet files..."
for split in train val; do
    echo "  Processing test ${split} split..."
    mkdir -p data/test_emotional_speech/${split}/parquet
    python tools/make_parquet_list.py \
        --num_utts_per_parquet 50 \
        --num_processes 4 \
        --src_dir data/test_emotional_speech/${split} \
        --des_dir data/test_emotional_speech/${split}/parquet
done

# Create test data lists
echo ""
echo "Creating test data list files..."
cat data/test_emotional_speech/train/parquet/data.list > data/test_emotional_train.data.list
cat data/test_emotional_speech/val/parquet/data.list > data/test_emotional_val.data.list

echo ""
echo "========================================="
echo "TEST COMPLETED SUCCESSFULLY!"
echo "========================================="
echo "Processed:"
echo "  Train: $(wc -l < data/test_emotional_speech/train/wav.scp) samples"
echo "  Val: $(wc -l < data/test_emotional_speech/val/wav.scp) samples"
echo ""
echo "Output files:"
echo "  Train parquets: $(wc -l < data/test_emotional_train.data.list) files"
echo "  Val parquets: $(wc -l < data/test_emotional_val.data.list) files"
echo ""
echo "Verification:"
echo "  ✓ JSONL -> Kaldi conversion works"
echo "  ✓ Speaker embedding extraction works"
echo "  ✓ Speech token extraction works"
echo "  ✓ Parquet creation works"
echo ""
echo "========================================="
echo "All stages completed without errors!"
echo "You can now run the full preprocessing:"
echo "  bash scripts/prepare_data.sh"
echo "========================================="
