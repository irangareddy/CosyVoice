#!/bin/bash
# Full Dataset Preparation Script
# This prepares all 7,442 CREMA-D audio files for training

set -e  # Exit on error

echo "=================================================="
echo "CosyVoice Full Dataset Preparation"
echo "=================================================="
echo ""

# Configuration
DATA_DIR="/workspace/CosyVoice/data/full"
WAV_DIR="$DATA_DIR/wav24k/cremad"
OUTPUT_DIR="$DATA_DIR"
PRETRAINED_MODEL="/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B"

# Count files
NUM_FILES=$(find $WAV_DIR -name "*.wav" | wc -l)
echo "Found $NUM_FILES audio files"
echo ""

# Stage 0: Create metadata files (wav.scp, text, utt2spk, spk2utt)
echo "=================================================="
echo "Stage 0: Creating metadata files"
echo "=================================================="

mkdir -p $OUTPUT_DIR

# Create wav.scp (audio_id -> file_path)
echo "Creating wav.scp..."
find $WAV_DIR -name "*.wav" | sort | while read wav_file; do
    filename=$(basename "$wav_file" .wav)
    echo "$filename $wav_file"
done > $OUTPUT_DIR/wav.scp

# Extract speaker IDs and emotions from filenames
# CREMA-D format: speakerID_sentence_emotion_intensity.wav
# Example: 1001_DFA_ANG_XX.wav
echo "Creating text file..."
cat $OUTPUT_DIR/wav.scp | while read line; do
    utt_id=$(echo $line | awk '{print $1}')
    # For CREMA-D, we don't have text transcriptions
    # Use emotion label as text for now
    emotion=$(echo $utt_id | cut -d'_' -f3)
    case $emotion in
        ANG) text="angry speech" ;;
        DIS) text="disgusted speech" ;;
        FEA) text="fearful speech" ;;
        HAP) text="happy speech" ;;
        NEU) text="neutral speech" ;;
        SAD) text="sad speech" ;;
        *) text="emotional speech" ;;
    esac
    echo "$utt_id $text"
done > $OUTPUT_DIR/text

# Create utt2spk (utterance_id -> speaker_id)
echo "Creating utt2spk..."
cat $OUTPUT_DIR/wav.scp | while read line; do
    utt_id=$(echo $line | awk '{print $1}')
    spk_id=$(echo $utt_id | cut -d'_' -f1)
    echo "$utt_id $spk_id"
done > $OUTPUT_DIR/utt2spk

# Create spk2utt (speaker_id -> list of utterance_ids)
echo "Creating spk2utt..."
cat $OUTPUT_DIR/utt2spk | \
    awk '{spk[$2]=spk[$2] " " $1} END {for (s in spk) print s, spk[s]}' | \
    sort > $OUTPUT_DIR/spk2utt

# Statistics
NUM_UTTS=$(wc -l < $OUTPUT_DIR/wav.scp)
NUM_SPKS=$(wc -l < $OUTPUT_DIR/spk2utt)

echo "✓ Metadata created:"
echo "  - Utterances: $NUM_UTTS"
echo "  - Speakers: $NUM_SPKS"
echo ""

# Stage 1: Extract speaker embeddings
echo "=================================================="
echo "Stage 1: Extracting speaker embeddings"
echo "=================================================="
echo "This will take ~10-15 minutes for 7,442 files..."
echo ""

mkdir -p $OUTPUT_DIR/speaker_embedding

python tools/extract_embedding.py \
    --model_dir $PRETRAINED_MODEL \
    --wav_scp $OUTPUT_DIR/wav.scp \
    --output_dir $OUTPUT_DIR/speaker_embedding \
    --batch_size 32

echo "✓ Speaker embeddings extracted"
echo ""

# Stage 2: Extract speech tokens
echo "=================================================="
echo "Stage 2: Extracting speech tokens"
echo "=================================================="
echo "This will take ~15-20 minutes for 7,442 files..."
echo ""

mkdir -p $OUTPUT_DIR/speech_token

python tools/extract_speech_token.py \
    --model_dir $PRETRAINED_MODEL \
    --wav_scp $OUTPUT_DIR/wav.scp \
    --output_dir $OUTPUT_DIR/speech_token \
    --batch_size 32

echo "✓ Speech tokens extracted"
echo ""

# Stage 3: Create train/dev split and generate parquet files
echo "=================================================="
echo "Stage 3: Creating train/dev split"
echo "=================================================="

# Split data (90% train, 10% dev)
total_lines=$(wc -l < $OUTPUT_DIR/wav.scp)
train_lines=$((total_lines * 90 / 100))

# Shuffle and split
cat $OUTPUT_DIR/wav.scp | shuf > $OUTPUT_DIR/wav.scp.shuffled
head -n $train_lines $OUTPUT_DIR/wav.scp.shuffled > $OUTPUT_DIR/wav.scp.train
tail -n +$((train_lines + 1)) $OUTPUT_DIR/wav.scp.shuffled > $OUTPUT_DIR/wav.scp.dev

echo "✓ Data split created:"
echo "  - Training: $(wc -l < $OUTPUT_DIR/wav.scp.train) utterances"
echo "  - Development: $(wc -l < $OUTPUT_DIR/wav.scp.dev) utterances"
echo ""

# Stage 4: Generate parquet files
echo "=================================================="
echo "Stage 4: Generating parquet files"
echo "=================================================="
echo "This will take ~20-30 minutes for 7,442 files..."
echo ""

# Training set
mkdir -p $OUTPUT_DIR/parquet/train
python tools/make_parquet_list.py \
    --num_utts_per_parquet 100 \
    --num_processes 4 \
    --wav_scp $OUTPUT_DIR/wav.scp.train \
    --text $OUTPUT_DIR/text \
    --utt2spk $OUTPUT_DIR/utt2spk \
    --speaker_embedding_dir $OUTPUT_DIR/speaker_embedding \
    --speech_token_dir $OUTPUT_DIR/speech_token \
    --output_dir $OUTPUT_DIR/parquet/train

# Development set
mkdir -p $OUTPUT_DIR/parquet/dev
python tools/make_parquet_list.py \
    --num_utts_per_parquet 100 \
    --num_processes 4 \
    --wav_scp $OUTPUT_DIR/wav.scp.dev \
    --text $OUTPUT_DIR/text \
    --utt2spk $OUTPUT_DIR/utt2spk \
    --speaker_embedding_dir $OUTPUT_DIR/speaker_embedding \
    --speech_token_dir $OUTPUT_DIR/speech_token \
    --output_dir $OUTPUT_DIR/parquet/dev

echo "✓ Parquet files generated"
echo ""

# Create data.list files
echo "=================================================="
echo "Stage 5: Creating data.list files"
echo "=================================================="

find $OUTPUT_DIR/parquet/train -name "*.parquet" | sort > $OUTPUT_DIR/parquet/train/data.list
find $OUTPUT_DIR/parquet/dev -name "*.parquet" | sort > $OUTPUT_DIR/parquet/dev/data.list

train_parquet=$(wc -l < $OUTPUT_DIR/parquet/train/data.list)
dev_parquet=$(wc -l < $OUTPUT_DIR/parquet/dev/data.list)

echo "✓ Data lists created:"
echo "  - Training parquet files: $train_parquet"
echo "  - Development parquet files: $dev_parquet"
echo ""

# Final summary
echo "=================================================="
echo "✅ DATA PREPARATION COMPLETE!"
echo "=================================================="
echo ""
echo "Summary:"
echo "  - Total audio files: $NUM_FILES"
echo "  - Speakers: $NUM_SPKS"
echo "  - Training samples: $(wc -l < $OUTPUT_DIR/wav.scp.train)"
echo "  - Development samples: $(wc -l < $OUTPUT_DIR/wav.scp.dev)"
echo "  - Training parquet files: $train_parquet"
echo "  - Development parquet files: $dev_parquet"
echo ""
echo "Dataset ready for training!"
echo "Next step: Run the training script"
echo ""
echo "=================================================="
