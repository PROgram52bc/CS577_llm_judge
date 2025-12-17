#!/bin/bash
# Goal: Generate data for the "Robustness Drop" bar chart. What it does: Runs the standard baseline and then re-runs it with every available data augmentation (OCR, Typos, etc.).

# Configuration
LLM_BACKEND="rcac"
MODEL_NAME="qwen2.5:72b"
SAMPLE_SIZE=40        # High sample size for significant results
BATCH_SIZE=8
LOG_DIR="logs/plot_A_robustness"
LABEL_SCHEME="2way"    # Using 2way usually shows clearer drops in robustness

mkdir -p "$LOG_DIR"

BASE_ARGS="--llm-backend $LLM_BACKEND --model-name $MODEL_NAME --sample-size $SAMPLE_SIZE --batch-size $BATCH_SIZE --log-dir $LOG_DIR --label-schemes $LABEL_SCHEME --experiment single"

echo ">>> [Plot A] Starting Robustness Experiments..."

# 1. Baseline (No noise)
echo "Running Baseline..."
python main.py $BASE_ARGS --label "baseline"

# 2. Augmentations
echo "Running OCR Augmentation..."
python main.py $BASE_ARGS --ocr-augment --label "ocr"

echo "Running Typo Augmentation..."
python main.py $BASE_ARGS --typos --label "typo"

echo "Running Non-Influential Word Injection..."
python main.py $BASE_ARGS --non-influential-words --label "non_influential"

echo "Running Hyphen Injection..."
python main.py $BASE_ARGS --hyphens --label "hyphen"

echo "Running Non-Unicode Injection..."
python main.py $BASE_ARGS --non-unicode --label "non_unicode"

echo "Running Synonym Substitution..."
python main.py $BASE_ARGS --substitute-synonyms --label "synonym"

echo "Running Paraphrasing..."
python main.py $BASE_ARGS --paraphrase --label "paraphrase"

echo ">>> [Plot A] Complete. Data saved in $LOG_DIR"
