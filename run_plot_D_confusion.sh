#!/bin/bash
# Goal: Generate a dense dataset for the Confusion Matrix heatmap. What it does: Runs a single-pass experiment with 5-way grading. Note: I changed this to 5way because a confusion matrix for 2way (Correct/Incorrect) is just a 2x2 grid and isn't very informative. A 5-way matrix reveals interesting nuances (e.g., "Does the model confuse 'Partial' with 'Incorrect'?")

# Configuration
LLM_BACKEND="rcac"
MODEL_NAME="qwen2.5:72b"
SAMPLE_SIZE=40
BATCH_SIZE=8
LOG_DIR="logs/plot_D_confusion"
LABEL_SCHEME="5way"    # 5-way gives the most interesting confusion matrix

mkdir -p "$LOG_DIR"

BASE_ARGS="--llm-backend $LLM_BACKEND --model-name $MODEL_NAME --sample-size $SAMPLE_SIZE --batch-size $BATCH_SIZE --log-dir $LOG_DIR --label-schemes $LABEL_SCHEME --experiment single"

echo ">>> [Plot D] Starting Confusion Matrix Experiment..."
echo "Generating data for 5-way classification..."

python main.py $BASE_ARGS

echo ">>> [Plot D] Complete. Data saved in $LOG_DIR"
