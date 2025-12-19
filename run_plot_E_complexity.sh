#!/bin/bash
# Goal: Generate data for the "Label Scheme Complexity" plot.
# What it does: Runs the single-pass experiment twice, once for 2-way grading and once for 5-way grading, to compare performance.

# Configuration
LLM_BACKEND="rcac"
MODEL_NAME="qwen2.5:72b"
SAMPLE_SIZE=40
BATCH_SIZE=8
LOG_DIR="logs/plot_E_complexity"
EXPERIMENT="single"

mkdir -p "$LOG_DIR"

BASE_ARGS="--llm-backend $LLM_BACKEND --model-name $MODEL_NAME --sample-size $SAMPLE_SIZE --batch-size $BATCH_SIZE --log-dir $LOG_DIR --experiment $EXPERIMENT"

echo ">>> [Plot E] Starting Label Scheme Complexity Experiments..."

# 1. 2-Way Grading
echo "Running 2-way classification..."
python main.py $BASE_ARGS --label-schemes "2way" --label "2way"

# 2. 3-Way Grading
echo "Running 3-way classification..."
python main.py $BASE_ARGS --label-schemes "3way" --label "3way"

# 3. 5-Way Grading
echo "Running 5-way classification..."
python main.py $BASE_ARGS --label-schemes "5way" --label "5way"

echo ">>> [Plot E] Complete. Data saved in $LOG_DIR"
