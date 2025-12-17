#!/bin/bash

# Goal: Generate data for the "Coverage vs. Accuracy" curve. What it does: Runs the consensus experiment multiple times, iterating through different consensus-threshold values (0.5 to 1.0).

# Configuration
LLM_BACKEND="rcac"
MODEL_NAME="qwen2.5:72b"
SAMPLE_SIZE=40
BATCH_SIZE=8
CONSENSUS_RUNS=3       # Number of voters per question
LOG_DIR="logs/plot_C_consensus"
LABEL_SCHEME="2way"

mkdir -p "$LOG_DIR"
BASE_ARGS="--llm-backend $LLM_BACKEND --model-name $MODEL_NAME --sample-size $SAMPLE_SIZE --batch-size $BATCH_SIZE --log-dir $LOG_DIR --label-schemes $LABEL_SCHEME --experiment consensus --consensus-runs $CONSENSUS_RUNS"

echo ">>> [Plot C] Starting Consensus Curve Experiments..."

# Iterate through thresholds to build the curve
# 0.5 = Majority vote (lowest strictness)
# 1.0 = Unanimous vote (highest strictness, lowest coverage)
for THRESHOLD in 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "Running Consensus with Threshold: $THRESHOLD..."
    python main.py $BASE_ARGS --consensus-threshold $THRESHOLD --label "consensus_$THRESHOLD"
done

echo ">>> [Plot C] Complete. Data saved in $LOG_DIR"
