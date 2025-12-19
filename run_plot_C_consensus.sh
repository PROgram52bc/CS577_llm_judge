#!/bin/bash

# Goal: Generate data for the "Coverage vs. Accuracy" curve efficiently.
# What it does: Runs a single consensus experiment that collects votes once,
# then internally calculates metrics for a range of thresholds.

# Configuration
LLM_BACKEND="rcac"
MODEL_NAME="qwen2.5:72b"
SAMPLE_SIZE=500
BATCH_SIZE=8
CONSENSUS_RUNS=10       # Number of voters per question
LOG_DIR="logs/plot_C_consensus"
LABEL_SCHEME="5way"

# Define the list of thresholds to be tested inside the experiment
THRESHOLDS="0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0"

mkdir -p "$LOG_DIR"

BASE_ARGS="--llm-backend $LLM_BACKEND --model-name $MODEL_NAME --sample-size $SAMPLE_SIZE --batch-size $BATCH_SIZE --log-dir $LOG_DIR --label-schemes $LABEL_SCHEME --consensus-runs $CONSENSUS_RUNS"

echo ">>> [Plot C] Starting Efficient Consensus Curve Experiment..."

# Run the experiment once, passing all thresholds to be tested internally
python main.py $BASE_ARGS --experiment consensus_curve --consensus-thresholds $THRESHOLDS

echo ">>> [Plot C] Complete. Data saved in $LOG_DIR"
