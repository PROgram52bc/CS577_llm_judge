#!/bin/bash

# Goal: Generate data for the "Coverage vs. Accuracy" curve efficiently for multiple label schemes.
# What it does: Runs the consensus_curve experiment three times (2-way, 3-way, 5-way),
# collecting votes once per run and internally calculating metrics for a range of thresholds.

# Configuration
LLM_BACKEND="rcac"
MODEL_NAME="qwen2.5:72b"
SAMPLE_SIZE=500
BATCH_SIZE=8
CONSENSUS_RUNS=10       # Number of voters per question
LOG_DIR="logs/plot_C_consensus"

# Define the list of thresholds to be tested inside the experiment
THRESHOLDS="0.55 0.65 0.75 0.85 0.95"

mkdir -p "$LOG_DIR"

echo ">>> [Plot C] Starting Efficient Consensus Curve Experiments for all schemes..."

for scheme in 2way 3way 5way; do
    echo "--- Running for Label Scheme: $scheme ---"
    
    BASE_ARGS="--llm-backend $LLM_BACKEND --model-name $MODEL_NAME --sample-size $SAMPLE_SIZE --batch-size $BATCH_SIZE --log-dir $LOG_DIR --label-schemes $scheme --consensus-runs $CONSENSUS_RUNS"

    # Run the experiment once per scheme, passing all thresholds to be tested internally
    python main.py $BASE_ARGS --experiment consensus_curve --consensus-thresholds $THRESHOLDS
done

echo ">>> [Plot C] Complete. Data saved in $LOG_DIR"
