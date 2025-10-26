# CS577 LLM Judge

This project provides a minimal framework for running "LLM-as-a-judge" experiments
with modular data loading, logging, and model interfaces. It supports both cloud
(model APIs such as OpenAI) and local heuristic models, enabling rapid
experimentation with grading tasks.

## Features

- Modular dataset loading via the `datasets` library or CSV files.
- Experiment-specific logging that captures prompts, model responses, and
  predictions.
- Pluggable model clients for API-based and local models.
- Extensible experiment architecture—each experiment is a self-contained module
  that produces a single log file per run.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the SciEntsBank experiment

```bash
python main.py
```

The script downloads a small slice of the SciEntsBank dataset, prompts the model
for grades, logs the interactions, and prints the resulting Cohen's Kappa score.

## Adding new experiments

1. Create a new module inside `experiments/` that subclasses
   `experiments.base.Experiment`.
2. Use `llm_judge.logging.get_run_logger` to obtain a dedicated log file for the
   run.
3. Implement your experiment logic in `run()` and return a scalar evaluation
   score.
