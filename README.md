# CS577 LLM Judge

This project provides a modular framework for running "LLM-as-a-judge" experiments.
It supports both local and cloud-based model integrations, flexible data loading, and
structured logging of evaluation runs.

## Features

- **LLM abstraction** with interchangeable backends, including a local rule-based
  implementation for offline testing and an OpenAI Chat API wrapper for cloud models.
- **Modular data loaders** that support Hugging Face datasets via the `datasets`
  library and local CSV files.
- **Experiment registry** that makes it straightforward to add new experiments while
  ensuring each experiment produces its own timestamped JSONL log file.
- **Metrics utilities** that currently include Cohen's Kappa for comparing LLM
  judgments with ground-truth labels.

## Installation

Create a virtual environment and install the required Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running an Experiment

The `main.py` entry point runs registered experiments. By default it executes the
SciEntsBank grading experiment using the local rule-based model:

```bash
python main.py
```

To select a different experiment, pass the `--experiment` flag.

Each run produces a JSONL log file in the `logs/` directory containing the prompts,
model responses, and predictions for every evaluated datapoint. The experiment also
prints the Cohen's Kappa score to standard output.

## Adding New Experiments

1. Create a new module in `llm_judge/experiments/` that subclasses
   `Experiment` and register it with the decorator from
   `llm_judge.experiments.registry`.
2. Use the shared data loading and logging utilities to keep code focused on the
   experiment logic.
3. Return a scalar metric from the `run` method so it can be surfaced in `main.py`.

## Cloud Model Integration

To use the OpenAI client, install the `openai` package and set the
`OPENAI_API_KEY` environment variable. Instantiate `OpenAIChatClient` and pass it to
an experiment instead of the local rule-based model.

