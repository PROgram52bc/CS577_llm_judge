# CS577 LLM Judge

This project provides a modular framework for running LLM-as-a-judge experiments. It
supports both local heuristics and remote API models, modular data loading, and a
logging system that records every prompt/response pair.

## Features

- **Data loading** via the Hugging Face `datasets` library or CSV files
- **Model abstraction** with interchangeable judge implementations
- **Experiment registry** so new experiments can be added without modifying the core
  runner
- **Structured logging** with one JSONL log file per experiment run containing the
  full LLM responses
- **Cohen's kappa scoring** for the included SciEntsBank grading experiment

## Running the minimal experiment

```bash
pip install -r requirements.txt  # optional if dependencies are missing
python main.py --limit 16 --split validation
```

The command downloads a subset of the `nkazi/SciEntsBank` dataset, grades the
examples with the default rule-based judge, logs the results under the `logs/`
folder, and prints the resulting Cohen's kappa score.

To try a remote API model, instantiate `OpenAIJudge` (requires the `openai`
package and `OPENAI_API_KEY`) and pass it to the experiment runner.

## Adding new experiments

1. Implement a function `run_experiment(logger, data_loader, judge)` in a new module
   under `llm_judge/experiments/`.
2. Register the experiment by importing the module and calling
   `registry.register("name", run_experiment)`.
3. Ensure the experiment writes to the provided logger exactly once and returns a
   scalar metric.

## Project structure

```
llm_judge/
├── data/               # dataset loaders
├── experiments/        # experiment implementations & registry
├── logging/            # logging utilities
└── models/             # model interfaces and implementations
```
