# CS577 LLM Judge

This project provides a modular framework for evaluating large language models as judges. It supports experiments that query both cloud-hosted APIs and local transformer models, loads datasets from Hugging Face or CSV files, and records experiment logs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the SciEntsBank experiment

The example experiment uses a mock LLM by default. Replace `MockLabelLLM` in `main.py` with an API or local model client to evaluate a real model.

### Available LLM clients

- `PurdueGenAIClient` connects to Purdue's GenAI OpenAI-compatible endpoint. Provide an API key via the constructor or the `PURDUE_GENAI_API_KEY` environment variable.
- `OllamaClient` streams prompts to a locally hosted Ollama model (defaults to `deepseek-r1:8b`) using the `ollama` CLI.

```bash
python main.py --sample-size 10 --log-dir logs
```

Logs are written to the specified directory and contain the full prompts, gold labels, and LLM responses for each datapoint. The command also prints the Cohen's kappa agreement score between the model predictions and ground-truth labels.
