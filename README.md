# CS577 LLM Judge

This project provides a modular framework for evaluating large language models as judges. It supports experiments that query both cloud-hosted APIs and local transformer models, loads datasets from Hugging Face or CSV files, and records experiment logs.

The current implementation focuses on a SciEntsBank short-answer grading benchmark and includes:

* A pluggable LLM client architecture with mock, OpenAI-compatible, and Purdue RCAC GenAI backends.
* Structured experiment logging with selectable CSV and JSONL outputs.
* Multiple agreement metrics, including Cohen's kappa, accuracy, Pearson correlation, and Spearman correlation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Copy `env.template.sh` to a safe location (ideally outside of the repository) and populate it with your API keys:

```bash
cp env.template.sh ~/.config/llm-judge-env.sh
```

Edit the copied file to include valid credentials, then source it before running experiments:

```bash
source ~/.config/llm-judge-env.sh
```

Keeping the populated file outside of the repository helps avoid accidentally committing secrets, while still letting you load the necessary environment variables with a single `source` command.

## Running the SciEntsBank experiment

The example experiment uses the mock labeler by default, which requires no external services. You can switch to an API backend at runtime with command-line flags:

```bash
python main.py --sample-size 10 --log-dir logs \
    --llm-backend openai --model-name gpt-4o-mini
```

If the required environment variables are set, the command above will invoke the specified OpenAI-compatible model. Replace `openai` with `rcac-genai` to use the Purdue RCAC GenAI service instead. You can override the API key passed to these backends with `--api-key`, otherwise the key is read from the environment.

Logs are written to the specified directory in both newline-delimited JSON (`.jsonl`) and CSV (`.csv`) formats by default. Use `--log-format` to select one or both formats explicitly:

```bash
python main.py --log-format json --log-format csv
```

After the run completes, the command-line output reports Cohen's kappa, accuracy, Pearson correlation, and Spearman correlation between the model predictions and gold labels.
