# CS577 LLM Judge

## Project status
- Implements the SciEntsBank short-answer grading experiment with configurable sample sizes.
- Supports mock, OpenAI-compatible, Purdue RCAC GenAI, Hugging Face transformers pipeline, and Ollama backends.
- Captures structured experiment logs in CSV and JSON formats.
- Reports multiple agreement metrics: Cohen's kappa, accuracy, Pearson correlation, and Spearman correlation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment configuration
Remote backends expect API keys to be provided through environment variables. Copy `env.template.sh` to a safe, untracked
location (for example `~/.config/cs577_llm_judge.sh`), edit the exports, and source the file before running experiments:

```bash
cp env.template.sh ~/.config/cs577_llm_judge.sh
${EDITOR:-nano} ~/.config/cs577_llm_judge.sh
source ~/.config/cs577_llm_judge.sh
```

You can also keep a customized copy inside the repository under a name that is ignored by Git and `source` it when needed.

## Running the SciEntsBank experiment

```bash
python main.py --sample-size 25 --log-dir logs
```

### Selecting an LLM backend

The CLI exposes a `--llm-backend` flag to choose how prompts are answered. Additional arguments let you configure each backend:

- `mock` *(default)* – uses deterministic heuristic labels for development.
- `openai` – set `--model-name` (defaults to `gpt-3.5-turbo`) and optionally `--api-key` (defaults to `$OPENAI_API_KEY`).
- `rcac` – Purdue RCAC GenAI endpoint. Use `--model-name` (defaults to `llama3.1:latest`), `--rcac-base-url`, and
  optionally `--api-key` (defaults to `$RCAC_GENAI_API_KEY`).
- `local_pipeline` – Hugging Face transformers pipeline; requires `--model-name` and optionally `--pipeline-task`.
- `ollama` – local Ollama runtime; accepts `--model-name` (defaults to `deepseek-r1:8b`) and `--ollama-command` for custom
  invocation commands.

Example using the RCAC backend:

```bash
python main.py --llm-backend rcac --model-name llama3.1:latest --sample-size 50
```

### Log formats

Use `--log-format` to control the log outputs. The flag can be provided multiple times; the default is to emit both CSV and
JSON artifacts for every run:

```bash
python main.py --log-format csv --log-format json
```

Log files are written to the directory specified by `--log-dir`, with filenames derived from the experiment name and timestamp.

## Metrics

After each run the CLI prints a summary containing:

- **Cohen's kappa** – agreement beyond chance with gold labels.
- **Accuracy** – percentage of exact label matches.
- **Pearson correlation** – linear correlation between gold and predicted scores.
- **Spearman correlation** – rank correlation capturing monotonic relationships.

These metrics are also available for downstream analysis by loading the generated CSV or JSON logs.
