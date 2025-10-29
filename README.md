# CS577 LLM Judge

CS577 LLM Judge is an experimentation harness for evaluating the quality of large language models acting as automated graders. The current focus is the SciEntsBank short-answer dataset, but the framework is structured so that additional datasets, experiments, and model backends can be added with minimal changes.

## Project status

* **Experiment coverage:** The `SciEntsBankKappaExperiment` runs end-to-end grading evaluations and now reports Cohen's kappa, Pearson correlation, Spearman correlation, and simple accuracy against the gold labels.
* **LLM backends:** The CLI can select among mock responses, OpenAI-compatible APIs, Purdue's RCAC GenAI endpoint, local Ollama installs, or Hugging Face `transformers` pipelines. API keys are sourced from environment variables by default.
* **Logging:** Each run writes structured logs in both JSON Lines and CSV formats (configurable via the CLI). Every record includes the prompt context, the raw LLM response, and the derived label.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure credentials by copying `env.example.sh` either inside or outside the repository, editing the secrets, and sourcing the file before running experiments:

```bash
cp env.example.sh env.local.sh
# edit env.local.sh to fill in your keys
source env.local.sh
```

## Running the SciEntsBank experiment

The CLI defaults to the deterministic mock model which requires no external dependencies. Replace the backend to target a real model:

```bash
# Run 25 examples with the OpenAI backend and CSV+JSON logs
python main.py \
  --sample-size 25 \
  --llm-backend openai \
  --openai-model gpt-4o-mini
```

Key CLI options:

| Option | Description |
| --- | --- |
| `--llm-backend {mock,openai,rcac,ollama,local}` | Choose the model runtime. |
| `--api-key` | Override the API key instead of relying on environment variables. |
| `--log-format {json,csv}` | Select one or both log formats (defaults to both when omitted). |
| `--log-dir` | Directory to store structured run logs. |

Each run prints the metric summary to stdout. Structured logs can be found under the supplied `--log-dir` with timestamped filenames for downstream analysis.
