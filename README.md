# CS577 LLM Judge

This repository hosts a lightweight evaluation harness for comparing large language models as automatic graders. The current focus is a SciEntsBank-based experiment that asks an LLM to score short-answer responses and then compares those scores against ground-truth labels using a suite of correlation metrics.

## Experiments

Two SciEntsBank-based experiments share the same dataset loading and logging infrastructure:

- **Single-pass kappa (`--experiment single`)** – runs `SciEntsBankKappaExperiment`, which issues one LLM grading call per sample and reports Cohen's kappa, accuracy, Pearson correlation, and Spearman correlation across the selected label schemes.
- **Consensus grading (`--experiment consensus`)** – runs `SciEntsBankConsensusExperiment`, which performs *N* independent LLM calls (configurable via `--consensus-runs`). Predictions are only kept when at least the configured proportion of runs agree (`--consensus-threshold`). Samples without sufficient agreement are marked as `withdrawn` in the per-example logs, excluded from accuracy and correlation metrics, and summarized through additional metrics such as `withdrawn_examples` and `withdraw_rate`.

Both experiments emit the same structured logging outputs (JSON, CSV, and/or text) with per-sample metadata and summary metrics at the end of the run.

## Data Augmentation Flags

Flags for augmenting the student answer in the SciEntsBankExperiments

- **--ocr-augment** - randomly replaces characters with optically similar characters such as O with 0
- **--typos** - randomly inserts typos where letters may be replaced with letters in the same area of the keyboard
- **--non-influential-words** - randomly adds words that won't influence the meaning of the sentence
- **--hyphens** - randomly adds hyphens before characters
- **--non-unicode** - randomly inserts non-unicode characters
- **--substitute-synonyms** - randomly substitutes words with synonyms
- **--paraphrase** - paraphrases text by translating it to German and back to English

## LLM clients and models

- **Mock** – deterministic label generator for testing the harness end-to-end.
- **OpenAI-compatible** – specify `--model-name` (for example `gpt-4o-mini`).
- **RCAC GenAI** – select Purdue's hosted models via `--rcac-model` or `--model-name`. Available shortcuts include `llama3.1:latest`, `llama4:latest`, `qwen2.5:72b`, and `gpt-oss:120b`.
- **Local transformers pipeline** – run Hugging Face models with `--llm-backend local-pipeline` and provide `--model-name` and `--pipeline-task`.
- **Ollama** – evaluate models served by a local Ollama runtime.

Structured run logs include a `withdrawn` field whenever the consensus experiment suppresses an answer and still capture all raw model outputs via the `llm_response`/`llm_responses` fields.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure environment variables

Sensitive credentials are expected to come from environment variables. Copy `env.template.sh` to an untracked file (for example `env.local.sh`), update the placeholder values, and source the file before running an experiment:

```bash
cp env.template.sh env.local.sh
${EDITOR:-nano} env.local.sh  # update API keys
source env.local.sh
```

`OpenAIClient` reads `OPENAI_API_KEY`, while `RCACGenAIClient` reads `RCAC_GENAI_API_KEY`. You can also provide the key explicitly via the `--api-key` command-line flag.

## Running the SciEntsBank experiments

The CLI exposes options for the sample size, logging configuration, backend selection, and experiment strategy. JSON and CSV logs are generated in the target directory by default.

```bash
python main.py \
    --sample-size 25 \
    --llm-backend mock \
    --experiment single \
    --log-dir logs
```

To run against a remote provider (using environment variables for credentials):

```bash
python main.py \
    --llm-backend openai \
    --model-name gpt-4o-mini \
    --experiment consensus \
    --consensus-runs 5 \
    --consensus-threshold 0.8 \
    --sample-size 50
```

For local evaluation with a transformers pipeline:

```bash
python main.py \
    --llm-backend local-pipeline \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --pipeline-task text-generation
```

After the run completes the terminal will report every metric that was computed. Log files for each example are stored under the selected `--log-dir` with timestamped filenames.
