# CS577 LLM Judge

This repository hosts a lightweight evaluation harness for comparing large language models as automatic graders. The tooling centres on SciEntsBank-based experiments that ask an LLM to score short-answer responses and then compares those scores against ground-truth labels using a suite of correlation metrics.

## Current status

- **Experiments**:
  - `SciEntsBankKappaExperiment` loads a sample from the public SciEntsBank dataset, prompts an LLM to grade each example once, and computes Cohen's kappa, accuracy, Pearson correlation, and Spearman correlation along with basic coverage statistics.
  - `SciEntsBankConsensusExperiment` reuses the same dataset loader but performs *N* independent LLM generations for every example. Only labels that meet a configurable agreement threshold are retained; the remainder are logged as withdrawn and reported via a withdraw rate metric.
- **LLM clients**: Select between a deterministic mock grader, OpenAI-compatible APIs, Purdue's RCAC GenAI endpoint, a local Hugging Face transformers pipeline, or a local Ollama runtime.
- **Logging**: Structured run logs can be emitted as newline-delimited JSON, CSV, and/or plain-text. JSON and CSV outputs are enabled by default.

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

`OpenAIClient` reads `OPENAI_API_KEY`, while `RCACGenAIClient` reads `RCAC_GENAI_API_KEY`. You can also provide the key explicitly via the `--api-key` command-line flag. RCAC currently offers the chat models `llama3.1:latest`, `llama4:latest`, `qwen2.5:72b`, and `gpt-oss:120b`.

## Running the SciEntsBank experiments

The CLI exposes options for the experiment flavour, sample size, logging configuration, and backend selection. JSON and CSV logs are generated in the target directory by default.

```bash
python main.py \
    --sample-size 25 \
    --llm-backend mock \
    --log-dir logs
```

To enable the consensus workflow (three runs and a 66% agreement threshold by default):

```bash
python main.py \
    --experiment consensus \
    --consensus-runs 5 \
    --consensus-threshold 0.8 \
    --llm-backend rcac \
    --model-name llama4:latest
```

To run against a remote provider (using environment variables for credentials):

```bash
python main.py \
    --llm-backend openai \
    --model-name gpt-4o-mini \
    --sample-size 50
```

For local evaluation with a transformers pipeline:

```bash
python main.py \
    --llm-backend local-pipeline \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --pipeline-task text-generation
```

After the run completes the terminal will report every metric that was computed, including withdraw rate for consensus runs. Log files for each example are stored under the selected `--log-dir` with timestamped filenames.
