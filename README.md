# CS577 LLM Judge

This repository hosts a lightweight evaluation harness for comparing large language models as automatic graders. The current focus is a SciEntsBank-based experiment that asks an LLM to score short-answer responses and then compares those scores against ground-truth labels.

## Installation

1.  Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Sensitive credentials (like API keys) are managed via environment variables.

1.  Copy the template file:
    ```bash
    cp env.template.sh env.local.sh
    ```
2.  Edit the new file (`env.local.sh`) to add your API keys.
3.  Source the file in your terminal before running experiments:
    ```bash
    source env.local.sh
    ```
    - `OpenAIClient` reads `OPENAI_API_KEY`.
    - `RCACGenAIClient` reads `RCAC_GENAI_API_KEY`.
    - `RANDOM_SEED` can also be set here for reproducible random sampling.

---

## The Evaluation Suite: Plots A-E

This harness is designed to run a suite of five experiments, each corresponding to a plot that answers a key research question about the LLM judge's behavior. The `run_plot_*.sh` scripts in the root directory are pre-configured to generate the data for each plot.

*   **Experiment A (Robustness Drop):** The goal is to measure how much the model's grading accuracy is affected by common "noise" in the student answers, such as typos, OCR errors, or paraphrasing. It tests the model's stability and semantic understanding.

*   **Experiment B (Null Model Vulnerability):** The goal is to test the model's security and vulnerability to adversarial attacks. It measures how well the model rejects junk inputs, from simple non-answers to malicious attempts at prompt injection.

*   **Experiment C (The "Trust Curve"):** The goal is to show the trade-off between accuracy and coverage. It demonstrates how letting the model abstain from grading when uncertain (by requiring a consensus among multiple "voters") can increase the accuracy of the remaining graded answers.

*   **Experiment D (Confusion Matrix):** The goal is to diagnose specific error patterns in the model's grading on a detailed, multi-category scale. It helps identify whether the model is systematically too lenient, too harsh, or frequently confuses certain categories (e.g., 'Partial' vs. 'Incorrect').

*   **Experiment E (Label Scheme Complexity):** The goal is to quantify how the complexity of the grading rubric affects the model's performance. It compares grading accuracy across simple (2-way) and more nuanced (3-way, 5-way) rubrics to measure the impact of task difficulty.

---

## Running Experiments

The `main.py` script is the entry point for all experiments. It is controlled by command-line arguments, which are bundled in the `run_plot_*.sh` scripts for convenience.

### Core Experiment Types

You can configure the experiment strategy with the `--experiment` flag:

-   `--experiment single`: Performs a single evaluation pass on each sample. This is used for generating data for Plots A, B, D, and E.
-   `--experiment consensus_curve`: The recommended and most efficient method for generating the Plot C trust curve. It collects votes once and internally analyzes them against a list of thresholds.
-   `--experiment consensus`: The original, less efficient method that tests only one consensus threshold per run.

### Data Augmentation

Several flags can be used to apply augmentations to the student's answer, primarily for Plot A:
`--ocr-augment`, `--typos`, `--non-influential-words`, `--hyphens`, `--non-unicode`, `--substitute-synonyms`, `--paraphrase`.

### LLM Backends

You can specify the LLM backend with `--llm-backend` and `--model-name`.
-   **`mock`**: A deterministic label generator for testing.
-   **`openai`**: For any OpenAI-compatible API.
-   **`rcac`**: For Purdue's hosted models (`llama3.1:latest`, `qwen2.5:72b`, etc.).
-   **`local-pipeline`**: For local Hugging Face `transformers` models.
-   **`ollama`**: For models served by a local Ollama runtime.

### Example Commands

While it's recommended to use the `run_plot_*.sh` scripts, you can also run `main.py` directly.

**Example 1: Generate data for the Plot C Trust Curve**
```bash
python main.py \
    --experiment consensus_curve \
    --llm-backend rcac \
    --model-name qwen2.5:72b \
    --sample-size 200 \
    --consensus-runs 10 \
    --consensus-thresholds 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0
```

**Example 2: Generate data for one bar in the Plot A Robustness test (e.g., with typos)**
```bash
python main.py \
    --experiment single \
    --llm-backend openai \
    --model-name gpt-4o-mini \
    --sample-size 40 \
    --typos \
    --label "typo" \
    --shuffle-seed 42
```
After runs are complete, the terminal will report summary metrics, and detailed log files can be found in the directory specified by `--log-dir` (defaults to `logs/`).
