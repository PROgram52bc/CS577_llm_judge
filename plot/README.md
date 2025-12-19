# Experimental Visualization Plan

This document outlines the visualization strategy for the five experimental scripts. Each plot is designed to address a specific research question regarding the robustness, security, and reliability of the LLM-as-a-judge system.

---

## Plot A: Robustness Drop (Sensitivity Analysis)
**Data Generation:** `run_plot_A_robustness.sh`  
The input CSV for this plot must contain a `label` column with the following strings:
`baseline`, `ocr`, `typo`, `non_influential`, `hyphen`, `non_unicode`, `synonym`, `paraphrase`.

**Plot Generation:** `python plot_A_robustness.py <path_to_input_csv>`

**Goal:** Quantify the degradation of grading reliability when the model faces noisy or imperfect input data.

* **Plot Type:** Vertical Bar Chart
* **X-Axis:** Augmentation Type
    * *Categories:* Baseline (Original), OCR Error, Typos, Non-Influential Words, Hyphens, Non-Unicode, Synonyms, Paraphrase.
* **Y-Axis:** Accuracy.
* **Visual Structure:**
    * A simple bar chart where each bar represents the model's accuracy for a specific augmentation type.
    * A **horizontal dashed line** extends from the Baseline bar across the chart. This provides a visual anchor to easily see the magnitude of the accuracy "drop" for each augmentation.
* **Interpretation:** Significant drops below the baseline indicate the model is brittle and relies on surface-level tokens rather than semantic meaning.

---

## Plot B: Null Model Vulnerability (Adversarial Attack)
**Data Generation:** `run_plot_B_null_models.sh` (The results are read from the main `summary_stats.csv` file).
The input CSV for this plot must contain:
1.  A `label` column with the experiment names (e.g., `control_real_answer`, `naive_solution`, etc.).
2.  The five count columns: `count_correct`, `count_partially_correct_incomplete`, `count_contradictory`, `count_irrelevant`, `count_non_domain`.

**Plot Generation:** `python plot_B_null_models.py <path_to_summary_stats.csv>`

**Goal:** Analyze the model's response distribution when faced with vacuous or adversarial answers.

* **Plot Type:** Stacked Bar Chart
* **X-Axis:** Input Strategy (e.g., Control, Naive Solution, Persuasive Ignore, etc.).
* **Y-Axis:** Percentage of Responses (%).
* **Visual Structure:**
    * Each bar represents an input strategy and sums to 100%.
    * Each bar is segmented by color, with each color representing a predicted label category (e.g., Correct, Irrelevant).
    * A legend identifies the color for each predicted label.
* **Interpretation:** For a robust model, the "control" bar should show a high percentage of 'Correct' or 'Partially Correct' responses, while the attack bars should show a very high percentage of 'Irrelevant' or 'Non-Domain' responses, indicating the model successfully identified and rejected the junk inputs.

---

## Plot C: The "Trust Curve" (Coverage vs. Accuracy)
**Data Generation:** `run_plot_C_consensus.sh`  
The input CSV for this plot must contain a `label` column where each label is a string representing the consensus threshold used for that data point (e.g., `"consensus_0.5"`, `"consensus_0.6"`).

**Plot Generation:** `python plot_C_consensus.py <path_to_input_csv>`

**Goal:** Demonstrate the utility of the consensus mechanism to trade coverage for accuracy.

* **Plot Type:** Scatter Plot with Color Gradient
* **X-Axis:** Coverage Percentage (%) (Logarithmic Scale)
    * *Formula:* 100% - Withdraw Rate
* **Y-Axis:** Accuracy (on the graded subset).
* **Visual Structure:**
    * A scatter plot where each point represents a different consensus threshold.
    * The color of each point corresponds to the consensus threshold level, indicated by a colorbar legend.
* **Interpretation:** A negative slope (higher accuracy at lower coverage, typically shown on the left side of the plot) justifies the use of "human-in-the-loop" systems where the model only grades when confident.

---

## Plot D: Confusion Matrix (Error Analysis)
**Data Generation:** `run_plot_D_confusion.sh`

**Plot Generation:** `python plot_D_confusion.py <path_to_input_csv>`

**Goal:** Diagnose specific grading biases when using a multi-category rubric.

* **Plot Type:** Heatmap
* **X-Axis:** Model Predicted Score (1, 2, 3, 4, 5).
* **Y-Axis:** Human Gold Label (1, 2, 3, 4, 5).
* **Visual Structure:**
    * A heatmap where darker cells indicate higher frequency.
    * **Diagonal (1,1 to 5,5):** Represents correct agreement.
    * **Upper-Right Triangle:** Model Score > Human Score (Leniency).
    * **Lower-Left Triangle:** Model Score < Human Score (Harshness).
* **Interpretation:** High density in the off-diagonal regions reveals specific failure modes (e.g., the model frequently confusing a score of "2" for a "3").

---

## Plot E: Label Scheme Complexity
**Data Generation:** `run_plot_E_complexity.sh`

**Plot Generation:** `python plot_E_complexity.py <path_to_input_csv>`

**Goal:** Measure how classification complexity affects the model's alignment with expert judgment.

* **Plot Type:** Combined Bar and Line Chart (with dual Y-axes).
* **X-Axis:** Label Scheme (`2-Way`, `3-Way`, `5-Way`).
* **Y-Axis (Primary):** Accuracy (plotted as bars).
* **Y-Axis (Secondary):** Cohen's Kappa and Spearman Correlation (plotted as lines/markers).
* **Visual Structure:**
    * Bars show the primary accuracy for each label scheme.
    * Two overlaid lines with distinct markers show the trend for other agreement metrics across the different schemes.
* **Interpretation:** Shows the performance degradation as the number of grading categories increases, quantifying the difficulty of more nuanced classification tasks.

