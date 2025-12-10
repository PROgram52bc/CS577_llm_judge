# Experimental Visualization Plan

This document outlines the visualization strategy for the four experimental scripts. Each plot is designed to address a specific research question regarding the robustness, security, and reliability of the LLM-as-a-judge system.

---

## Plot A: Robustness Drop (Sensitivity Analysis)
**Data Generation:** `run_plot_A_robustness.sh`  

**Plot Generation:** [TODO: Run this command to produce the plot, should take input file as argument]

**Goal:** Quantify the degradation of grading reliability when the model faces noisy or imperfect input data.

* **Plot Type:** Vertical Bar Chart
* **X-Axis:** Augmentation Type
    * *Categories:* Baseline (Original), OCR Error, Typos, Non-Influential Words, Hyphens, Non-Unicode, Synonyms, Paraphrase.
* **Y-Axis:** Agreement Metric (Cohen’s Kappa or Accuracy).
* **Visual Structure:**
    * Sort bars with **Baseline** on the far left.
    * Add a **horizontal dashed line** extending from the Baseline bar across the chart. This provides a visual anchor to easily see the magnitude of the "drop" for each augmentation.
* **Interpretation:** Significant drops below the baseline indicate the model is brittle and relies on surface-level tokens rather than semantic meaning.

---

## Plot B: Null Model Vulnerability (Adversarial Attack)
**Data Generation:** `run_plot_B_null_models.sh`  

**Plot Generation:** [TODO: Run this command to produce the plot, should take input file as argument]

**Goal:** Validate the findings of [Zheng et al., 2025] by testing if the model assigns high scores to vacuous or adversarial answers.

* **Plot Type:** Grouped Bar Chart (or Box Plot)
* **X-Axis:** Input Strategy
    * *Control:* Real Student Answer.
    * *Naive Attacks:* "Solution", "I don't know".
    * *Persuasive Attacks:* "Ignore directions...", "This response offers...".
    * *Structured Attack:* JSON/Prompt Injection.
* **Y-Axis:** Average Predicted Score (Scale: 1.0 – 5.0).
* **Visual Structure:**
    * Group the bars by attack category.
    * Use distinct colors for **Control** vs. **Attacks**.
* **Interpretation:** If the "Structured Attack" or "Persuasive" bars are comparable to the "Real Student Answer" bar (e.g., > 1.0), the model is vulnerable to content-independent exploits.

---

## Plot C: The "Trust Curve" (Coverage vs. Accuracy)
**Data Generation:** `run_plot_C_consensus.sh`  

**Plot Generation:** [TODO: Run this command to produce the plot, should take input file as argument]

**Goal:** Demonstrate the utility of the consensus mechanism. We hypothesize that allowing the model to abstain (withdraw) from uncertain cases increases the reliability of the remaining grades.

* **Plot Type:** Line Plot with Markers
* **X-Axis:** Coverage Percentage (%)
    * *Formula:* % - Withdraw Rate
    * *Scale:* 0% to 100%.
* **Y-Axis:** Accuracy (on the graded subset).
* **Visual Structure:**
    * Plot points corresponding to consensus thresholds (0.5, 0.6, ..., 1.0).
    * The line should ideally slope **upwards to the left**.
* **Interpretation:** A negative slope (higher accuracy at lower coverage) justifies the use of "human-in-the-loop" systems where the model only grades when confident.

---

## Plot D: Confusion Matrix (Error Analysis)
**Data Generation:** `run_plot_D_confusion.sh`

**Plot Generation:** [TODO: Run this command to produce the plot, should take input file as argument]

**Goal:** Diagnose specific grading biases. Does the model punish valid answers (harshness) or reward incorrect ones (hallucination)?

* **Plot Type:** Heatmap
* **X-Axis:** Model Predicted Score (1, 2, 3, 4, 5).
* **Y-Axis:** Human Gold Label (1, 2, 3, 4, 5).
* **Visual Structure:**
    * Use a sequential color palette (e.g., Blues). Darker cells indicate higher frequency.
    * **Diagonal (1,1 to 5,5):** Represents correct agreement.
    * **Upper-Right Triangle:** Model Score > Human Score (Leniency).
    * **Lower-Left Triangle:** Model Score < Human Score (Harshness).
* **Interpretation:** High density in the off-diagonal regions reveals specific failure modes (e.g., the model frequently confusing a score of "2" for a "3").
