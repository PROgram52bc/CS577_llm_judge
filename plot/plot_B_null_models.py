

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_B_null_models(input_file, output_dir='plot/logs/plot_B_null_models'):
    """
    Generates Plot B: Model Response Distribution for Null Model Vulnerability.

    This plot is a stacked bar chart showing the percentage distribution of the model's
    predicted labels ('Correct', 'Partially Correct', etc.) for different adversarial input strategies.
    """
    # Read the pre-aggregated data from the summary stats file
    df = pd.read_csv(input_file)

    # --- Data Preprocessing ---
    # Filter for only the desired attack types for this plot
    experiment_labels = [
        'control_real_answer',
        'naive_solution',
        'naive_i_dont_know',
        'persuasive_ignore',
        'persuasive_this_response',
        'structured_json_injection'
    ]
    df = df[df['label'].isin(experiment_labels)]
    df['label'] = pd.Categorical(df['label'], categories=experiment_labels, ordered=True)
    df = df.sort_values('label')

    # Define the columns with the counts and their corresponding user-friendly names
    count_columns = [
        'count_correct',
        'count_partially_correct_incomplete',
        'count_contradictory',
        'count_irrelevant',
        'count_non_domain'
    ]
    plot_labels = ['Correct', 'Partially Correct', 'Contradictory', 'Irrelevant', 'Non-Domain']
    
    # Define colors for each label category
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#7f7f7f']
    
    # Extract the count data and convert to percentages
    counts = df[count_columns].values
    totals = counts.sum(axis=1)
    # Avoid division by zero if a row has no graded examples
    percentages = np.divide(counts, totals[:, np.newaxis], out=np.zeros_like(counts, dtype=float), where=totals[:, np.newaxis] != 0) * 100
    
    # Transpose the data for plotting (categories as rows, experiments as columns)
    data = percentages.T

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Clean up the experiment labels for the x-axis
    experiments_display_names = [name.replace('_', ' ').title() for name in experiment_labels]
    width = 0.6
    bottom = np.zeros(len(experiments_display_names))

    for i, label_name in enumerate(plot_labels):
        ax.bar(experiments_display_names, data[i], width, label=label_name, bottom=bottom, color=colors[i], alpha=0.9)
        bottom += data[i]

    # --- Aesthetics ---
    ax.set_ylabel('Percentage of Responses (%)')
    ax.set_title('Plot B: Model Response Distribution vs. Adversarial Attacks', fontsize=16, weight='bold')
    ax.tick_params(axis='x', rotation=25)
    ax.set_ylim(0, 105) # Set Y-axis to go up to 100%

    # Place legend outside to keep the chart clean
    ax.legend(title="Predicted Label", bbox_to_anchor=(1.04, 1), loc='upper left')

    plt.tight_layout()
    fig.subplots_adjust(right=0.75) # Adjust layout to make space for the legend

    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'plot_B_null_models.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Plot B: Null Model Vulnerability as a stacked bar chart.")
    parser.add_argument('input_file', type=str, help='Path to the summary_stats.csv file.')
    args = parser.parse_args()
    
    plot_B_null_models(args.input_file)
