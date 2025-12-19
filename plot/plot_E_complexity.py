
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_E_complexity(input_file, output_dir='plot/logs/plot_E_complexity'):
    """
    Generates Plot E: Label Scheme Complexity (2-way vs 5-way).

    This plot compares key performance metrics (Accuracy, Cohen's Kappa, Spearman Correlation)
    between a 2-way and a 5-way grading scheme to show how classification
    complexity affects model performance.
    """
    # Read the data
    df = pd.read_csv(input_file)

    # --- Data Preprocessing ---
    # Filter for only the '2way', '3way', and '5way' experiment labels
    df_filtered = df[df['label'].isin(['2way', '3way', '5way'])]
    if len(df_filtered) != 3:
        raise ValueError("Input CSV must contain exactly three rows with labels '2way', '3way', and '5way'.")

    # Sort by the label to ensure consistent order
    df_sorted = df_filtered.set_index('label').loc[['2way', '3way', '5way']].reset_index()

    categories = ['2-Way Grading', '3-Way Grading', '5-Way Grading']
    accuracy = df_sorted['accuracy'].tolist()
    cohen_kappa = df_sorted['cohen_kappa'].tolist()
    spearman_corr = df_sorted['spearman_correlation'].tolist()

    x = np.arange(len(categories))
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- Metric 1: Bars (Primary Y-Axis) ---
    ax1.set_xlabel('Label Scheme')
    ax1.set_ylabel('Accuracy', color='tab:blue', fontweight='bold')
    ax1.bar(x, accuracy, color='lightblue', alpha=0.7, label='Accuracy', width=0.4)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)

    # --- Metrics 2 & 3: Lines/Markers (Secondary Y-Axis) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Cohen's Kappa / Spearman Correlation", color='black', fontweight='bold')

    # Metric 2: Cohen's Kappa (Line Plot)
    line, = ax2.plot(x, cohen_kappa, color='tab:red', marker='o', linewidth=2,
                     label="Cohen's Kappa", markersize=8)

    # Metric 3: Spearman Correlation (Scatter/Diamond Plot)
    scatter = ax2.scatter(x, spearman_corr, color='green', marker='D', s=80,
                          label='Spearman Correlation', zorder=10)

    ax2.set_ylim(0, 1.1)

    # --- Combined Legend ---
    # Manually gather handles from both axes for a unified legend
    bars_handle = ax1.get_legend_handles_labels()[0][0]
    ax1.legend([bars_handle, line, scatter],
               ['Accuracy', "Cohen's Kappa", 'Spearman Correlation'],
               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.title('Plot E: Performance vs. Label Scheme Complexity', fontsize=16, weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)


    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'plot_E_complexity.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Plot E: Label Scheme Complexity.")
    parser.add_argument('input_file', type=str, help='Path to the consolidated input CSV file.')
    args = parser.parse_args()

    plot_E_complexity(args.input_file)
