
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_E_complexity(input_file, output_dir='plot/logs/plot_E_complexity'):
    """
    Generates Plot E: Label Scheme Complexity (2-way vs 3-way vs 5-way).

    This plot compares key performance metrics (Accuracy, Cohen's Kappa, Spearman Correlation)
    between different grading schemes. It now includes error bars (representing standard
    deviation) if multiple data points for each scheme are present in the input file.
    """
    # Read the data
    df = pd.read_csv(input_file)

    # --- Data Preprocessing ---
    # Filter for the relevant experiment labels
    schemes = ['2way', '3way', '5way']
    df_filtered = df[df['label'].isin(schemes)]

    # Group by label scheme and calculate mean and standard deviation for each metric
    grouped = df_filtered.groupby('label')
    means = grouped[['accuracy', 'cohen_kappa', 'spearman_correlation']].mean().loc[schemes]
    stds = grouped[['accuracy', 'cohen_kappa', 'spearman_correlation']].std().loc[schemes].fillna(0) # Fill NaN with 0 if only one sample

    categories = ['2-Way Grading', '3-Way Grading', '5-Way Grading']
    accuracy_mean = means['accuracy'].tolist()
    kappa_mean = means['cohen_kappa'].tolist()
    spearman_mean = means['spearman_correlation'].tolist()

    accuracy_std = stds['accuracy'].tolist()
    kappa_std = stds['cohen_kappa'].tolist()
    spearman_std = stds['spearman_correlation'].tolist()

    x = np.arange(len(categories))
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # --- Metric 1: Bars (Primary Y-Axis) ---
    ax1.set_xlabel('Label Scheme', fontsize=12)
    ax1.set_ylabel('Accuracy', color='tab:blue', fontweight='bold', fontsize=12)
    # Plot bars with error bars for standard deviation
    ax1.bar(x, accuracy_mean, yerr=accuracy_std, capsize=5,
            color='lightblue', alpha=0.8, label='Accuracy', width=0.4,
            ecolor='darkblue', zorder=3) # Highest zorder for blue bars
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)

    # --- Metrics 2 & 3: Lines/Markers (Secondary Y-Axis) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Cohen's Kappa / Spearman Correlation", color='black', fontweight='bold', fontsize=12)

    # Metric 2: Cohen's Kappa (Line Plot with Error Bars)
    ax2.errorbar(x, kappa_mean, yerr=kappa_std, color='tab:red', marker='o',
                 linewidth=2, label="Cohen's Kappa", markersize=8, capsize=5, zorder=2) # Middle zorder for red

    # Metric 3: Spearman Correlation (Scatter/Diamond Plot with Error Bars)
    # Use errorbar with no line for a scatter-like plot
    ax2.errorbar(x, spearman_mean, yerr=spearman_std, color='green', marker='D',
                 linestyle='None', label='Spearman Correlation', markersize=8, capsize=5, zorder=1) # Lowest zorder for green

    # --- Combined Legend ---
    # Manually gather handles from both axes for a unified legend
    # For errorbar, the handle is a container. We need to grab the main line/marker from it.
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

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
