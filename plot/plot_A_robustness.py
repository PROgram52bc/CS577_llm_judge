

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_A_robustness(input_file, output_dir='plot/logs/plot_A_robustness'):
    """
    Generates Plot A: Robustness Drop (Sensitivity Analysis).

    This plot compares key performance metrics (Accuracy, Cohen's Kappa, Spearman Correlation)
    across different data augmentations. It includes error bars (representing standard
    deviation) if multiple data points for each augmentation type are present.
    """
    # Read the data
    df = pd.read_csv(input_file)

    # --- Data Preprocessing ---
    # Use the 'label' column for augmentation type
    df.rename(columns={'label': 'augmentation_type'}, inplace=True)

    # Filter for only the desired augmentation types
    allowed_labels = [
        'baseline', 'ocr', 'typo', 'non_influential', 'hyphen',
        'non_unicode', 'synonym', 'paraphrase'
    ]
    df_filtered = df[df['augmentation_type'].isin(allowed_labels)]

    # Group by augmentation type and calculate mean and standard deviation
    grouped = df_filtered.groupby('augmentation_type')
    metrics = ['accuracy', 'cohen_kappa', 'spearman_correlation']
    means = grouped[metrics].mean()
    stds = grouped[metrics].std().fillna(0)

    # Order the results: baseline first, then alphabetically
    ordered_labels = ['baseline'] + sorted([l for l in allowed_labels if l != 'baseline'])
    means = means.loc[ordered_labels]
    stds = stds.loc[ordered_labels]

    categories = [label.replace('_', ' ').title() for label in ordered_labels]
    accuracy_mean = means['accuracy'].tolist()
    kappa_mean = means['cohen_kappa'].tolist()
    spearman_mean = means['spearman_correlation'].tolist()

    accuracy_std = stds['accuracy'].tolist()
    kappa_std = stds['cohen_kappa'].tolist()
    spearman_std = stds['spearman_correlation'].tolist()

    x = np.arange(len(categories))
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Metric 1: Bars (Primary Y-Axis) ---
    ax1.set_xlabel('Augmentation Type', fontsize=12)
    ax1.set_ylabel('Accuracy', color='tab:blue', fontweight='bold', fontsize=12)
    ax1.bar(x, accuracy_mean, yerr=accuracy_std, capsize=5,
            color='lightblue', alpha=0.8, label='Accuracy', width=0.6,
            ecolor='darkblue', zorder=3)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')

    # --- Metrics 2 & 3: Lines/Markers (Secondary Y-Axis) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Cohen's Kappa / Spearman Correlation", color='black', fontweight='bold', fontsize=12)

    ax2.errorbar(x, kappa_mean, yerr=kappa_std, color='tab:red', marker='o',
                 linewidth=2, label="Cohen's Kappa", markersize=8, capsize=5, zorder=2)

    ax2.errorbar(x, spearman_mean, yerr=spearman_std, color='green', marker='D',
                 linestyle='None', label='Spearman Correlation', markersize=8, capsize=5, zorder=1)

    ax2.set_ylim(0, 1.1)

    # --- Combined Legend ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=10)


    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'plot_A_robustness.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Plot A: Robustness Drop Analysis.")
    parser.add_argument('input_file', type=str, help='Path to the consolidated input CSV file.')
    args = parser.parse_args()
    
    plot_A_robustness(args.input_file)
