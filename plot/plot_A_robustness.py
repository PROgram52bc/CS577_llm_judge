
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_A_robustness(input_file, output_dir='plot/logs/plot_A_robustness'):
    """
    Generates Plot A: Robustness Drop (Sensitivity Analysis).

    This plot is a vertical bar chart showing the degradation of grading reliability
    (Cohen's Kappa or Accuracy) when the model faces noisy or imperfect input data.
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
    df = df[df['augmentation_type'].isin(allowed_labels)]

    # Identify baseline
    baseline_row = df[df['augmentation_type'] == 'baseline']
    if baseline_row.empty:
        raise ValueError("Could not find baseline data (label='baseline') within the filtered labels.")

    baseline_accuracy = baseline_row['accuracy'].iloc[0]

    # Sort data for plotting: Baseline first, then others alphabetically
    df_baseline = df[df['augmentation_type'] == 'baseline']
    df_augmentations = df[df['augmentation_type'] != 'baseline']
    df_augmentations = df_augmentations.sort_values('augmentation_type')
    df_sorted = pd.concat([df_baseline, df_augmentations])
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar chart
    sns.barplot(x='augmentation_type', y='accuracy', data=df_sorted, ax=ax, palette='viridis', hue='augmentation_type', dodge=False)

    # Horizontal line for baseline
    ax.axhline(y=baseline_accuracy, color='r', linestyle='--', linewidth=2, label=f'Baseline Accuracy ({baseline_accuracy:.2f})')

    # --- Aesthetics ---
    ax.set_title('Plot A: Robustness Drop (Sensitivity Analysis)', fontsize=16, weight='bold')
    ax.set_xlabel('Augmentation Type', fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Add a legend for the horizontal line
    handles, labels = ax.get_legend_handles_labels()
    # The ax.axhline adds a handle and label, which we can display.
    # If you also want to keep the bar labels (if any), you would manage them here.
    # For this plot, only the baseline horizontal line needs a label in the legend.
    ax.legend(handles=[handles[0]], labels=[labels[0]], loc='best')

    plt.tight_layout()

    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'plot_A_robustness.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Plot A: Robustness Drop Analysis.")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    args = parser.parse_args()
    
    plot_A_robustness(args.input_file)
