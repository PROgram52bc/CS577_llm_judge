
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np

def plot_B_null_models(input_file, output_dir='plot/logs/plot_B_null_models'):
    """
    Generates Plot B: Null Model Vulnerability (Adversarial Attack).

    This plot is a grouped bar chart showing the average predicted score for different
    input strategies, including real answers and various adversarial attacks.
    """
    # Read the data
    df = pd.read_csv(input_file)

    # --- Data Preprocessing ---
    # The CSV is now pre-aggregated. We expect 'label' and 'accuracy' columns.
    if 'label' not in df.columns or 'accuracy' not in df.columns:
        raise ValueError("Input CSV must contain 'label' and 'accuracy' columns.")
    
    df.rename(columns={'label': 'attack_type', 'accuracy': 'acc'}, inplace=True)

    # Filter for only the desired attack types
    category_order = [
        'control_real_answer',
        'naive_solution',
        'naive_i_dont_know',
        'persuasive_ignore',
        'persuasive_this_response',
        'structured_json_injection'
    ]
    df = df[df['attack_type'].isin(category_order)]

    # Sort for consistent plotting order
    df['attack_type'] = pd.Categorical(df['attack_type'], categories=category_order, ordered=True)
    df_sorted = df.sort_values('attack_type')


    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create the bar plot
    sns.barplot(x='attack_type', y='acc', data=df_sorted, ax=ax, palette='plasma', hue='attack_type', dodge=False)

    # --- Aesthetics ---
    ax.set_title('Plot B: Null Model Vulnerability (Adversarial Attack)', fontsize=16, weight='bold')
    ax.set_xlabel('Input Strategy', fontsize=12)
    ax.set_ylabel('Average Predicted Score (1-5)', fontsize=12)
    ax.set_ylim(0, 1.2) # Scores are 0-1
    ax.tick_params(axis='x', rotation=45)
    
    # Add text labels for scores on bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    va='center', xytext=(0, 9), 
                    textcoords='offset points')

    plt.tight_layout()

    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'plot_B_null_models.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Plot B: Null Model Vulnerability Analysis.")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file (format similar to plot A/C).')
    args = parser.parse_args()
    
    plot_B_null_models(args.input_file)
