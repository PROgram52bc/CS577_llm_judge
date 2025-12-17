
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re

def plot_C_consensus(input_file, output_dir='plot/logs/plot_C_consensus'):
    """
    Generates Plot C: The "Trust Curve" (Coverage vs. Accuracy).

    This plot is a line plot demonstrating the relationship between coverage 
    (100% - withdraw_rate) and accuracy. It shows how accuracy on the graded 
    subset changes as the model is allowed to abstain more or less.
    """
    # Read the data
    df = pd.read_csv(input_file)

    # --- Data Preprocessing ---
    # Filter for only labels starting with "consensus_"
    df = df[df['label'].astype(str).str.startswith('consensus_')]

    # Calculate Coverage
    df['coverage'] = (1 - df['withdraw_rate']) * 100
    
    # Use the 'label' column for consensus threshold
    if 'label' not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column for consensus thresholds.")
    # Strip the "consensus_" prefix before converting to numeric
    df['consensus_threshold'] = pd.to_numeric(df['label'].str.replace('consensus_', ''))
    
    # Sort by coverage for a clean line plot
    df_sorted = df.sort_values('coverage')

    if df_sorted.empty:
        raise ValueError("The processed dataframe is empty. Check input file and data processing steps.")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Line plot
    ax.plot(df_sorted['coverage'], df_sorted['accuracy'], marker='o', linestyle='-', color='b')

    # Add markers for each threshold
    for i, row in df_sorted.iterrows():
        ax.text(row['coverage'], row['accuracy'] + 0.005, f"{row['consensus_threshold']:.1f}", 
                ha='center', fontsize=9)

    # --- Aesthetics ---
    ax.set_title('Plot C: The "Trust Curve" (Coverage vs. Accuracy)', fontsize=16, weight='bold')
    ax.set_xlabel('Coverage (%)', fontsize=12)
    ax.set_ylabel('Accuracy (on Graded Subset)', fontsize=12)
    ax.set_xlim(df_sorted['coverage'].min() - 5, 105)
    ax.set_ylim(df_sorted['accuracy'].min() - 0.05, df_sorted['accuracy'].max() + 0.05)

    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'plot_C_consensus.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Plot C: The "Trust Curve".')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    args = parser.parse_args()
    
    plot_C_consensus(args.input_file)
