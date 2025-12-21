

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re

def plot_C_consensus(input_file, output_dir='plot/logs/plot_C_consensus'):
    """
    Generates Plot C: The "Trust Curve" (Coverage vs. Accuracy) for multiple label schemes.

    This plot shows the relationship between coverage and accuracy for 2-way, 3-way,
    and 5-way label schemes, using different markers for each.
    """
    # Read the data from the summary stats file
    df = pd.read_csv(input_file)

    # --- Data Preprocessing ---
    # Filter for only consensus experiment data
    df_consensus = df[df['label'].astype(str).str.startswith('consensus_')].copy()

    if df_consensus.empty:
        raise ValueError("No consensus data found in the input file. Make sure labels start with 'consensus_'.")

    # Calculate Coverage and extract numeric threshold
    df_consensus['coverage'] = (1 - df_consensus['withdraw_rate']) * 100
    df_consensus['consensus_threshold'] = pd.to_numeric(df_consensus['label'].str.replace('consensus_', ''))
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define markers and z-order for each scheme
    marker_map = {
        '5way': {'marker': 'o', 'label': '5-Way Scheme', 'zorder': 2},
        '3way': {'marker': 's', 'label': '3-Way Scheme', 'zorder': 3},
        '2way': {'marker': '^', 'label': '2-Way Scheme', 'zorder': 4}
    }

    # Determine the global min/max threshold for a consistent color scale
    vmin = df_consensus['consensus_threshold'].min()
    vmax = df_consensus['consensus_threshold'].max()

    scatter_plot = None
    # Group by label scheme and plot each one
    for scheme, group_df in df_consensus.groupby('label_scheme'):
        if scheme in marker_map:
            style = marker_map[scheme]
            group_df_sorted = group_df.sort_values('coverage')
            
            # Plot the markers with color based on the threshold
            scatter_plot = ax.scatter(
                group_df_sorted['coverage'], 
                group_df_sorted['accuracy'], 
                marker=style['marker'], 
                c=group_df_sorted['consensus_threshold'],
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                s=150,  # size of markers
                label=style['label'],
                zorder=style['zorder'],
                edgecolors='black',
                linewidth=0.5
            )

    # --- Aesthetics ---
    ax.set_xlabel('Coverage (%)', fontsize=15)
    ax.set_ylabel('Accuracy (on Graded Subset)', fontsize=15)
    ax.tick_params(axis='both', labelsize=15) # Set tick label size
    
    # Manually create handles for the legend to show only marker shapes, no color
    legend_handles = []
    for scheme_key, style_info in marker_map.items():
        legend_handles.append(plt.Line2D([0], [0], marker=style_info['marker'], color='black', label=style_info['label'], linestyle='None', markersize=10))
    
    ax.legend(handles=legend_handles, title="Label Scheme", framealpha=0, fontsize=15, title_fontsize=15)
    
    # Add a colorbar for the consensus threshold, if we plotted anything
    if scatter_plot:
        cbar = fig.colorbar(scatter_plot, ax=ax, pad=0.01)
        cbar.set_label('Consensus Threshold', fontsize=15)

    # Set reasonable limits if data exists
    if not df_consensus.empty:
        ax.set_xlim(min(df_consensus['coverage'].min(), 100) * 0.9, 105)
        ax.set_ylim(df_consensus['accuracy'].min() - 0.05, 1.05)

    plt.tight_layout()

    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'plot_C_consensus.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Plot C: The "Trust Curve" for multiple label schemes.')
    parser.add_argument('input_file', type=str, help='Path to the summary_stats.csv file.')
    args = parser.parse_args()
    
    plot_C_consensus(args.input_file)
