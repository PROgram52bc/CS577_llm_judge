
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_D_confusion(input_file, output_dir='plot/logs/plot_D_confusion'):
    """
    Generates Plot D: Confusion Matrix (Error Analysis).

    This plot is a heatmap that visualizes the confusion matrix between the model's
    predicted scores and the human gold labels, helping to diagnose specific
    grading biases like harshness or leniency.
    """
    # Read the data
    df = pd.read_csv(input_file)

    # --- Data Preprocessing ---
    # Filter out withdrawn examples, as they have no prediction
    df_graded = df[df['withdrawn'] == False].copy()
    
    if df_graded.empty:
        raise ValueError("No graded (non-withdrawn) examples found in the input file.")

    # Ensure labels are treated as integers
    df_graded['gold_label_id'] = df_graded['gold_label_id'].astype(int)
    df_graded['predicted_label_id'] = df_graded['predicted_label_id'].astype(int)

    # Get unique label IDs and their corresponding names
    unique_label_info = df_graded[['gold_label_id', 'gold_label_name']].drop_duplicates()

    # Define the desired explicit order for the axes
    desired_order = [
        'correct', 
        'partially_correct_incomplete', 
        'contradictory', 
        'irrelevant', 
        'non_domain'
    ]
    
    # Reorder the label info based on the desired order
    # This ensures that any labels not in the data are excluded
    unique_label_info = unique_label_info.set_index('gold_label_name').reindex(desired_order).dropna().reset_index()

    # Extract sorted numeric IDs and create display names using an explicit mapping
    numeric_labels = unique_label_info['gold_label_id'].tolist()
    
    label_display_mapping = {
        'correct': 'Correct',
        'partially_correct_incomplete': 'Partial',
        'contradictory': 'Contradictory',
        'irrelevant': 'Irrelevant',
        'non_domain': 'Non-Domain'
    }
    
    # Use the mapping, with a fallback to the original name if a label isn't in the map
    display_labels = unique_label_info['gold_label_name'].map(label_display_mapping).fillna(unique_label_info['gold_label_name']).tolist()

    # Generate the confusion matrix
    cm = confusion_matrix(df_graded['gold_label_id'], df_graded['predicted_label_id'], labels=numeric_labels)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=display_labels, yticklabels=display_labels,
                linewidths=.5, linecolor='black', annot_kws={"fontsize": 15})

    # --- Aesthetics ---

    ax.set_xlabel('Model Predicted Score', fontsize=15)
    ax.set_ylabel('Human Gold Label', fontsize=15)
    ax.tick_params(axis='x', rotation=20, labelsize=15) # x-axis tick labels
    ax.tick_params(axis='y', labelsize=15) # y-axis tick labels
    
    # Annotations for interpretation
    ax.text(0.5, 1.05, 'Diagonal (Correct), Upper-Triangle (Harshness), Lower-Triangle (Leniency)', 
            transform=ax.transAxes, ha='center', fontsize=15, style='italic')

    plt.tight_layout()

    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'plot_D_confusion.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Plot D: Confusion Matrix.")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    args = parser.parse_args()
    
    plot_D_confusion(args.input_file)
