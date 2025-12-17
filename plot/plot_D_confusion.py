
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

    # Define the full range of possible labels (e.g., 1-5)
    min_label = min(df_graded['gold_label_id'].min(), df_graded['predicted_label_id'].min())
    max_label = max(df_graded['gold_label_id'].max(), df_graded['predicted_label_id'].max())
    labels = sorted(list(set(df_graded['gold_label_id']) | set(df_graded['predicted_label_id'])))

    # Generate the confusion matrix
    cm = confusion_matrix(df_graded['gold_label_id'], df_graded['predicted_label_id'], labels=labels)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='black')

    # --- Aesthetics ---
    ax.set_title('Plot D: Confusion Matrix (Error Analysis)', fontsize=16, weight='bold')
    ax.set_xlabel('Model Predicted Score', fontsize=12)
    ax.set_ylabel('Human Gold Label', fontsize=12)
    
    # Annotations for interpretation
    ax.text(0.5, 1.05, 'Diagonal (Correct), Upper-Triangle (Leniency), Lower-Triangle (Harshness)', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')

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
