import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_sweep_results(df, model_path, plots_dir='evaluation/output'):
    """Generates a scatter plot for Success/Failure and Error vs Radius."""
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    name = os.path.basename(model_path)

    # 1. Plot Success/Failure as circles
    successes = df[df['success'] == 1]
    failures = df[df['success'] == 0]

    ax1.scatter(successes['radius'], [1] * len(successes), 
                color='green', marker='o', s=50, alpha=0.6, label='Success')
    ax1.scatter(failures['radius'], [0] * len(failures), 
                color='red', marker='o', s=50, alpha=0.6, label='Failure')

    ax1.set_xlabel('Turning Radius (m)', fontsize=12)
    ax1.set_ylabel('Success (1) / Failure (0)', fontsize=12)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Failure', 'Success'])
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)

    # 2. Plot Tracking Error (CTE) on twin axis
    ax2 = ax1.twinx()
    ax2.scatter(df['radius'], df['mean_cte'], 
                color='blue', marker='x', alpha=0.5, label='Mean CTE')
    
    # Add a trend line for the CTE
    if len(df) > 1:
        z = np.polyfit(df['radius'], df['mean_cte'], 1)
        p = np.poly1d(z)
        ax2.plot(df['radius'], p(df['radius']), "b--", alpha=0.8, linewidth=1.5)

    ax2.set_ylabel('Mean Tracking Error (m)', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')

    # Title and Legend
    plt.title(f'Performance Benchmarks vs. Path Radius\nModel: {name}', fontsize=14)
    
    # Combining legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower left', frameon=True)

    fig.tight_layout()
    
    save_path = f"{plots_dir}/sweep_scatter_{name.replace('.ph','')}.png"
    plt.savefig(save_path)
    print(f"Sweep plot saved to: {save_path}")
    plt.show()




sns.set_style("whitegrid")

def generate_comparison_boxplots(models, raw_data_dir, plots_dir):
    """
    Combines individual model data and creates a boxplot for each metric.
    """
    combined_data = []

    # 1. Load data from individual CSV files
    for model_name in models:
        file_path = os.path.join(raw_data_dir, f"{model_name}_raw_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["Controller"] = model_name
            combined_data.append(df)
    
    if not combined_data:
        print("No raw data found to plot.")
        return

    full_df = pd.concat(combined_data, ignore_index=True)

    # 2. Filter metrics to plot (exclude non-numeric and success columns)
    metrics_to_plot = [col for col in full_df.columns if col not in ["Controller", "success", "collision", "Model Name"]]

    os.makedirs(plots_dir, exist_ok=True)
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 6))
        
        # FIX: Added 'hue="Controller"' and 'legend=False' to satisfy the deprecation warning
        sns.boxplot(
            data=full_df, 
            x="Controller", 
            y=metric, 
            hue="Controller", 
            palette="Set2", 
            legend=False
        )
        
        # Overlay the stripplot for data density visualization
        sns.stripplot(
            data=full_df, 
            x="Controller", 
            y=metric,
            color="black", 
            alpha=0.3, 
            jitter=True
        )
        
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Performance Distribution: {metric.replace('_', ' ').title()}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        save_path = os.path.join(plots_dir, f"boxplot_{metric}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved: {save_path}")