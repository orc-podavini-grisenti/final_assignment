import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def plot_metric_box(df, metric, ylabel, save_path):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Controller", y=metric)
    sns.stripplot(
        data=df, x="Controller", y=metric,
        color="black", alpha=0.5, jitter=True
    )
    plt.ylabel(ylabel)
    plt.title(metric.replace("_", " "))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
