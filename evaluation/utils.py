import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt

def print_nav_evaluation_report(df_raw, model_alias, verbose=True):
    # 1. Calculate Aggregates & Statistics
    success_rate = df_raw["success"].mean() * 100
    collision_rate = df_raw["collision"].mean() * 100
    
    avg_steps, std_steps = df_raw["steps"].mean(), df_raw["steps"].std()
    avg_energy, std_energy = df_raw["energy"].mean(), df_raw["energy"].std()
    avg_safety, std_safety = df_raw["safety"].mean(), df_raw["safety"].std()
    avg_path_length, std_path_length = df_raw["path_length"].mean(), df_raw["path_length"].std()

    # 2. Prepare Status String
    if success_rate >= 98.0:
        status_str = "✅ EXCELLENT"
    elif success_rate >= 90.0:
        status_str = "⚠️ GOOD"
    else:
        status_str = f"❌ POOR ({100 - success_rate:.1f}% Fail)"

    # 3. Define Data Points (Label, Value)
    # Reorganized to fit your Navigation specific metrics
    metrics_list = [
        ("Status", status_str),
        ("Model", model_alias),
        ("Date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
        
        ("Success Rate", f"{success_rate:.1f} %"),
        ("Collision Rate", f"{collision_rate:.1f} %"),
        ("Safety Margin", f"{avg_safety:.3f} ± {std_safety:.3f} m"),
        
        ("Mean Steps", f"{avg_steps:.1f} ± {std_steps:.1f}"),
        ("Avg Energy", f"{avg_energy:.2f} ± {std_energy:.2f}"),
        ("Path Lenght", f"{avg_path_length:.2f} ± {std_path_length:.2f}"),
    ]

    # 4. Format into Columns (3 Columns Wide)
    num_columns = 3
    num_rows = math.ceil(len(metrics_list) / num_columns)
    table_rows = []

    for r in range(num_rows):
        row_data = []
        for c in range(num_columns):
            # Fill column by column
            idx = c * num_rows + r
            if idx < len(metrics_list):
                label, value = metrics_list[idx]
                row_data.extend([label, value])
            else:
                row_data.extend(["", ""]) # Empty filler
        table_rows.append(row_data)

    # 5. Generate Table
    headers = ["Metric", "Value"] * num_columns
    report_title = f" NAVIGATION EVALUATION REPORT: {model_alias.upper()}"
    table_output = tabulate(table_rows, headers=headers, tablefmt="fancy_grid")

    # 6. Final Print
    if verbose:
        print(f"\n{'='*85}")
        print(f"{report_title.center(85)}")
        print(f"{'='*85}")
        print(table_output)
        print("\n")