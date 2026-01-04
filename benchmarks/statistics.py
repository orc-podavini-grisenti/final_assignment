import itertools
import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy.stats import wilcoxon

def interpret_effect_size(d):
    """
    Returns the magnitude interpretation of Cliff's Delta.
    Thresholds based on Romano et al. (2006).
    """
    abs_d = abs(d)
    if abs_d < 0.147:
        return "Negligible"
    elif abs_d < 0.33:
        return "Small"
    elif abs_d < 0.474:
        return "Medium"
    else:
        return "Large"

def paired_stat_test(df, metrics_of_interest):
    """
    Generates a professional statistical report comparing controllers.
    """
    controllers = df["Controller"].unique()
    if len(controllers) != 2:
        print("⚠️  Statistical test requires exactly 2 controllers.")
        return

    c1, c2 = controllers[0], controllers[1]
    
    print(f"📊 HEAD-TO-HEAD ANALYSIS: {c1} vs {c2}")
    
    table_data = []

    for metric in metrics_of_interest:
        x = df[df.Controller == c1][metric].values
        y = df[df.Controller == c2][metric].values

        # 1. Wilcoxon Test
        try:
            stat, p = wilcoxon(x, y)
        except ValueError:
            # Handle cases where all values are identical (e.g. 0 collisions)
            p = 1.0 

        # 2. Cliff's Delta
        gt = sum(a > b for a, b in itertools.product(x, y))
        lt = sum(a < b for a, b in itertools.product(x, y))
        delta = (gt - lt) / (len(x) * len(y))
        
        # 3. Interpretation
        is_sig = p < 0.05
        mag = interpret_effect_size(delta) # Use the helper function provided earlier
        
        # Determine Winner
        # Note: For Errors/Time/Jitter, LOWER is better. For Success, HIGHER is better.
        # This logic assumes LOWER IS BETTER (standard for error metrics). 
        # You might need to flip logic for 'Success'.
        
        winner = "Draw"
        if is_sig:
            if delta > 0: # C1 > C2
                winner = c2 # C2 is smaller (better)
            elif delta < 0: # C1 < C2
                winner = c1 # C1 is smaller (better)
        
        # Special formatting for P-value
        p_str = f"{p:.5f} *" if is_sig else f"{p:.3f}"
        
        table_data.append([
            metric, 
            p_str, 
            f"{delta:.2f} ({mag})", 
            winner
        ])

    headers = ["Metric", "P-Value (<0.05?)", "Cliff's Delta (Effect)", "Winner (Lower is better)"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("\n")