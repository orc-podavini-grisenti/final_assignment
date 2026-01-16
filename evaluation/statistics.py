# ==============================================================================
#  STATISTICAL HELPERS
# ==============================================================================

import itertools
import numpy as np
from tabulate import tabulate
from scipy.stats import wilcoxon


def interpret_effect_size(d):
    """ Returns the magnitude interpretation of Cliff's Delta. """
    abs_d = abs(d)
    if abs_d < 0.147: return "Negligible"
    elif abs_d < 0.33: return "Small"
    elif abs_d < 0.474: return "Medium"
    else: return "Large"


def paired_stat_test(df1, df2, metrics_of_interest, name1, name2):
    """ Generates a statistical report comparing two controllers. """
    print(f"📊 HEAD-TO-HEAD ANALYSIS: {name1} vs {name2}")
    table_data = []

    for metric in metrics_of_interest:
        if metric not in df1.columns or metric not in df2.columns: 
            continue

        # 1. Cast to float and align lengths
        x = df1[metric].values.astype(float)
        y = df2[metric].values.astype(float)
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        # 2. CHECK FOR IDENTICAL DISTRIBUTIONS
        # If all differences are zero, wilcoxon will fail with a RuntimeWarning.
        if np.array_equal(x, y) or np.all(x - y == 0):
            p = 1.0  # Distributions are identical, so p-value is 1
        else:
            try:
                # 3. Perform Wilcoxon test
                stat, p = wilcoxon(x, y)
            except (ValueError, IndexError, TypeError):
                p = 1.0 

        # 4. Cliff's Delta (Calculated normally even if identical)
        gt = sum(a > b for a, b in itertools.product(x, y))
        lt = sum(a < b for a, b in itertools.product(x, y))
        divisor = (len(x) * len(y))
        delta = (gt - lt) / divisor if divisor > 0 else 0
        
        # 5. Formatting and Winner Logic
        is_sig = p < 0.05 if not np.isnan(p) else False
        mag = interpret_effect_size(delta)
        
        winner = "Draw"
        if is_sig:
            if metric == "success":
                winner = name1 if delta > 0 else name2
            else:
                winner = name1 if delta < 0 else name2
        
        # Handle cases where p might still be NaN
        p_str = f"{p:.5f} *" if is_sig else (f"{p:.3f}" if not np.isnan(p) else "1.000")
        
        table_data.append([metric, p_str, f"{delta:.2f} ({mag})", winner])

    headers = ["Metric", "P-Value (<0.05?)", "Cliff's Delta", "Winner"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("\n")