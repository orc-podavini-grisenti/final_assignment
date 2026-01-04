from pathlib import Path
from tabulate import tabulate
from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner
from controllers.lyapunov_controller import LyapunovController, LyapunovParams
from controllers.rl_controller import RLController

from benchmarks.controllers_benchmark import ControllerBenchmark
from benchmarks.plots import (
    plot_metric_box
)
from benchmarks.statistics import paired_stat_test


# ---- Setup ----
MODEL = Path("outputs/models_saved/experiments/run_20260103_101956/policy_model.pth")
RESULTS = Path("benchmarks/results")
RESULTS.mkdir(exist_ok=True)

env = UnicycleEnv()
planner = DubinsPlanner(curvature_max=1.5, step_size=0.05)

controllers = {
    "Lyapunov": LyapunovController(LyapunovParams(2.0, 5.0)),
    "RL": RLController(MODEL)
}

benchmark = ControllerBenchmark(env, planner, controllers)

df, trial_data , summary = benchmark.run(n_trials=50)

df.to_csv(RESULTS / "metrics.csv", index=False)
summary.to_csv(RESULTS / "summary.csv")


# --- Prints ---
# Filter out 'Trial' and 'Collision' if it's always 0
if 'Trial_mean' in summary.columns:
    summary.drop(columns=['Trial_mean', 'Trial_std'], inplace=True, errors='ignore')
    summary.drop(columns=['Collision_mean'], inplace=True, errors='ignore')


# Separate metrics into logical groups for cleaner reading
groups = {
    "🎯 ACCURACY (Lower is better)": ["RMS_CTE_mean", "Mean_CTE_mean", "Max_CTE_mean"],
    "⚡ EFFICIENCY": ["Completion_Time_mean", "Control_Energy_mean", "Mean_Spead_mean"],
    "🎮 SMOOTHNESS (Lower is better)": ["Control_Jitter_mean"],
    "✅ RELIABILITY (Higher is better)": ["Success_mean"]
}

print("\n" + "="*60)
print("🏁 BENCHMARK REPORT CARD 🏁")
print("="*60)

for group_name, metrics in groups.items():
    # Filter only existing columns
    valid_metrics = [m for m in metrics if m in summary.columns]
    if not valid_metrics: continue

    subset = summary[valid_metrics].T # Transpose for better readability
    
    # Format specific rows (Percentage for Success/Collision)
    if "RELIABILITY" in group_name:
        # Multiply by 100 for percentage display
        subset = subset * 100 
        subset.index = [idx.replace("_mean", " (%)") for idx in subset.index]
    else:
        # Clean up index names
        subset.index = [idx.replace("_mean", "") for idx in subset.index]

    print(f"\n{group_name}")
    print(tabulate(subset, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))

print("\n" + "="*60 + "\n")

# ---- Plots ----
plot_metric_box(df, "RMS_CTE", "RMS CTE [m]", RESULTS / "cte_box.png")
plot_metric_box(df, "Control_Jitter", "std(ω)", RESULTS / "jitter_box.png")
plot_metric_box(df, "Completion_Time", "Time [s]", RESULTS / "time_box.png")
plot_metric_box(df, "Mean_Spead", "Velocity [m/s]", RESULTS / "mean_speed_box.png")
plot_metric_box(df, "Control_Energy", "Energy [mean(abs(ω))]", RESULTS / "energy_box.png")


# ---- Statistics ----
paired_stat_test(df, ("RMS_CTE", "Control_Jitter", "Completion_Time", "Mean_Spead", "Control_Energy"))
