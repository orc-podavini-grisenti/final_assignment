import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate  

# --- Import your modules ---
from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner
from controllers.rl_controller import RLController
from utils.simulation import run_simulation  

# --- Import local utils ---
from evaluation.utils import plot_sweep_results

"""
POLICY EVALUATION & COMPARISON SCRIPT
-------------------------------------
This script provides a framework to evaluate trained RL models and compare them against 
each other using standardized metrics. It operates in two modes:

1. SINGLE MODE (--mode single):
   - Conducts a robust statistical evaluation of a single .pth model file.
   - Runs a batch of simulations (default 50-100 episodes) with randomized but 
     reproducible scenarios.
   - Outputs a detailed performance report and saves results to a centralized CSV 
     database for long-term tracking.

2. COMPARE MODE (--mode compare):
   - Aggregates historical data from the 'policy_comparison.csv' file.
   - Generates a "Leaderboard" ranking models based on a multi-objective hierarchy: 
     Success Rate > Accuracy (CTE) > Smoothness > Efficiency (Tortuosity).
   - Produces visualization plots to identify performance gaps between different versions.

3. SWEEP MODE (--mode sweep):
   - Stress-tests a model by incrementally varying the path difficulty (turning radius).
   - Identifies the "Breaking Point" or the minimum radius a model can handle before 
     failing.
   - Useful for defining the operational design domain (ODD) of a specific controller.

EXAMPLE USAGE:
1. SINGLE MODE:  python ./scripts/run_policy_evaluation.py --mode single --model path/to/model.pth --name "v1_baseline"
2. COMPARE MODE:  python ./scripts/run_policy_evaluation.py --mode compare
3. SWEEP MODE:    python ./scripts/run_policy_evaluation.py --mode sweep --model path/to/model.pth -min_r 0.7 --max_r 3.0
"""

# --- CONSTANTS ---
CSV_FILE = "evaluation/output/TT_policy_comparison.csv"
PLOTS_DIR = "evaluation/output"


# ==============================================================================
#  1. SINGLE EVALUATION MODE
# ==============================================================================
def evaluate_single_model(model_path, model_alias, num_episodes, seed):
    """
    Runs simulation for a single model, prints a detailed report, and returns aggregated metrics.
    """
    print(f"--> Loading Model: {model_path}")
    
    # 1. Load Controller
    try:
        controller = RLController(model_path=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # 2. Setup Environment
    env = UnicycleEnv()
    np.random.seed(seed) # Ensure scenarios are reproducible
    
    # Storage for raw run data
    results = {
        "success": [], "cte": [], "max_cte": [], "smoothness": [], 
        "tortuosity": [], "velocity": [], "collision": [], 
        "steps": [], "energy": []
    }

    print(f"Starting {num_episodes} episodes...")

    for i in range(num_episodes):
        # Generate Scenario
        radius = np.clip(np.random.normal(1.65, 0.3), 0.8, 2.5)
        k_max = 1.0 / radius
        
        # Reset the environment with a unique but reproducible seed per episode
        # This ensures Episode 1 is always the same for every model, 
        # but Episode 1 is different from Episode 2.
        episode_seed = seed + i 
        env.reset(seed=episode_seed)

        planner = DubinsPlanner(curvature=k_max, step_size=0.05)
        path = planner.get_path(env.state, env.goal)

        # Run Simulation
        sim_data = run_simulation(env, path, controller, render=False, max_steps=1000)
        
        if sim_data:
            results["success"].append(sim_data["is_success"])
            results["cte"].append(sim_data["mean_error"])
            results["max_cte"].append(sim_data["max_error"])
            results["smoothness"].append(sim_data["control_smoothness"])
            results["tortuosity"].append(sim_data["tortuosity"])
            results["velocity"].append(sim_data["mean_v"])
            results["collision"].append(sim_data["collision"])
            results["steps"].append(sim_data["steps"])
            results["energy"].append(sim_data["energy_consumption"])
            
        print(f"Progress: {i+1}/{num_episodes}", end="\r")

    print("\nEvaluation Complete.")

    # 3. Aggregate Metrics
    total_runs = len(results["success"])
    if total_runs == 0: return None

    # Calculate Aggregates
    success_rate = np.mean(results["success"]) * 100
    avg_steps = np.mean(results["steps"])
    avg_cte = np.mean(results["cte"])
    max_cte_avg = np.mean(results["max_cte"])
    avg_smoothness = np.mean(results["smoothness"])
    avg_tortuosity = np.mean(results["tortuosity"])
    avg_velocity = np.mean(results["velocity"])
    avg_energy = np.mean(results["energy"])
    
    # 4. PREPARE THE TABULAR DISPLAY
    
    # 4.1. Status String
    if success_rate >= 98.0:
        status_str = "✅ EXCELLENT"
    elif success_rate >= 90.0:
        status_str = "⚠️ GOOD"
    else:
        status_str = f"❌ POOR ({100 - success_rate:.1f}% Fail)"

    # 4.2. Define Data Points (Label, Value)
    metrics_list = [
        ("Status", status_str),
        ("Steps (Avg)", f"{avg_steps}"),
        ("Success Rate", f"{success_rate:.1f} %"),
        
        ("Mean CTE", f"{avg_cte:.4f} m"),
        ("Max CTE (Avg)", f"{max_cte_avg:.4f} m"),
        ("Tortuosity", f"{avg_tortuosity:.3f}"),
        
        ("Avg Velocity", f"{avg_velocity:.2f} m/s"),
        ("Smoothness", f"{avg_smoothness:.4f}"),
        ("Avg Energy", f"{avg_energy:.2f}")
    ]

    # 4.3. Format into Columns (3 Columns Wide)
    num_columns = 3
    num_rows = math.ceil(len(metrics_list) / num_columns)
    table_rows = []

    for r in range(num_rows):
        row_data = []
        for c in range(num_columns):
            idx = c * num_rows + r
            if idx < len(metrics_list):
                label, value = metrics_list[idx]
                row_data.extend([label, value])
            else:
                row_data.extend(["", ""])
        table_rows.append(row_data)

    # 4.4. Print Table
    headers = ["Metric", "Value"] * num_columns
    print("\n" + "="*60)
    print(f" EVALUATION REPORT: {model_alias if model_alias else 'Unknown Model'}")
    print("="*60)
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
    print("\n")

    # Return dictionary for CSV saving
    agg_metrics = {
        "Model Name": model_alias if model_alias else os.path.basename(model_path),
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Episodes": total_runs,
        "Success Rate (%)": success_rate,
        "Mean Steps": avg_steps,
        "Collision Rate (%)": np.mean(results["collision"]) * 100,
        "Mean CTE (m)": avg_cte,
        "Avg Smoothness": avg_smoothness,
        "Avg Tortuosity": avg_tortuosity,
        "Avg Velocity (m/s)": avg_velocity,
        "Model Path": model_path 
    }

    return agg_metrics


def append_to_csv(metrics, csv_file = CSV_FILE):
    """
    Appends a dictionary of metrics as a new row in the CSV.
    """
    # Convert dict to DataFrame
    df_new = pd.DataFrame([metrics])
    
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    df_final.to_csv(csv_file, index=False)
    print(f"Metrics saved to {csv_file}")




# ==============================================================================
#  2. COMPARISON MODE (Unchanged)
# ==============================================================================
def comparison_analysis(csv_file = CSV_FILE):
    if not os.path.exists(csv_file):
        print(f"No comparison file found at {csv_file}. Run 'single' mode first.")
        return

    df = pd.read_csv(csv_file)
    
    if df.empty:
        print("CSV is empty.")
        return

    # Sort Leaderboard
    leaderboard = df.sort_values(
        by=["Success Rate (%)", "Mean CTE (m)", "Avg Smoothness", "Avg Tortuosity", "Mean Steps"], 
        ascending=[False, True, True, True, True]
    )

    print("\n" + "="*100)
    print(" POLICY COMPARISON LEADERBOARD")
    print("="*100)
    print_cols = ["Model Name", "Success Rate (%)", "Mean CTE (m)", "Avg Smoothness", "Avg Tortuosity", "Mean Steps"]
    print(leaderboard[print_cols].to_string(index=False))
    print("="*100 + "\n")

    # Generate Comparison Plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(df["Model Name"], df["Success Rate (%)"], color='skyblue', edgecolor='black')
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.3)
    plt.ylabel("Success Rate (%)")
    plt.title("Reliability Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/comparison_success.png")
    
    print(f"Plots saved to {PLOTS_DIR}/")




# ==============================================================================
#  3. RADIUS SWEEP MODE
# ==============================================================================
def radius_sweep(model_path, min_r, max_r, steps, seed):
    """
    Evaluates policy across a linear range of radii to find performance limits.
    """
    print(f"--> Starting Radius Sweep for: {model_path}")
    controller = RLController(model_path=model_path)
    env = UnicycleEnv()
    
    # Generate linear range of radii
    test_radii = np.linspace(min_r, max_r, steps)
    sweep_results = []

    for r in test_radii:
        k_max = 1.0 / r
        # Use a fixed seed for the environment to ensure radius is the only variable
        env.reset(seed=seed)
        planner = DubinsPlanner(curvature=k_max, step_size=0.05)
        path = planner.get_path(env.state, env.goal)
        
        res = run_simulation(env, path, controller, render=False, max_steps=1000)
        
        if res:
            # print('Success: ', res['is_success'] , " seed = ", seed)
            sweep_results.append({
                "radius": r,
                "success": int(res["is_success"]),
                "mean_cte": res["mean_error"],
                "max_cte": res["max_error"],
                "smoothness": res["control_smoothness"]
            })
            status = "PASS" if res["is_success"] else "FAIL"
            print(f"Radius {r:.2f}m: {status} | CTE: {res['mean_error']:.4f}")

    df_sweep = pd.DataFrame(sweep_results)
    plot_sweep_results(df_sweep, model_path, plots_dir = model_path)
    return df_sweep

