import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tabulate import tabulate  

# --- Import your modules ---
from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner
from models.navigation_network import NavAgent
from utils.simulation_2 import navigation_simulation  
from utils.normalization import ObservationNormalizer

# --- Import local utils ---
from evaluation.utils import generate_comparison_boxplots, print_nav_evaluation_report
from evaluation.statistics import paired_stat_test


# --- CONSTANTS ---
CSV_GENERAL = "evaluation/output/NAV_performance.csv"
CSV_PATH_QUAL = "evaluation/output/NAV_path_quality.csv"
RAW_DATA_DIR = "evaluation/output/raw_nav_data"


# --- UTILS ---
def save_to_csv(metrics, file_path):
    """Utility to append results to a specific CSV file."""
    df_new = pd.DataFrame([metrics])
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_final.to_csv(file_path, index=False)



'''
NB: We split the single evaluation intwo parts. this becouse the path effiency evaluation can be done
    only with an empty enviroment. Since the Dubins Planner baseline implemented is a very simple
    versio that don't consider obstacle avoidance
'''
def evaluate_single_nav_model(model_path, model_alias, num_episodes=50, seed=0, render=False, verbose=True):
    """
    Conducts a comprehensive 'Mission Reliability' evaluation for a trained RL navigation agent.
    The evaluation focuses on four key pillars:
    1. Reliability: Success vs. Collision rates.
    2. Efficiency: Time (steps) and Energy consumption.
    3. Safety: Minimum distance maintained from obstacles.
    4. Path Quality: Total distance traveled.

    Args:
        model_path (str): File system path to the trained '.pth' model weights.
        model_alias (str): A unique string identifier (e.g., 'v1_baseline') used for 
            logging results in CSV files and naming raw data exports.
        num_episodes (int, optional): The number of simulation trials to run. 
            Defaults to 50 (should be set to >0 for a valid evaluation).
        seed (int, optional): The base random seed for reproducibility. Each episode 
            'i' uses 'seed + i' to ensure varied but deterministic scenarios. Defaults to 0.
        render (bool, optional): If True, opens a visualization window to display 
            the robot's movement in real-time. Defaults to False.
        verbose (True, optional): If True, prints a detailed summary report (Mean ± Std) 
            to the console after completion. Defaults to True.

    Returns:
        dict: A dictionary containing the aggregated mean metrics (Success Rate, 
            Collision Rate, Avg Energy, etc.) for the entire evaluation run.
    """
    
    """PART 1: MISSION RELIABILITY (Success, Collision, Safety, Energy)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Environment
    env = UnicycleEnv()
    if seed is not None:
        print(f"🎲 Using Seed: {seed}")

    env.reset(seed=seed)

    # 2. Setup Agent and Normalizer
    n_rays = env.obs_cfg.get('n_rays', 20)
    obs_dim = 3 + n_rays # [rho, alpha, d_theta] + lidar
    action_dim = 2
    
    agent = NavAgent(obs_dim, action_dim, hidden_dim=512, device=device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval() # Set to evaluation mode

    normalizer = ObservationNormalizer(max_dist=5.0, lidar_range=env.rob_cfg['lidar_range'])

    
    # Dictionary to store episode-by-episode data for stats
    raw_results = {"success": [], "collision": [], "steps": [], 
                   "energy": [], "safety": [], "path_length": []}

    print(f"Starting {num_episodes} episodes...")

    for i in range(num_episodes):
        env.reset(seed=seed + i)
        
        sim_data = navigation_simulation(env, agent, normalizer, render=render)
        
        if sim_data:
            raw_results["success"].append(int(sim_data["is_success"]))
            raw_results["collision"].append(int(sim_data["collision"]))
            raw_results["steps"].append(sim_data["steps"])
            raw_results["energy"].append(sim_data["energy_consumption"])
            raw_results["safety"].append(sim_data["safety_margin"])
            raw_results["path_length"].append(sim_data["path_length"])
        
        print(f"Progress: {i+1}/{num_episodes}", end="\r")

    print("\nEvaluation Complete.")

    # 1. Save Raw Data for Stats/Plotting
    df_raw = pd.DataFrame(raw_results)
    raw_path = os.path.join(RAW_DATA_DIR, f"{model_alias}_general_raw.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df_raw.to_csv(raw_path, index=False)

    # 2. Calculate Aggregates for Leaderboard
    metrics = {
        "Model Name": model_alias,
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Success Rate (%)": df_raw["success"].mean() * 100,
        "Collision Rate (%)": df_raw["collision"].mean() * 100,
        "Mean Steps": df_raw["steps"].mean(),
        "Avg Energy": df_raw["energy"].mean(),
        "Safety Margin": df_raw["safety"].mean(), 
        "Path Lenght": df_raw["path_length"].mean()
    }

    save_to_csv(metrics, CSV_GENERAL)
    print_nav_evaluation_report(df_raw, model_alias, verbose=verbose)
    return metrics



 
def evaluate_single_nav_model_path(model_path, model_alias, num_episodes=50, seed=0, render = False, verbose=True):
    """PART 2: PATH EFFICIENCY (vs Dubins Planner)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Environment
    empty_env_path = "configs/empty_env.yaml"
    env = UnicycleEnv(empty_env_path)
    if seed is not None:
        print(f"🎲 Using Seed: {seed}")
    
    env.reset(seed=seed)

    # 2. Setup Agent and Normalizer
    n_rays = env.obs_cfg.get('n_rays', 20)
    obs_dim = 3 + n_rays # [rho, alpha, d_theta] + lidar
    action_dim = 2
    
    agent = NavAgent(obs_dim, action_dim, hidden_dim=512, device=device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval() # Set to evaluation mode

    normalizer = ObservationNormalizer(max_dist=5.0, lidar_range=env.rob_cfg['lidar_range'])

    raw_path_results = {"efficiency": [], "actual_len": [], "ideal_len": []}

    for i in range(num_episodes):
        env.reset(seed=seed + i)
        
        # Calculate Ideal Reference (Dubins)
        radius = env.rob_cfg['v_max'] / env.rob_cfg['w_max']
        planner = DubinsPlanner(curvature_max=1.0/radius, step_size=0.05)
        ideal_path = planner.get_path(env.state, env.goal)
        ideal_len = np.sum(np.linalg.norm(np.diff(ideal_path[:, :2], axis=0), axis=1))

        sim_data = navigation_simulation(env, agent, normalizer, render=render, ideal_path=ideal_path)
        
        if sim_data and sim_data["is_success"]:
            raw_path_results["actual_len"].append(sim_data["path_length"])
            raw_path_results["ideal_len"].append(ideal_len)
            eff = sim_data["path_length"] / ideal_len if sim_data["path_length"] > 0 else 0
            raw_path_results["efficiency"].append(eff)
        
        print(f"Progress: {i+1}/{num_episodes}", end="\r")

    print("\nEvaluation Complete.")

    # 1. Save Raw Data
    df_raw = pd.DataFrame(raw_path_results)
    raw_path = os.path.join(RAW_DATA_DIR, f"{model_alias}_path_raw.csv")
    df_raw.to_csv(raw_path, index=False)

    mean_efficiency, std_efficiency = df_raw["efficiency"].mean(), df_raw["efficiency"].std()
    mean_actual_path_lenght, std_actual_path_lenght = df_raw["actual_len"].mean(), df_raw["actual_len"].std()
    mean_ideal_path_lenght, std_ideal_path_lenght = df_raw["ideal_len"].mean(), df_raw["ideal_len"].std()


    # 2. Calculate Aggregates
    metrics = {
        "Model Name": model_alias,
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Nav Efficiency": mean_efficiency if not df_raw.empty else 0,
        "Avg Actual Path Length": mean_actual_path_lenght if not df_raw.empty else 0,
        "Avg Ideal Path Lenght": mean_ideal_path_lenght if not df_raw.empty else 0,
    }

    save_to_csv(metrics, CSV_PATH_QUAL)

    report_title = "NAV PATH EFFICIENCY"
    print(f"\n{'='*85}")
    print(f"{report_title.center(85)}")
    print(f"\t Avg Ideal Path Length:\t = \t {mean_ideal_path_lenght:.4f} ± {std_ideal_path_lenght:.4f}")
    print(f"\t Avg Actual Path Length: = \t {mean_actual_path_lenght:.4f} ± {std_actual_path_lenght:.4f}")
    print(f"\t Avg Efficiency:\t = \t {mean_efficiency:.4f} ± {std_efficiency:.4f}")
    print("\n")
    print(f"{'='*85}")

    return metrics




def navigation_comparison_analysis(mode="general"):
    """
    Leaderboard + Statistical Significance + Boxplots.
    mode: 'general' or 'path'
    """
    file_path = CSV_GENERAL if mode == "general" else CSV_PATH_QUAL
    suffix = "general_raw" if mode == "general" else "path_raw"
    
    # Define which columns to run stats on based on the mode
    if mode == "general":
        nav_metrics = ["success", "collision", "steps", "energy", "safety"]
    else:
        nav_metrics = ["efficiency", "actual_len"]

    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)
    df_latest = df.sort_values('Date').groupby('Model Name').tail(1)
    models = df_latest["Model Name"].unique()

    # --- 1. PRINT LEADERBOARD ---
    print(f"\n🏆 NAVIGATION LEADERBOARD ({mode.upper()})")
    print(tabulate(df_latest, headers='keys', tablefmt='fancy_grid', showindex=False))

    # --- 2. STATISTICAL ANALYSIS (Wilcoxon/T-Test) ---
    if len(models) >= 2:
        print(f"\n⚖️  STATISTICAL SIGNIFICANCE (Reference: {models[0]})")
        for i in range(1, len(models)):
            path_a = os.path.join(RAW_DATA_DIR, f"{models[0]}_{suffix}.csv")
            path_b = os.path.join(RAW_DATA_DIR, f"{models[i]}_{suffix}.csv")
            
            if os.path.exists(path_a) and os.path.exists(path_b):
                df_a, df_b = pd.read_csv(path_a), pd.read_csv(path_b)
                # Reusing your utility function
                paired_stat_test(df_a, df_b, nav_metrics, models[0], models[i])

        # --- 3. PLOTTING ---
        print(f"\n📊 Generating {mode} comparative boxplots...")
        generate_comparison_boxplots(
            models=models, 
            raw_data_dir=RAW_DATA_DIR, 
            plots_dir=f"evaluation/output/plots_{mode}",
            # Important: Ensure your generate_comparison_boxplots uses these suffixes
            file_suffix=f"_{suffix}.csv" 
        )