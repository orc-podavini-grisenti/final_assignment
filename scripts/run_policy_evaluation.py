import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from simulation import run_simulation
from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner
from controllers.rl_controller import RLController

def sample_radius(mean=1.65, std=0.3, min_r=0.8, max_r=2.5):
    """Samples a radius using a Gaussian distribution, strictly constrained."""
    return np.clip(np.random.normal(loc=mean, scale=std), min_r, max_r)

def plot_benchmarks(radii, success_list, error_list):
    """Generates a graph showing success rate and avg distance related to the radius."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    success_ints = np.array(success_list).astype(int)
    color_map = ['red' if s == 0 else 'green' for s in success_ints]
    ax1.scatter(radii, success_ints, c=color_map, alpha=0.6, label='Episode Result')
    ax1.set_xlabel('Radius (m)')
    ax1.set_ylabel('Success (1) / Failure (0)', color='black')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Failure', 'Success'])

    ax2 = ax1.twinx()
    ax2.scatter(radii, error_list, color='blue', marker='x', alpha=0.5, label='Mean Tracking Error')
    ax2.set_ylabel('Mean Tracking Error (m)', color='blue')
    
    if len(radii) > 1:
        z = np.polyfit(radii, error_list, 1)
        p = np.poly1d(z)
        ax2.plot(radii, p(radii), "b--", alpha=0.8, label='Error Trend')

    plt.title('Performance Benchmarks vs. Path Radius')
    fig.tight_layout()
    
    os.makedirs("outputs/plots", exist_ok=True)
    plot_path = f"outputs/plots/bench_graph_{datetime.now().strftime('%m%d_%H%M')}.png"
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")
    plt.show()

def run_benchmark(num_episodes=50, fixed_radius=None):
    model_path = "outputs/models_saved/experiments/run_20260103_175743/policy_model.pth"
    env = UnicycleEnv() 
    controller = RLController(model_path=model_path)

    radii_log = []
    success_log = []
    error_log = []
    
    print(f"Benchmarking {num_episodes} episodes...")

    for i in range(num_episodes):
        radius = fixed_radius if fixed_radius else sample_radius()
        k_max = 1.0 / radius
        planner = DubinsPlanner(curvature_max=k_max, step_size=0.05)
        
        # Increased max_steps to 2000 as per your previous observation
        res = run_simulation(env, planner, controller, render=False, max_steps=2000)
        
        if res:
            radii_log.append(radius)
            success_log.append(res['is_success'])
            error_log.append(res['mean_error'])
            
        print(f"Progress: {i+1}/{num_episodes} (R: {radius:.2f}m)", end="\r")

    # --- NEW: Summary Calculations ---
    total_runs = len(success_log)
    if total_runs > 0:
        success_count = sum(success_log)
        success_rate = (success_count / total_runs) * 100
        avg_distance_error = np.mean(error_log)

        print("\n" + "="*40)
        print(f"BENCHMARK SUMMARY (N={total_runs})")
        print("-" * 40)
        print(f"Success Percentage:      {success_rate:.2f}%")
        print(f"Average Tracking Error:  {avg_distance_error:.4f} m")
        print("="*40 + "\n")
    else:
        print("\nNo data collected during benchmark.")

    # Generate the Visualization
    plot_benchmarks(radii_log, success_log, error_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_episodes", type=int, default=50)
    parser.add_argument("-r", "--radius", type=float)
    args = parser.parse_args()

    run_benchmark(num_episodes=args.num_episodes, fixed_radius=args.radius)