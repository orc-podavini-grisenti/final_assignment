import argparse
import time
import math
import random
import os
import numpy as np
from tabulate import tabulate
from utils.simulation import run_simulation
from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner

# Import your controllers
from controllers.lyapunov_controller import LyapunovController, LyapunovParams
from controllers.rl_controller import RLController

"""
WHAT THE SCRIPT DOES:
--------------------
This script executes a single trajectory tracking simulation for a unicycle robot. 
It allows you to compare a classical Lyapunov-based controller against a 
trained Reinforcement Learning (RL) policy. 

It is particularly useful for:
- Debugging: Visualizing how a specific controller behaves in real-time.
- Scenario Testing: Using the --seed parameter to reproduce exact environment 
  initializations (robot start pose and goal location) to see how different 
  controllers handle the same challenge.

  
HOW TO RUN THIS SCRIPT:
-----------------------
1. RL only with specific model and seed: 
   $ python ./scripts/run_trajectory_tracking.py --type rl --model v1_no_baseline --seed 1

2. Lyapunov only: 
   $ python ./scripts/run_trajectory_tracking.py --type lyapunov --seed 123

3. Both with a specific model and seed: 
   $ python ./scripts/run_trajectory_tracking.py --type both --model v1_no_baseline --seed 65

NB: add the final parameter --save_video to store the video of the episode
"""

def get_controller(c_type, model_name=None):
    """Factory function to instantiate the requested controller."""
    if c_type == "rl":
        # Default fallback if no model is provided
        if not model_name:
            model_name = "v1_no_baseline"
            
        # Path: training/<model_name>/policy_model.pth
        model_path = os.path.join("training", model_name, "policy_model.ph")
        
        print(f"🔵 Loading RL Controller from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find model at {model_path}")
            
        return RLController(model_path=model_path)
    
    elif c_type == "lyapunov":
        print("🟢 Loading Lyapunov Controller...")
        return LyapunovController(LyapunovParams(K_P=2.0, K_THETA=5.0))
    
    return None


def run_visual_episode(controller_type, seed=None, model_name=None, save_video=False):
    """Runs a single simulation episode with the specified controller."""
    env = UnicycleEnv()
    np.random.seed(seed) 
    
    if seed is not None:
        print(f"🎲 Using Seed: {seed}")
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset() 
    else:
        env.reset()

    radius = np.clip(np.random.normal(1.65, 0.3), 0.8, 2.5)
    k_max = 1.0 / radius
    planner = DubinsPlanner(curvature=k_max, step_size=0.05)
    path = planner.get_path(env.state, env.goal)

    controller = get_controller(controller_type, model_name=model_name)

    # Prepare video recording parameters
    video_path = None
    if save_video:
        os.makedirs("_documentations/videos", exist_ok=True)
        # Construct filename
        filename = f"{controller_type}"
        if model_name:
            filename += f"_{model_name}"
        filename += f"_seed{seed}.mp4"
        
        video_path = os.path.join("_documentations/videos", filename)
        print(f"📹 Video path set to: {video_path}")

    print(f"▶️  Starting visual demonstration with {controller_type.upper()}...")
    
    # Pass the video path to the simulation runner
    # (Assuming run_simulation supports a 'video_path' or similar argument)
    result = run_simulation(env, path, controller, render=True, video_path=video_path)

    if result:
        # ... [Metric printing logic remains the same] ...
        status_str = "✅ SUCCESS" if result["is_success"] else "❌ FAILED"
        dt_val = getattr(env, 'dt', 0.05)
        metrics_list = [
            ("Status", status_str),
            ("Sim Time", f"{result['sim_time']:.2f} s"),
            ("Steps", f"{result['steps']} (dt={dt_val})"),
            ("Tortuosity", f"{result.get('tortuosity', 1.0):.3f}"),
            ("Final Dist Error", f"{result['distance_error']:.4f} m"),
            ("Final Heading Error", f"{result['heading_error']:.4f} rad"),
            ("Mean CTE", f"{result['mean_error']:.4f} m"),
            ("Max CTE", f"{result.get('max_error', 0.0):.4f} m"),
            ("Avg Linear Vel", f"{result.get('mean_v', 0.0):.2f} m/s"),
            ("Avg Angular Vel", f"{result.get('mean_omega', 0.0):.2f} rad/s"),
            ("Energy", f"{result.get('energy_consumption', 0.0):.2f}"),
            ("Smoothness", f"{result.get('control_smoothness', 0.0):.4f}")
        ]

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

        headers = ["Metric", "Value"] * num_columns
        print("\n" + tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
    
    time.sleep(1.0)

def main():
    parser = argparse.ArgumentParser(description="Run visual demonstrations.")
    parser.add_argument("--type", type=str, default="rl", choices=["rl", "lyapunov", "both"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    
    # NEW FLAG ADDED HERE
    parser.add_argument(
        "--save_video", 
        action="store_true", 
        help="If set, saves the simulation as an .mp4 file in the '_documentations/videos/' folder."
    )
    
    args = parser.parse_args()
    effective_seed = args.seed if args.seed is not None else random.randint(0, 10000)

    if args.type == "both":
        print(f"\nDOUBLE DEMO MODE: Comparing RL vs Lyapunov (Seed: {effective_seed})")
        run_visual_episode("rl", seed=effective_seed, model_name=args.model, save_video=args.save_video)
        print("\n" + "-"*30 + "\n")
        run_visual_episode("lyapunov", seed=effective_seed, save_video=args.save_video)
    else:
        run_visual_episode(args.type, seed=effective_seed, model_name=args.model, save_video=args.save_video)

if __name__ == "__main__":
    main()