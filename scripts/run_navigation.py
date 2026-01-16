import argparse
import time
import math
import random
import os
import numpy as np
import torch
from tabulate import tabulate

# Environment and Simulation Utilities
from envs.unicycle_env import UnicycleEnv
from utils.simulation import navigation_simulation
from utils.normalization import ObservationNormalizer
from models.navigation_network import NavActor

"""
WHAT THE SCRIPT DOES:
--------------------
This script executes a single autonomous navigation episode for a unicycle robot. 
Unlike trajectory tracking, the robot here is not forced to follow a pre-defined 
path. Instead, it uses a trained RL policy and LIDAR sensors to perceive 
obstacles and navigate toward the goal in real-time.

It is particularly useful for:
- Navigation Benchmarking: Evaluating how the RL agent handles obstacle avoidance 
  compared to a theoretical ideal (Dubins path).
- Perception Testing: Observing how the agent reacts to sensor inputs (Lidar).
- Reproducibility: Using the --seed parameter to test the model against 
  specific obstacle configurations.

HOW TO RUN THIS SCRIPT:
-----------------------
1. Run with a specific trained model: 
   $ python ./scripts/run_navigation.py --model nav_1 --seed 3086

2. Run and save a video of the navigation performance: 
   $ python ./scripts/run_navigation.py --model nav_1 --save_video

3. Test without a fixed seed (random scenario):
   $ python ./scripts/run_navigation.py --model nav_1

"""

def get_navigation_agent(model_name, obs_dim, action_dim, device):
    """Loads the trained NavAgent and its weights."""
    # Construct model path
    model_path = os.path.join("training", model_name, "nav_actor_final.pth")
    
    if not os.path.exists(model_path):
        # Fallback to direct model path if the full experiment path isn't used
        raise FileNotFoundError(f"Could not find model at {model_path}")

    print(f"🔵 Loading Navigation Agent from: {model_path}")
    
    agent = NavActor(obs_dim, action_dim, hidden_dim=512).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval() # Set to evaluation mode
    return agent

def run_nav_episode(model_name, seed=None, save_video=False):
    """Runs a single navigation episode using the RL policy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Environment
    env = UnicycleEnv("env_nav_ev")
    if seed is not None:
        print(f"🎲 Using Seed: {seed}")
        env.reset(seed=seed)
    else:
        env.reset()

    # 2. Setup Agent and Normalizer
    n_rays = env.obs_cfg.get('n_rays', 20)
    obs_dim = 3 + n_rays # [rho, alpha, d_theta] + lidar
    action_dim = 2
    
    agent = get_navigation_agent(model_name, obs_dim, action_dim, device)
    normalizer = ObservationNormalizer(max_dist=5.0, lidar_range=env.rob_cfg['lidar_range'])

    # 3. Prepare Video Path
    video_path = None
    if save_video:
        os.makedirs("_documentations/navigation_videos", exist_ok=True)
        filename = f"nav_{model_name}_seed{seed}.mp4".replace("/", "_")
        video_path = os.path.join("_documentations/navigation_videos", filename)
        print(f"📹 Video path set to: {video_path}")

    print(f"▶️  Starting Autonomous Navigation with model: {model_name}...")
    
    # 4. Run Simulation
    # Note: Using the navigation_simulation function designed for autonomous agents
    result = navigation_simulation(
        env, 
        agent, 
        normalizer, 
        render=True, 
        max_steps=1000, 
        video_path=video_path
    )

    # 5. Display Metrics
    if result:
        status_str = "✅ SUCCESS" if result["is_success"] else ("💥 COLLISION" if result["collision"] else "⏱️ TIMEOUT")
        
        metrics_list = [
            ("Status", status_str),
            ("Steps to Goal", f"{result['steps']}"),
            ("Total Reward", f"{result['total_reward']:.2f}"),
            ("Path Length", f"{result['path_length']:.2f} m"),
    
            ("Safety Margin", f"{result['safety_margin']:.3f} m"),
            ("Energy", f"{result['energy_consumption']:.2f}"),
            ("Final Distance", f"{result['final_distance_to_goal']:.4f} m"),
            ("Final Angle Error", f"{result['final_angle_to_goal']:.4f} rad")
        ]

        # Formatting table (borrowed from your reference script)
        num_columns = 2
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
    parser = argparse.ArgumentParser(description="Run visual demonstration of the Navigation Agent.")
    parser.add_argument("--model", type=str, required=True, help="Folder name of the experiment or path to .pth file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save_video", action="store_true", help="Saves the simulation as an .mp4 file")
    
    args = parser.parse_args()
    effective_seed = args.seed if args.seed is not None else random.randint(0, 10000)

    run_nav_episode(model_name=args.model, seed=effective_seed, save_video=args.save_video)

if __name__ == "__main__":
    main()