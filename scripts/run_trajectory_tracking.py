import argparse
import time
import math
import random
from tabulate import tabulate
from utils.simulation import run_simulation
from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner

# Import your controllers
from controllers.lyapunov_controller import LyapunovController, LyapunovParams
from controllers.rl_controller import RLController

"""
HOW TO RUN THIS SCRIPT:
-----------------------
1. RL only with specific seed:     $ python ./scripts/run_trajectory_tracking.py --type rl --seed 42
2. Lyapunov only:                  $ python ./scripts/run_trajectory_tracking.py --type lyapunov --seed 123
3. Both with same seed:            $ python ./scripts/run_trajectory_tracking.py --type both --seed 65
"""

def get_controller(c_type):
    """Factory function to instantiate the requested controller."""
    if c_type == "rl":
        print("🔵 Loading RL Controller...")
        model_path = "training/experiments/run_20260105_151445/policy_model.pth"
        return RLController(model_path=model_path)
    
    elif c_type == "lyapunov":
        print("🟢 Loading Lyapunov Controller...")
        return LyapunovController(LyapunovParams(K_P=2.0, K_THETA=5.0))
    
    return None




def run_visual_episode(controller_type, seed=None):
    """Runs a single simulation episode with the specified controller."""
    env = UnicycleEnv()
    
    # Use the provided seed to ensure reproducibility
    if seed is not None:
        print(f"🎲 Using Seed: {seed}")
        try:
            env.reset(seed=seed)
        except TypeError:
            # Fallback if env doesn't support seed in reset; 
            # Note: some envs require random.seed(seed) or np.random.seed(seed)
            env.reset() 
    else:
        env.reset()

    planner = DubinsPlanner(curvature=1, step_size=0.05)
    path = planner.get_path(env.state, env.goal)

    controller = get_controller(controller_type)

    print(f"▶️  Starting visual demonstration with {controller_type.upper()}...")
    result = run_simulation(env, path, controller, render=True)

    if result:
        # Success/Failure logic
        if result["is_success"]:
            status_str = "✅ SUCCESS"
        else:
            reasons = []
            if result.get("truncated"): reasons.append("Time Limit")
            if result.get("collision"): reasons.append("Collision")
            status_str = f"❌ FAILED ({', '.join(reasons)})" if reasons else "❌ FAILED"

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




def run_double_demo(seed):
    """Runs RL and Lyapunov controllers consecutively on the same scenario."""
    print("\n" + "="*50)
    print(f"DOUBLE DEMO MODE: Comparing RL vs Lyapunov (Seed: {seed})")
    print("="*50 + "\n")

    # Run RL
    run_visual_episode("rl", seed=seed)
    print("\n" + "-"*30 + "\n")
    # Run Lyapunov
    run_visual_episode("lyapunov", seed=seed)




def main():
    parser = argparse.ArgumentParser(description="Run visual demonstrations.")
    parser.add_argument(
        "--type", 
        type=str, 
        default="rl", 
        choices=["rl", "lyapunov", "both"],
        help="Choose controller: 'rl', 'lyapunov', or 'both'"
    )
    # Added Seed Parameter
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the environment initialization to ensure reproducibility."
    )
    
    args = parser.parse_args()

    # If no seed is provided and we are doing 'both', 
    # we generate one here so both controllers use the same one.
    effective_seed = args.seed if args.seed is not None else random.randint(0, 10000)

    if args.type == "both":
        run_double_demo(seed=effective_seed)
    else:
        run_visual_episode(args.type, seed=effective_seed)

if __name__ == "__main__":
    main()