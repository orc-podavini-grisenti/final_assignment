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
1. RL only:       $ python run_single_demo.py --type rl
2. Lyapunov only: $ python run_single_demo.py --type lyapunov
3. Both:          $ python run_single_demo.py --type both
"""

def get_controller(c_type):
    """Factory function to instantiate the requested controller."""
    if c_type == "rl":
        print("🔵 Loading RL Controller...")
        # Ensure this path is correct relative to where you run the script
        model_path = "training/experiments/run_20260105_151445/policy_model.pth"
        return RLController(model_path=model_path)
    
    elif c_type == "lyapunov":
        print("🟢 Loading Lyapunov Controller...")
        return LyapunovController(LyapunovParams(K_P=2.0, K_THETA=5.0))
    
    return None




def run_visual_episode(controller_type, seed=None):
    """Runs a single simulation episode with the specified controller."""
    
    # 1. Initialize Environment & Planner
    env = UnicycleEnv()
    
    # Passing a seed ensures both controllers solve the exact same scenario 
    # if run consecutively (assuming UnicycleEnv supports seeding in reset)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset() # Fallback if env doesn't support explicit seed arg
    else:
        env.reset()

    planner = DubinsPlanner(curvature=1, step_size=0.05)
    path = planner.get_path(env.state, env.goal)

    # 2. Instantiate Controller
    controller = get_controller(controller_type)

    # 3. Run Simulation
    print(f"▶️  Starting visual demonstration with {controller_type.upper()}...")
    result = run_simulation(env, path, controller, render=True)

    # 4. Print Organized Results
    if result:
        # 4.1. Prepare Status String 
        if result["is_success"]:
            status_str = "✅ SUCCESS"
        else:
            reasons = []
            if result.get("truncated"): reasons.append("Time Limit")
            if result.get("collision"): reasons.append("Collision")
            status_str = f"❌ FAILED ({', '.join(reasons)})" if reasons else "❌ FAILED"

        dt_val = getattr(env, 'dt', 0.05)

        # 4.2. Define All Data Points 
        # List of tuples: (Label, Value)
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

        # 4.3. Format into Columns (Column-Major Order) 
        num_columns = 3  # We want 3 pairs of (Metric, Value) horizontally
        # Calculate required rows: ceil(total_items / num_columns)
        num_rows = math.ceil(len(metrics_list) / num_columns)
        
        table_rows = []
        
        for r in range(num_rows):
            row_data = []
            for c in range(num_columns):
                # Calculate index in the original list: index = col * num_rows + row
                idx = c * num_rows + r
                
                if idx < len(metrics_list):
                    label, value = metrics_list[idx]
                    row_data.extend([label, value])
                else:
                    # Pad empty cells if list runs out
                    row_data.extend(["", ""])
            
            table_rows.append(row_data)

        # 4.4. Print Table 
        # Create headers: Metric | Value | Metric | Value ...
        headers = ["Metric", "Value"] * num_columns
        
        print("\n" + tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
    
    # Small pause between demos
    time.sleep(1.0)




def run_double_demo():
    """Runs RL and Lyapunov controllers consecutively on the same scenario."""
    print("\n" + "="*50)
    print("DOUBLE DEMO MODE: Comparing RL vs Lyapunov")
    print("="*50 + "\n")

    # Pick a fixed seed so they face the same start/goal configuration
    demo_seed = 42 
    dynamic_seed = random.randint(0, 10000)

    # Run RL
    run_visual_episode("rl", seed=dynamic_seed)
    
    print("\n" + "-"*30 + "\n")
    
    # Run Lyapunov
    run_visual_episode("lyapunov", seed=dynamic_seed)





def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run visual demonstrations.")
    parser.add_argument(
        "--type", 
        type=str, 
        default="rl", 
        choices=["rl", "lyapunov", "both"],
        help="Choose controller: 'rl', 'lyapunov', or 'both'"
    )
    args = parser.parse_args()

    if args.type == "both":
        run_double_demo()
    else:
        run_visual_episode(args.type)

if __name__ == "__main__":
    main()