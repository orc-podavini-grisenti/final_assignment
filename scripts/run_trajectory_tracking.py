import argparse
from utils.simulation import run_simulation
from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner

# Import your controllers
from controllers.lyapunov_controller import LyapunovController, LyapunovParams
from controllers.rl_controller import RLController

"""
HOW TO RUN THIS SCRIPT:
-----------------------
You can choose which controller to visualize by passing the '--type' argument 
in your terminal.

1. To run the Reinforcement Learning (RL) controller:
   $ python run_single_demo.py --type rl

2. To run the Lyapunov (Classic Control) controller:
   $ python run_single_demo.py --type lyapunov

If you run it without arguments, it defaults to 'rl'.
"""

def run_single_demo():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run a visual demonstration with a selected controller.")
    parser.add_argument(
        "--type", 
        type=str, 
        default="rl", 
        choices=["rl", "lyapunov"],
        help="Choose the controller type: 'rl' or 'lyapunov'"
    )
    
    # Parse the arguments
    args = parser.parse_args()

    # 2. Initialize Environment & Planner
    env = UnicycleEnv()
    planner = DubinsPlanner(curvature_max=1, step_size=0.05)
    
    # 3. Instantiate the selected controller
    if args.type == "rl":
        print("🔵 Loading RL Controller...")
        model_path = "outputs/models_saved/experiments/run_20260103_101956/policy_model.pth"
        controller = RLController(model_path=model_path)
    
    elif args.type == "lyapunov":
        print("🟢 Loading Lyapunov Controller...")
        controller = LyapunovController(LyapunovParams(K_P=2.0, K_THETA=5.0))

    # 4. Run Simulation
    print(f"Starting visual demonstration with {args.type.upper()} controller...")
    result = run_simulation(env, planner, controller, render=True)
    
    # 5. Print Results
    if result:
        if result["is_success"]: status = "SUCCESS"
        else:
            status = "FAILED (unkown reason)"
            if result["truncated"]: status += " (Time Limit)"
            if result["collision"]: status += " (Collision)"

            
        print(f"Result: {status} | Final Error: {result['distance_error']:.4f}m")

if __name__ == "__main__":
    run_single_demo()