import argparse
import random
import os
from evaluation.navigation_evaluation import (
    evaluate_single_nav_model, 
    evaluate_single_nav_model_path, 
)

"""
NAVIGATION POLICY EVALUATION & COMPARISON SCRIPT
------------------------------------------------
This script provides a framework to evaluate trained RL Navigation models.
Unlike trajectory tracking, navigation is evaluated on its ability to reach a 
goal autonomously while avoiding obstacles and optimizing path efficiency.

It operates in two modes:

1. NAV MODE:
   General Performance: Success Rate, Collision Rate, Safety, and Energy.

2. PATH MODE (--path_evaluation):
   Path Quality: Efficiency compared to the ideal Dubins path.
   
EXAMPLE USAGE:
1. NAV MODE:  python ./scripts/run_nav_evaluation.py --model path/to/model.pth --name "nav_v1"
2. PATH MODE: python ./scripts/run_nav_evaluation.py --model path/to/model.pth --name "nav_v1" --path_evaluation


REAL USAGE EXAMPLES:
# 1. Evaluate General Performance (Success, Collision, etc.)
python ./scripts/run_nav_evaluation.py --model training/nav_1/nav_actor_final.pth --name "nav_v1" --render

# 2. Evaluate Path Quality (Comparison against Dubins Path)
python ./scripts/run_nav_evaluation.py --model training/nav_1/nav_actor_final.pth --name "nav_v1" --path_evaluation --render

"""

def get_args():
    parser = argparse.ArgumentParser(description="Run Navigation Evaluation")
    parser.add_argument("--model", type=str, help="Path to the trained .pth navigation model.")
    parser.add_argument("--name", type=str, help="Model alias for CSV logging (e.g., 'v1_baseline').")
    
    # Use action="store_true" so these act as flags (no extra argument needed)
    parser.add_argument("--path_evaluation", action="store_true", help="Run the Path Quality Evaluation vs Dubins")
    parser.add_argument("--render", action="store_true", help="Render the evaluation episodes")
    
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducible scenarios.")
    
    return parser.parse_args()

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    
    if not args.model or not args.name:
        print("❌ Error: --model and --name are required for single mode.")
    else:
        print(f"🚀 Starting Full Evaluation for Navigation Model: {args.name}")
        
        
        if not args.path_evaluation: 
            # 1. Run General Performance Evaluation (Success, Collision, Safety)
            # This function saves results to 'NAV_general_performance.csv' internally
            evaluate_single_nav_model(
                model_path=args.model, 
                model_alias=args.name, 
                num_episodes=args.episodes, 
                seed=args.seed,
                render=args.render
            )
        else: 
        
            # 2. Run Path Quality Evaluation (vs. Dubins Planner)
            # This function saves results to 'NAV_path_quality.csv' internally
            evaluate_single_nav_model_path(
                model_path=args.model, 
                model_alias=args.name, 
                num_episodes=args.episodes, 
                seed=args.seed,
                render=args.render
            )