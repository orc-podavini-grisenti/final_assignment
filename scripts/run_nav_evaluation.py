import argparse
import random
import os
from evaluation.navigation_evaluation import (
    evaluate_single_nav_model, 
    # evaluate_single_nav_model_path, 
    navigation_comparison_analysis
)

"""
NAVIGATION POLICY EVALUATION & COMPARISON SCRIPT
------------------------------------------------
This script provides a framework to evaluate trained RL Navigation models.
Unlike trajectory tracking, navigation is evaluated on its ability to reach a 
goal autonomously while avoiding obstacles and optimizing path efficiency.

It operates in two modes:

1. SINGLE MODE (--mode single):
   - Conducts two distinct evaluations for a specific model:
        a) General Performance: Success Rate, Collision Rate, Safety, and Energy.
        b) Path Quality: Efficiency compared to the ideal Dubins path.
   - Saves results into two separate CSV databases for long-term tracking.
   - Generates raw data for statistical significance testing.

2. COMPARE MODE (--mode compare):
   - Aggregates data from the 'NAV_general_performance.csv' and 'NAV_path_quality.csv'.
   - Generates "Leaderboards" ranking models based on Reliability and Path Optimality.
   - Produces statistical analysis (Wilcoxon/T-Tests) and comparative boxplots.

EXAMPLE USAGE:
1. SINGLE MODE:  python ./scripts/run_nav_evaluation.py --mode single --model path/to/model.pth --name "nav_v1"
2. COMPARE MODE: python ./scripts/run_nav_evaluation.py --mode compare --comp_type general

REAL USAGE: 
1 NAV 1 SINGLE MODE:
    python ./scripts/run_nav_evaluation.py --mode single --model training/nav_1/nav_agent_final.pth --name "nav_v1"
"""

def get_args():
    parser = argparse.ArgumentParser(description="Run Navigation Evaluation")
    parser.add_argument("--mode", type=str, choices=["single", "compare"], required=True, 
                        help="Evaluation mode: 'single' for one model, 'compare' for leaderboard.")
    parser.add_argument("--model", type=str, help="Path to the trained .pth navigation model.")
    parser.add_argument("--name", type=str, help="Model alias for CSV logging (e.g., 'v1_baseline').")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible scenarios.")
    
    # Comparison specific
    parser.add_argument("--comp_type", type=str, choices=["general", "path"], default="general",
                        help="Which leaderboard to display in compare mode.")
    
    return parser.parse_args()

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    if args.mode == "single":
        if not args.model or not args.name:
            print("❌ Error: --model and --name are required for single mode.")
        else:
            print(f"🚀 Starting Full Evaluation for Navigation Model: {args.name}")
            
            # 1. Run General Performance Evaluation (Success, Collision, Safety)
            # This function saves results to 'NAV_general_performance.csv' internally
            evaluate_single_nav_model(
                model_path=args.model, 
                model_alias=args.name, 
                num_episodes=args.episodes, 
                seed=args.seed
            )
            
            print("-" * 50)
            
            # 2. Run Path Quality Evaluation (vs. Dubins Planner)
            # This function saves results to 'NAV_path_quality.csv' internally
            # evaluate_single_nav_model_path(
            #    model_path=args.model, 
            #    model_alias=args.name, 
            #    num_episodes=args.episodes, 
            #    seed=args.seed
            # )

    elif args.mode == "compare":
        print(f"📊 Running Comparison Analysis [Mode: {args.comp_type.upper()}]")
        # This calls the leaderboard, statistical tests, and boxplot generation
        navigation_comparison_analysis(mode=args.comp_type)