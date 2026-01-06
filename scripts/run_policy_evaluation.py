import argparse
from evaluation.trajectory_tracking_env import evaluate_single_model, append_to_csv, comparison_analysis, radius_sweep


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



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["single", "compare", "sweep"], required=True)
    parser.add_argument("--model", type=str, help="Path to the .pth model")
    parser.add_argument("--name", type=str, help="Model alias for CSV")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    # Sweep specific arguments
    parser.add_argument("--min_r", type=float, default=0.7, help="Minimum radius to test")
    parser.add_argument("--max_r", type=float, default=2.5, help="Maximum radius to test")
    parser.add_argument("--steps_r", type=int, default=15, help="Number of radius increments")
    
    return parser.parse_args()



# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    if args.mode == "single":
        if not args.model:
            print("Error: --model argument is required for single mode.")
            exit(1)
            
        metrics = evaluate_single_model(args.model, args.name, args.episodes, args.seed)
        
        if metrics:
            append_to_csv(metrics)

    elif args.mode == "compare":
        comparison_analysis()
    
    elif args.mode == "sweep":
        if not args.model:
            print("Error: --model required for sweep mode.")
        else:
            radius_sweep(args.model, args.min_r, args.max_r, args.steps_r, args.seed)