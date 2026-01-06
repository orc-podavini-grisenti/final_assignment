import numpy as np
import pandas as pd

from utils.simulation import run_simulation

class ControllerBenchmark:
    def __init__(self, env, planner, controllers):
        self.env = env
        self.planner = planner
        self.controllers = controllers

    def run(self, n_trials):
        rows = []       # List to store flat dictionary rows for the DataFrame
        trial_data = {k: [] for k in self.controllers} # Store full trajectories if needed

        print(f"🚀 Starting Benchmark over {n_trials} trials...")

        # --- 1. Run all Trials ---
        for trial in range(n_trials):
            # --- OUTER LOOP: MAP GENERATION ---
            # 1. Full Reset: Generates new random goal and obstacles
            self.env.reset()
            
            # 2. Capture the environment configuration (Start/Goal)
            start = self.env.state.copy()
            goal = self.env.goal.copy()
            
            # 3. Plan ONCE for this specific map
            path = self.planner.get_path(start, goal)
            
            if path is None:
                print(f"⚠️ Trial {trial}: Planner failed to find path. Skipping.")
                continue

            # --- INNER LOOP: CONTROLLER EVALUATION ---
            for name, ctrl in self.controllers.items():
                
                # 4. ROBOT RESET ONLY
                # We must reset the robot to the exact start pose of this trial
                # WITHOUT regenerating obstacles or the goal.
                self.env.reset_robot()
               
                # Sanity Check: Ensure state matches planner start
                if not np.allclose(self.env.state, start, atol=1e-5):
                    print(f'❌ Error: Robot reset failed in Trial {trial} for {name}.')
                    break

                # 5. Run Trial via run_simulation
                run_stats = run_simulation(self.env, path, ctrl, render=False)

                # Store detailed stats (trajectories) for plotting later
                trial_data[name].append(run_stats)

                # 6. Extract Scalar Metrics for DataFrame
                # We map the keys from run_simulation to readable CSV columns
                row = {
                    "Controller": name,
                    "Trial": trial,
                    
                    # Outcomes
                    "Success": run_stats["is_success"],
                    "Collision": run_stats["collision"],
                    "Time_Limit": run_stats["truncated"],
                    "Sim_Time": run_stats["sim_time"],
                    "Steps": run_stats["steps"],
                    
                    # Accuracy
                    "Final_Dist_Error": run_stats["distance_error"],
                    "Final_Head_Error": run_stats.get("heading_error", 0.0),
                    "Mean_CTE": run_stats["mean_error"],
                    "Max_CTE": run_stats["max_error"],
                    
                    # Efficiency & Quality
                    "Tortuosity": run_stats.get("tortuosity", 1.0),
                    "Energy": run_stats.get("energy_consumption", 0.0),
                    "Smoothness": run_stats.get("control_smoothness", 0.0),
                    "Avg_Lin_Vel": run_stats.get("mean_v", 0.0),
                    "Avg_Ang_Vel": run_stats.get("mean_omega", 0.0)
                }
                rows.append(row)

        # --- 2. Process Results ---
        results_df = pd.DataFrame(rows)

        # Calculate Mean and Std Dev for all numeric columns, grouped by Controller
        # We filter for numeric types to avoid crashing on boolean/string columns
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        
        summary_df = results_df.groupby("Controller")[numeric_cols].agg(['mean', 'std'])
        
        # Flatten MultiIndex columns (e.g., ('Energy', 'mean') -> 'Energy_mean')
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]       

        print("✅ Benchmark Complete.")
        return results_df, trial_data, summary_df