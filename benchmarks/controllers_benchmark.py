import numpy as np
import pandas as pd
from benchmarks.metrics import compute_metrics
from benchmarks.utils import cross_track_error

class ControllerBenchmark:
    def __init__(self, env, planner, controllers):
        self.env = env
        self.planner = planner
        self.controllers = controllers

    def run_trial(self, controller, path):
        """
        Executes a trial using the robust logic from run_simulation 
        but captures detailed metrics for analysis.
        """
        self.env.set_render_trajectory(path[:, :2])

        # --- 1. Metric Initialization ---
        metrics = {
            "cte": [],
            "linear_vel": [],
            "angular_vel": [],
            "positions": [],
            "sim_time": 0.0,
            "success": False,
            "collision": False
        }

        # --- 2. Configuration (Aligned with Simulation Script) ---
        dt = self.env.env_cfg["dt"]
        V_CRUISE = 0.4
        WAYPOINT_TOLERANCE = 0.1
        PARKING_DIST = 0.3
        GOAL_TOL = 0.1  # Success radius
        MAX_TIME = 30.0
        
        path_idx = 0
        max_idx = len(path) - 1
        goal_pose = path[-1] # Assuming last point is goal

        while metrics["sim_time"] < MAX_TIME:
            # Current State
            state = self.env.state
            rx, ry, rtheta = state
            dist_to_final = np.linalg.norm(state[:2] - goal_pose[:2])

            # --- A. STRATEGY SELECTION ---
            # Use Parking mode if close to end, otherwise follow path
            if dist_to_final < PARKING_DIST:
                target_x, target_y, target_theta = goal_pose[:3]
                v_ref, omega_ref = 0.0, 0.0
            else:
                target_x, target_y, target_theta, target_k = path[path_idx]
                v_ref, omega_ref = V_CRUISE, V_CRUISE * target_k

            # --- B. CONSTRUCT OBSERVATION ---
            dx, dy = target_x - rx, target_y - ry
            rho = np.sqrt(dx**2 + dy**2)
            
            # Crucial: Angle Normalization [-pi, pi]
            alpha = (np.arctan2(dy, dx) - rtheta + np.pi) % (2 * np.pi) - np.pi
            dtheta = (target_theta - rtheta + np.pi) % (2 * np.pi) - np.pi
            
            obs = np.array([rho, alpha, dtheta])

            # --- C. GET CONTROL & STEP ---
            action = controller.get_action(obs, v_ref=v_ref, omega_ref=omega_ref)
            _, _, terminal, truncated, info = self.env.step(action)

            # --- D. RECORD METRICS ---
            # Calculate CTE specifically for analysis (even if controller doesn't use it)
            current_cte, _ = cross_track_error(state[:2], path)
            
            metrics["cte"].append(current_cte)
            metrics["linear_vel"].append(action[0])
            metrics["angular_vel"].append(action[1])
            metrics["positions"].append(state[:2].copy())
            metrics["sim_time"] += dt

            # --- E. UPDATE LOGIC ---
            # Advance waypoint if reached and not at the end
            if dist_to_final >= PARKING_DIST and rho < WAYPOINT_TOLERANCE and path_idx < max_idx:
                path_idx += 1

            # --- F. TERMINATION CHECKS ---
            # 1. Success
            if (dist_to_final < GOAL_TOL and np.abs(dtheta) < GOAL_TOL) or info.get('is_success', False): 
                metrics["success"] = True
                break
            
            # 2. Collision or Gym Termination
            if info.get('collision', False):
                metrics["collision"] = True
                break

        metrics["positions"] = np.array(metrics["positions"])
        return metrics



    def run(self, n_trials):
        rows = []       # csv data rows, one row for each trial, it contain the trial data
        trial_data = {k: [] for k in self.controllers}

        # --- 1. Run all the Trials and collect data ---
        for trial in range(n_trials):
            # --- OUTER LOOP: MAP GENERATION ---
            # 1. Full Reset: Generates new random goal
            self.env.reset()
            
            # 2. Capture the environment configuration
            start = self.env.state.copy()
            goal = self.env.goal.copy()
            
            # 3. Plan ONCE for this specific map
            path = self.planner.get_path(start, goal)
            
            if path is None:
                print(f"Trial {trial}: Planner failed to find path.")
                continue

            # --- INNER LOOP: CONTROLLER EVALUATION ---
            for name, ctrl in self.controllers.items():
                
                # 4. ROBOT RESET ONLY
                # Do NOT call self.env.reset() here! It will change the goal pose
                # Instead, use the self.env.reset_robot() 
                self.env.reset_robot()
               
                # Double check state matches planner start (just in case)
                if not np.allclose(self.env.state, start, atol=1e-5):
                    print(f'Error: Robot reset failed in Trial {trial} for {name}.')
                    print(f'Expected: {start}, Got: {self.env.state}')
                    break

                # 5. Run Trial
                run_stats = self.run_trial(ctrl, path)

                trial_data[name].append(run_stats)

                # 6. Log Metrics
                row = compute_metrics(run_stats)
                row["Controller"] = name
                row["Trial"] = trial
                row["Success"] = run_stats["success"]
                row["Collision"] = run_stats["collision"]
                rows.append(row)

        results_df = pd.DataFrame(rows)

        # Calculate Mean and Std Dev for all numeric columns, grouped by Controller
        summary_df = results_df.groupby("Controller").agg(['mean', 'std'])
        
        # Flatten the column names (e.g., ('RMS_CTE', 'mean') -> 'RMS_CTE_mean')
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]       

        return results_df, trial_data, summary_df