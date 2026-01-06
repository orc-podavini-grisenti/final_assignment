import numpy as np
import torch


V_CRUISE = 0.4 
WAYPOINT_TOLERANCE = 0.1


def cross_track_error(position, path):
    """
    Calculates the minimum distance from the current position to the path.
    path: (N, 3) or (N, 4) array [x, y, theta, ...]
    position: (2,) array [x, y]
    """
    min_dist = float("inf")
    best_idx = 0

    # Iterate over all path segments
    for i in range(len(path) - 1):
        p1 = path[i, :2]
        p2 = path[i + 1, :2]
        seg = p2 - p1

        # Handle simplified case where points are identical
        if np.linalg.norm(seg) < 1e-6:
            d = np.linalg.norm(position - p1)
        else:
            # Project position onto the line segment defined by p1 and p2
            # t is the projection factor (0.0 to 1.0 clamped to segment)
            t = np.dot(position - p1, seg) / np.dot(seg, seg)
            t = np.clip(t, 0.0, 1.0)
            proj = p1 + t * seg
            d = np.linalg.norm(position - proj)

        if d < min_dist:
            min_dist = d
            best_idx = i

    return min_dist, best_idx



def run_simulation(env, path, controller, render=False, max_steps=500):
    """
    Core execution logic shared between single-run and evaluation.
    Returns: dictionary containing summary stats and full time-series metrics.
    """
    start_pose = env.state
    goal_pose = env.goal
    
    if path is None:
        return None

    if render:
        env.render_mode = "human"
        env.set_render_trajectory(path[:, :2])
    
    
    # Initialize the Step Metrics Containers 
    metrics = {
        "cte": [], "linear_vel": [], "angular_vel": [], 
        "positions": [], "thetas": [], "time": []
    }


    # Utils 
    path_idx = 0                    # counter to track the waypoints of the path
    max_idx = len(path) - 1         # number of waypoints in the path
    dt = env.env_cfg["dt"]          # enviroment dt 

    prev_action = np.zeros(2)            # store the privius action, used to compute the smoothness
    prev_pos = np.array(start_pose[:2])  # store the privius position, used to compute the distance traveled

    # Performance Trackers
    final_dist = 0.0                # final distance from the robot and the goal
    final_heading_error = 0.0       # final angle between the robot and the goal

    steps_taken = 0                 # number of step taken in the simulation
    total_energy = 0.0              # total energy used by the robot in the sim        
    total_smoothness = 0.0          # total smoothness of the control inputs
    total_distance_traveled = 0.0   # total disctance driven by the robot
    
    # Calculate Reference Path Length (for Tortuosity)
    ref_path_length = np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))

    # Flags
    success = False         # the robot has reached the goal inside the env thresholds
    collision = False       # the robot has collided with an obstacle
    truncated = False       # max step reached
             

    # --- CONTROL LOOP ---
    for t in range(max_steps):
        steps_taken = t
        current_time = t * dt
        
        # 1. Capture Current State
        rx, ry, rtheta = env.state
        current_pos = np.array([rx, ry])
        

        # --- A. GET THE STEP TARGET ---
        target_x, target_y, target_theta, target_k = path[path_idx]
        v_ref, omega_ref = V_CRUISE, V_CRUISE * target_k


        # --- B. CONSTRUCT OBSERVATION (For Controller) ---
        dx, dy = target_x - rx, target_y - ry
        rho = np.sqrt(dx**2 + dy**2)
        alpha = (np.arctan2(dy, dx) - rtheta + np.pi) % (2 * np.pi) - np.pi
        d_theta = (target_theta - rtheta + np.pi) % (2 * np.pi) - np.pi
        
        tracking_obs = np.array([rho, alpha, d_theta])


        # --- C. GET CONTROL ---
        action = controller.get_action(tracking_obs, v_ref=v_ref, omega_ref=omega_ref)
        # update the render
        if render:
            env.render()
        

        # --- D. LOG METRICS --- 
        metrics["positions"].append((rx, ry))
        metrics["thetas"].append(rtheta)
        metrics["linear_vel"].append(float(action[0]))
        metrics["angular_vel"].append(float(action[1]))
        metrics["time"].append(current_time)


        # --- E. STEP ENVIRONMENT ---
        _, terminal, truncated, info = env.step(action)


        # --- F. CALCULATE METRIX ---
        # Capture the new Current State
        rx, ry, rtheta = env.state
        current_pos = np.array([rx, ry])

        # 0. Distance to the very end of the path
        dist_to_final = np.linalg.norm(current_pos - goal_pose[:2])
        final_dist = dist_to_final

        # 1. CTE. Calculate the true perpendicular distance to the closest path segment
        real_cte, _ = cross_track_error(current_pos, path)
        metrics["cte"].append(real_cte)  # <--- Now logging the Real CTE

        # 2. Energy: Sum of squares of control inputs
        total_energy += (action[0]**2 + action[1]**2) * dt

        # 3. Smoothness: Change in control (delta action)
        total_smoothness += np.linalg.norm(action - prev_action)

        # 4. Traveled Distance: For tortuosity
        total_distance_traveled += np.linalg.norm(current_pos - prev_pos)

        # Update trackers
        prev_action = action.copy()
        prev_pos = current_pos.copy()
        
        
        # --- G. WAYPOINT SWITCHING LOGIC --- 
        if rho < WAYPOINT_TOLERANCE and path_idx < max_idx:
            path_idx += 1


        # --- H. TERMINATION CHECKS ---
        if t == max_steps - 1:
            print(f"SIMULATION: Truncated after max_steps: {max_steps}   env seed: {env.current_seed}")
            truncated = True

        if terminal or truncated:
            success = info.get('is_success', False)
            collision = info.get('collision', False)

            # compute the heading error 
            current_theta = env.state[2]
            goal_theta = env.goal[2]
            final_heading_error = np.arctan2(np.sin(current_theta - goal_theta), np.cos(current_theta - goal_theta))
            break
            

    # --- FINAL RETURN ---
    return {
        # Outcome
        "is_success": success,
        "collision": collision,
        "truncated": truncated,
        "distance_error": final_dist,
        "heading_error": final_heading_error,
        "progress": path_idx / max_idx if max_idx > 0 else 1.0,

        # Performance Metrics
        "mean_error": np.mean(metrics["cte"]) if metrics["cte"] else 0.0,
        "max_error": np.max(metrics["cte"]) if metrics["cte"] else 0.0,
        "mean_v": np.mean(metrics["linear_vel"]) if metrics["linear_vel"] else 0.0,
        "mean_omega": np.mean(metrics["angular_vel"]) if metrics["angular_vel"] else 0.0,
        
        # Efficiency & Comfort
        "energy_consumption": total_energy,
        "control_smoothness": total_smoothness / (steps_taken + 1), # Avg jerk
        "tortuosity": total_distance_traveled / ref_path_length if ref_path_length > 0 else 1.0,
        
        # Meta
        "steps": steps_taken + 1,
        "sim_time": metrics["time"][-1] if metrics["time"] else 0.0,
        "trajectory": metrics 
    }