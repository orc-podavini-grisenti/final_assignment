import numpy as np
import torch
import imageio


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



def run_simulation(env, path, controller, render=False, max_steps=500, video_path=None):
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
    
    # Frame buffer for video
    frames = []
    save_video = video_path is not None
    
    
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
            # If we are saving a video, we need the RGB array
            if save_video:
                # Capture frame for video (ensure render_mode is "rgb_array")
                frame = env.render()
                if frame is not None:
                    frame = frame[:, :, ::-1]
                    frames.append(frame)
            else:
                # Normal human visualization
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
    

    # --- SAVE VIDEO ---
    if save_video and len(frames) > 0:
        dt = env.env_cfg.get("dt", 0.05) 
        fps = int(1 / dt)

        print(f"🎬 Saving video at {fps} FPS to match real-time...")
        imageio.mimsave(video_path, frames, fps=fps)


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




from utils.reward import NavigationReward
def navigation_simulation(env, agent, normalizer, render=False, ideal_path=None, max_steps=1000, video_path=None):
    """
    Runs an autonomous navigation episode using a trained RL agent.
    
    Returns:
        dict: Metrics focused on success, safety, and efficiency.
    """
    obs = env.get_obs()
    frames = []
    save_video = video_path is not None

    reward_calculator = NavigationReward(env.rob_cfg['v_max'], env.env_cfg['dt'])


    if render:
        env.render_mode = "human"
    
    metrics = {
        "rewards": [],
        "positions": [],
        "steps": 0,
        "energy": 0.0,
        # "min_obstacle_dist": float('inf')
    }
    
    # Starting parameters for efficiency calculation
    start_pos = np.array(env.state[:2])
    goal_pos = np.array(env.goal[:2])
    initial_distance = np.linalg.norm(start_pos - goal_pos)

    success = False
    collision = False

    # --- SIMULATION LOOP ---
    for t in range(max_steps):
        metrics["steps"] = t + 1
        
        # 1. Normalize the state (rho, alpha, d_theta + lidar rays)
        normalized_obs = normalizer.normalize(obs)
        device = next(agent.parameters()).device
        normalized_obs = torch.from_numpy(normalized_obs).float().to(device)
        if normalized_obs.dim() == 1:
            normalized_obs = normalized_obs.unsqueeze(0) # (Dim,) -> (1, Dim)
        # print('N OBS:', normalized_obs)
        
        # 2. Get Deterministic Action for Evaluation
        with torch.no_grad():
            action = agent.get_action(normalized_obs, deterministic=True)[0]
            # print('ACTION: ', action)
            if action[0] > env.rob_cfg["v_max"]: print("⚠️ Action Linear Velocity ", action[0], " over robot limits: ", env.rob_cfg["v_max"])
            if action[1] > env.rob_cfg["w_max"]: print("⚠️ Action Angular Velocity ", action[1], " over robot limits: ", env.rob_cfg["w_max"])
        
        # Clip action for environment (physical limits)
        action_np = action.cpu().numpy()
        action_clipped = np.clip(action_np, -1.0, 1.0)
        
        # 3. Step Environment
        new_obs, terminated, truncated, info = env.step(action_clipped)
        reward = reward_calculator.compute_reward(new_obs, action_clipped, info, truncated)

        
        # 4. Log Metrics
        metrics["rewards"].append(reward)
        metrics["positions"].append(env.state[:2].copy())
        metrics["energy"] += (action_clipped[0]**2 + action_clipped[1]**2) * env.env_cfg["dt"]
        
        # Track safety using raw lidar data (usually from index 3 onwards)
        lidar_data = new_obs[3:] 
        if lidar_data is not None and lidar_data.size > 0:
            # Ensure the key exists in metrics before updating it
            if "min_obstacle_dist" not in metrics:
                metrics["min_obstacle_dist"] = np.min(lidar_data)
            else:
                metrics["min_obstacle_dist"] = min(metrics["min_obstacle_dist"], np.min(lidar_data))
        else:
            metrics["min_obstacle_dist"] = None

        # 5. Rendering
        if render:
            env.set_render_trajectory(metrics["positions"])
            if ideal_path is not None: env.set_render_trajectory(ideal_path, second_path = True)
            
            if save_video:
                frame = env.render() 
                if frame is not None:
                    frame = frame[:, :, ::-1]
                    frames.append(frame)
            else:
                env.render()

        obs = new_obs
        if terminated or truncated:
            success = info.get('is_success', False)
            collision = info.get('collision', False)
            break

        # if obs[0] <= 0.1:
            # print('SIMULATION: position reached but with the wrong orientation, ', obs[2], 'rad')

    # --- FINAL CALCULATIONS ---
    total_dist_traveled = 0.0
    if len(metrics["positions"]) > 1:
        pts = np.array(metrics["positions"])
        total_dist_traveled = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    
    if save_video and len(frames) > 0:
        fps = int(1 / env.env_cfg.get("dt", 0.05))
        imageio.mimsave(video_path, frames, fps=fps)

    return {
        "is_success": success,
        "collision": collision,
        "steps": metrics["steps"],
        "total_reward": sum(metrics["rewards"]),
        "path_length": total_dist_traveled,

        # Goal Errors 
        "final_distance_to_goal": obs[0],
        "final_angle_to_goal": obs[2],


        "safety_margin": metrics["min_obstacle_dist"],
        "energy_consumption": metrics["energy"]
    }