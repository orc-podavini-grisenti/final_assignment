import numpy as np
import torch

def run_simulation(env, planner, controller, render=False, max_steps=500):
    """
    Core execution logic shared between single-run and evaluation.
    Returns: stats dictionary containing performance metrics.
    """
    obs, info = env.reset()
    start_pose = env.state
    goal_pose = env.goal
    path = planner.get_path(start_pose, goal_pose)
    
    if path is None:
        return None

    if render:
        env.render_mode = "human"
        env.set_render_trajectory(path[:, :2])

    # Configuration
    V_CRUISE = 0.4 
    path_idx = 0
    max_idx = len(path) - 1
    WAYPOINT_TOLERANCE = 0.1
    PARKING_DIST = 0.3
    
    ep_errors = []
    
    for t in range(max_steps):
        dist_to_final = np.linalg.norm(env.state[:2] - goal_pose[:2])
        
        # --- A. STRATEGY SELECTION ---
        if dist_to_final < PARKING_DIST:
            target_x, target_y, target_theta = goal_pose
            v_ref, omega_ref = 0.0, 0.0
        else:
            target_x, target_y, target_theta, target_k = path[path_idx]
            v_ref, omega_ref = V_CRUISE, V_CRUISE * target_k

        # --- B. CONSTRUCT OBSERVATION ---
        rx, ry, rtheta = env.state
        dx, dy = target_x - rx, target_y - ry
        rho = np.sqrt(dx**2 + dy**2)
        alpha = (np.arctan2(dy, dx) - rtheta + np.pi) % (2 * np.pi) - np.pi
        d_theta = (target_theta - rtheta + np.pi) % (2 * np.pi) - np.pi
        
        tracking_obs = np.array([rho, alpha, d_theta])
        ep_errors.append(rho)

        # --- C. GET CONTROL & STEP ---
        action = controller.get_action(tracking_obs, v_ref=v_ref, omega_ref=omega_ref)
        _, _, terminal, truncated, info = env.step(action)
        
        if render:
            env.render()

        # --- D. UPDATE LOGIC ---
        if dist_to_final >= PARKING_DIST and rho < WAYPOINT_TOLERANCE and path_idx < max_idx:
            path_idx += 1

        if terminal or truncated:
            break

    # Compile results
    return {
        "is_success": info.get('is_success', False),
        "collision": info.get('collision', False),
        "distance_error": dist_to_final,
        "mean_error": np.mean(ep_errors),
        "steps": t,
        "progress": path_idx / max_idx if max_idx > 0 else 1.0
    }