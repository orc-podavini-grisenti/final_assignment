import numpy as np
import torch
import imageio

def navigation_simulation(env, agent, normalizer, render=False, max_steps=1000, video_path=None):
    """
    Runs an autonomous navigation episode using a trained RL agent.
    
    Returns:
        dict: Metrics focused on success, safety, and efficiency.
    """
    state, _ = env.reset()
    frames = []
    save_video = video_path is not None
    
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
        normalized_state = normalizer.normalize(state)
        
        # 2. Get Deterministic Action for Evaluation
        with torch.no_grad():
            action, _, _, _ = agent.get_action(normalized_state, deterministic=True)
        
        # 3. Step Environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # 4. Log Metrics
        metrics["rewards"].append(reward)
        metrics["positions"].append(env.state[:2].copy())
        metrics["energy"] += (action[0]**2 + action[1]**2) * env.env_cfg["dt"]
        
        # Track safety using raw lidar data (usually from index 3 onwards)
        lidar_data = state[3:] 
        if lidar_data:
            metrics["min_obstacle_dist"] = min(metrics["min_obstacle_dist"], np.min(lidar_data))
        else:
            metrics["min_obstacle_dist"] = None

        # 5. Rendering
        if render:
            if save_video:
                frame = env.render() 
                if frame is not None:
                    frames.append(frame)
            else:
                env.render()

        state = next_state
        if terminated or truncated:
            success = info.get('is_success', False)
            collision = info.get('collision', False)
            break

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
        # Navigation Efficiency: Straight-line distance / Actual distance
        "nav_efficiency": initial_distance / total_dist_traveled if total_dist_traveled > 0 else 0,
        "safety_margin": metrics["min_obstacle_dist"],
        "energy_consumption": metrics["energy"]
    }