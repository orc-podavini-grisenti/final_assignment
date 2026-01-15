import numpy as np
import torch
import imageio
from utils.reward_2 import NavigationReward

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
        new_obs, terminated, truncated, info = env.step(np.abs(action_clipped))
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