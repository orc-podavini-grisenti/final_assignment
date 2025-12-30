import cv2
import numpy as np
import time

from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner
from controllers.lyapunov_controller import LyapunovController, LyapunovParams

def execute_dubins():
    # 1. Initialize
    env = UnicycleEnv()
    env.render_mode = "human"
    
    # Planner & Controller
    planner = DubinsPlanner(curvature_max=1.5, step_size=0.05) # Small step size for smooth tracking
    controller = LyapunovController(LyapunovParams(K_P=2.0, K_THETA=5.0))

    obs, info = env.reset()
    
    # 2. Plan Path
    start_pose = env.state
    goal_pose = env.goal
    
    print(f"Planning from {start_pose} to {goal_pose}")
    # Returns [x, y, theta, k]
    path = planner.get_path(start_pose, goal_pose)
    
    if path is None:
        print("Planning failed!")
        return

    # Visuals
    env.set_render_trajectory(path[:, :2]) # Send just X,Y for plotting

    # 3. Execution Loop
    print("Executing Path...")
    
    # Reference velocity (Cruise speed)
    V_CRUISE = 0.5 
    
    # We iterate through the path points. 
    # Since the planner generates spatial points, we assume we want to visit 
    # one point per simulation step (or use a 'Lookahead' index).
    # For simplicity, we simply track the "next" point in the list.
    
    path_idx = 0
    max_idx = len(path) - 1
    
    # Tolerance to switch to next waypoint
    WAYPOINT_TOLERANCE = 0.1 

    while True:
        # A. Determine Current Target Waypoint
        # Get the reference state from the path
        ref_x, ref_y, ref_theta, ref_k = path[path_idx]
        
        # Calculate Feed-forward commands
        # v_ref = V_CRUISE
        # omega_ref = v_ref * curvature
        v_ref = V_CRUISE
        omega_ref = V_CRUISE * ref_k
        
        # B. Construct Observation relative to this specific waypoint
        # (We manually calculate the error obs because 'env.step' returns obs relative to the FINAL goal)
        rx, ry, rtheta = env.state
        
        dx = ref_x - rx
        dy = ref_y - ry
        rho = np.sqrt(dx**2 + dy**2)
        alpha = (np.arctan2(dy, dx) - rtheta + np.pi) % (2 * np.pi) - np.pi
        d_theta = (ref_theta - rtheta + np.pi) % (2 * np.pi) - np.pi
        
        # Current 'Tracking' Observation
        tracking_obs = np.array([rho, alpha, d_theta])

        # C. Get Control Action
        # Pass the feed-forward references!
        action = controller.get_action(tracking_obs, v_ref=v_ref, omega_ref=omega_ref)
        
        # D. Step Environment
        # Note: We ignore the env's reward/done here because we are following our own path task
        _, _, collision, _, _ = env.step(action)
        env.render()
        
        # E. Update Waypoint Logic (Simple pure pursuit style)
        if rho < WAYPOINT_TOLERANCE and path_idx < max_idx:
            path_idx += 1
            
        # Check finish conditions
        dist_to_final = np.linalg.norm(env.state[:2] - goal_pose[:2])
        if dist_to_final < 0.2 and path_idx == max_idx:
            print("🎉 Destination Reached!")
            break
            
        if collision:
            print("❌ Collision!")
            break

        # Quit on ESC
        if cv2.waitKey(20) == 27:
            break

    print("Execution finished.")

if __name__ == "__main__":
    execute_dubins()