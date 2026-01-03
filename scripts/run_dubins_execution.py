import cv2
import numpy as np
import time

from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner
from controllers.lyapunov_controller import LyapunovController, LyapunovParams
from controllers.rl_controller import RLController

def execute_dubins():
    # 1. Initialize
    env = UnicycleEnv()
    env.render_mode = "human"
    
    # Planner & Controller
    planner = DubinsPlanner(curvature_max=1.5, step_size=0.05) 
    # controller = LyapunovController(LyapunovParams(K_P=2.0, K_THETA=5.0))
    controller = RLController(model_path="models_saved/experiments/run_20260103_101956/policy_model.pth")

    obs, info = env.reset()
    
    # 2. Plan Path
    start_pose = env.state
    goal_pose = env.goal
    
    print(f"Planning from {start_pose} to {goal_pose}")
    path = planner.get_path(start_pose, goal_pose)
    
    if path is None:
        print("Planning failed!")
        return

    # Visuals: Inject path into env for rendering
    env.set_render_trajectory(path[:, :2]) 

    # 3. Execution Loop
    print("Executing Path...")
    
    V_CRUISE = 0.4 
    path_idx = 0
    max_idx = len(path) - 1
    
    WAYPOINT_TOLERANCE = 0.1  # Distance to switch to next waypoint
    PARKING_DIST = 0.3        # Distance to switch to parking mode

    while True:
        # --- A. STRATEGY SELECTION ---
        # Calculate distance to the final goal
        dist_to_final = np.linalg.norm(env.state[:2] - goal_pose[:2])
        
        target_x, target_y, target_theta = 0, 0, 0
        v_ref, omega_ref = 0, 0
        mode_debug = ""

        if dist_to_final < PARKING_DIST:
            # === MODE: PARKING ===
            # Target is the absolute final goal
            target_x, target_y, target_theta = goal_pose
            
            # Zero feed-forward velocity -> Activates Stabilization logic
            v_ref = 0.0
            omega_ref = 0.0
            mode_debug = "PARKING"
            
        else:
            # === MODE: TRACKING ===
            # Target is the current waypoint on the curve
            # path[i] contains [x, y, theta, k]
            target_x, target_y, target_theta, target_k = path[path_idx]
            
            # Feed-forward tracking commands
            v_ref = V_CRUISE
            omega_ref = V_CRUISE * target_k
            mode_debug = f"TRACKING (Idx {path_idx}/{max_idx})"

        # --- B. CONSTRUCT OBSERVATION ---
        # Calculate error relative to the SELECTED target (Waypoint or Goal)
        rx, ry, rtheta = env.state
        
        dx = target_x - rx
        dy = target_y - ry
        rho = np.sqrt(dx**2 + dy**2)
        
        # Angle to target relative to robot heading
        alpha = (np.arctan2(dy, dx) - rtheta + np.pi) % (2 * np.pi) - np.pi
        
        # Orientation error relative to robot heading
        d_theta = (target_theta - rtheta + np.pi) % (2 * np.pi) - np.pi
        
        # This is the 'obs' we feed to the controller
        tracking_obs = np.array([rho, alpha, d_theta])

        # --- C. GET CONTROL ACTION ---
        action = controller.get_action(tracking_obs, v_ref=v_ref, omega_ref=omega_ref)

        # --- D. STEP ENVIRONMENT ---
        _, _, terminal, truncated, info = env.step(action)
        env.render()
        
        # --- E. UPDATE LOGIC ---
        
        # Update Waypoint (Only if in Tracking Mode)
        if dist_to_final >= PARKING_DIST:
            # If close to current waypoint, move to next
            if rho < WAYPOINT_TOLERANCE and path_idx < max_idx:
                path_idx += 1
        
        # Debug Print (Optional)
        print(f"Mode: {mode_debug} | Dist: {dist_to_final:.3f} | Action: {action}")

        # Check finish conditions
        if terminal:
            if info.get('is_success', False):
                print(f"🎉 Destination Reached! Final Error: {dist_to_final:.4f}m")
                break
            if info.get('collision', False):
                print("❌ Collision!")
                break

        if truncated:
            print("⏱ Time Limit Exceeded!")
            break

        # Quit on ESC
        if cv2.waitKey(20) == 27:
            break

    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    execute_dubins()