import os
import cv2
import math
import yaml

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from envs.obstacle import ObstacleManager


class UnicycleEnv(gym.Env):
    """
    A custom Gymnasium environment for a unicycle robot.
    
    State: [x, y, theta] (global frame)
    Action: [v, omega] (linear and angular velocity)
    Observation: Ego-centric vector 
                 [rho_goal, alpha_goal, theta_goal_rel, d_obs_1, phi_obs_1, ...]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path=None):
        super(UnicycleEnv, self).__init__()

        # --- 1. Load Configuration ---
        if config_path is None:
            # Default to config/env.yaml relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir) 
            config_path = os.path.join(project_root, 'configs', 'env.yaml')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # --- 2. Apply Config Parameters ---
        self.env_cfg = config['environment']    # Enviroment Configuration
        self.rob_cfg = config['robot']          # Robot Configuration
        self.goal_cfg = config['goal']        # Goal Configuration
        self.obs_cfg = config['obstacles']    # Obstacle Configuration
        self.rew_cfg = config['rewards']      # Reward Configuration
        
        
        # --- Action Space ---
        # FIX: Define the arrays explicitly as float32 to match Gymnasium's expectation
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # --- Observation Space ---
        obs_dim = 3 + (2 * self.obs_cfg['n_obstacles'])
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

        # Internal State (Initialize with float32 for consistency)
        self.state = np.zeros(3, dtype=np.float32)  # x, y, theta
        self.goal = np.zeros(3, dtype=np.float32)   # x, y, theta
        self.obstacles = []
        self.current_step = 0

        # --- Modules ---
        self.obstacle_manager = ObstacleManager(lidar_range=self.obs_cfg['lidar_range'])
        
        # --- Render Stuffs --- 
        self.trajectory_buffer = None
        



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Initialize Robot State (x, y, theta)
        self.state = np.array([0.0, 0.0, 0.0])
        
        # 2. Random Goal Generation (e.g., within a 5x5 box)
        self._spawn_goal()
        
        # 3. Initialize Obstacles
        spawn_mode = self.obs_cfg['spawn_mode']
        
        if spawn_mode == 'fixed':
            fixed_obstacles = self.obs_cfg.get('fixed_obstacles', [])
            self.obstacle_manager.generate_fixed_obstacles(fixed_obstacles)
            
            # Validation Loop: Ensure Goal doesn't hit Fixed Obstacles
            max_retries = 100       # use a max retries to avoid infinit loop
            i = 0
            while self.obstacle_manager.check_collision(self.goal, self.goal_cfg['threshold']):
                if i > max_retries:
                    raise RuntimeError("Could not find valid goal position after 100 tries.")
                
                # If both are fixed and colliding, it's a config error
                if self.goal_cfg['spawn_mode'] == 'fixed':
                    raise ValueError("Configuration Error: Fixed Goal is inside a Fixed Obstacle!")
                
                # Try a new random goal
                self._spawn_goal()
                i += 1

        else:
            self.obstacle_manager.generate_random_obstacles(
                num_obstacles=self.obs_cfg['n_obstacles'], 
                bounds=self.env_cfg['world_bounds'],
                robot_pos=self.state,
                goal_pos=self.goal,
                min_clearance=0.5,
                np_random=self.np_random
            )
            # NB: We don't need to check that the goal collide with obstacle; becouse generate_random_obstacles
            # menage alredy it; by generate obstacle that avoid the goal
            
        self.current_step = 0
        
        return self.get_obs(), {}

    def reset_robot(self):
        """
        Resets the robot to the starting state and resets the timer,
        BUT keeps the current obstacles and goal configurations.
        """
        # 1. Reset Robot State (Matches the logic in your main reset)
        self.state = np.array([0.0, 0.0, 0.0])
        
        # 2. Reset Simulation Timer/Counters
        self.current_step = 0
        
        # 3. Return observation (standard Gym API)
        return self.get_obs(), {}



    def step(self, action):
        """
        Updates the robot state based on the provided action.

        Args:
            action (np.ndarray): A 1D array of shape (2,) containing 
                [v_raw, w_raw] in the range [-1, 1].

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): The new state representation.
                - terminated (bool): Whether the episode has ended (goal or collision).
                - truncated (bool): Whether the episode ended due to time limits.
                - info (dict): Diagnostic information (success, collision flags).
        """
        # --- 1. Unpack and Scale Actions ---
        # Map [-1, 1] to physical limits
        v_raw, w_raw = action
        v = np.interp(v_raw, [-1, 1], [ self.rob_cfg['v_min'], self.rob_cfg['v_max'] ])
        w = np.interp(w_raw, [-1, 1], [ self.rob_cfg['w_min'], self.rob_cfg['w_max'] ])
        

        # --- 2. Apply Kinematics (Unicycle Model) ---
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        # theta_dot = w
        x, y, theta = self.state
        dt = self.env_cfg['dt']

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + w * dt
        
        # Normalize theta to [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        
        # Update the state to the new one
        self.state = np.array([x_new, y_new, theta_new])
        self.current_step += 1

        # Define the new observable
        obs = self.get_obs()
        

        # --- 3. Make the Checks ---
              
        # 1) Check Collision
        collision = self.obstacle_manager.check_collision(self.state, self.rob_cfg['radius'])
        
        # 2) Check Goal Reached
        # The goal is reached when: 
        #       -> the distance robot-goal is under the config dist_treshold
        #       -> the angle robot-goal is under the config angle_treshold
        
        # A. Distance to goal check
        dist_to_final = np.linalg.norm(self.state[:2] - self.goal[:2])
        
        # B. Heading alignment check (NEW)
        current_theta = self.state[2]
        goal_theta = self.goal[2]
        heading_error = np.arctan2(np.sin(current_theta - goal_theta), np.cos(current_theta - goal_theta))

        # C. Success Condition
        is_success = (dist_to_final < self.goal_cfg['dist_threshold']) and \
                     (np.abs(heading_error) < self.goal_cfg['angle_threshold']) 

        
        # --- 4. Prepare the output ---
        if collision:
            terminated = True
        elif is_success:
            terminated = True
        else:
            terminated = False
            
        # Truncation (Time limit)
        truncated = self.current_step >= self.env_cfg['max_steps']
        
        info = {
            "is_success": is_success,
            "collision": collision,
        }
        
        return obs, terminated, truncated, info



    ''' Constructs the observation vector. '''
    def get_obs(self):
        """
        Constructs the Ego-Centric Observation Vector.
        
        Instead of Global Cartesian coordinates (x, y), we use a Polar (Robot-Centric) 
        reference frame. This allows the learned policy to be 'Invariant to Rotation' 
        and 'Invariant to Translation'—meaning the agent learns to approach a target 
        regardless of where it is on the map or which way it is facing.
        
        Vector Components:
        ------------------
        1. rho (ρ): [0, inf]        Euclidean distance to the goal.
            Represents 'How far do I need to travel?' (Linear Velocity cue)
        
        2. alpha (α): [-pi, pi]     The angle of the goal *relative* to the robot's current heading.
            Represents 'Where should I steer?' (Angular Velocity cue)
            If alpha = 0, the goal is straight ahead. If alpha > 0, goal is to the left; alpha < 0, goal is to the right.

        3. d_theta (δθ): [-pi, pi]
            The difference between the desired goal orientation and current heading.
            Exential to reach the goal with the desired orientation.s
        
        4. Obstacle Data (d_obs, phi_obs):
           - Vector of closest obstacles, also in polar coordinates (dist, angle).
        """
        x, y, theta = self.state
        gx, gy, gtheta = self.goal
        
        # --- Goal Observations (Polar coordinates in robot frame) ---
        dx = gx - x
        dy = gy - y
        
        rho = np.sqrt(dx**2 + dy**2) # Distance to goal
        target_angle = np.arctan2(dy, dx)
        alpha = target_angle - theta # Angle to goal relative to robot heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi # Normalize
        
        d_theta = gtheta - theta # Orientation error
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        
        obs_vec = [rho, alpha, d_theta]
        
        # --- Obstacle Observations ---
        obs_data = self.obstacle_manager.get_closest_obstacles(self.state, self.obs_cfg['n_obstacles'])
        
        # Flatten obstacle data
        flat_obs = []
        for dist, angle in obs_data:
            flat_obs.extend([dist, angle])
                
        return np.array(obs_vec + flat_obs, dtype=np.float32)



    ''' Private Function that menage the spawn of a random goal.'''
    def _spawn_goal(self):
        # Get World Bounds and Margin
        wb = self.env_cfg['world_bounds']
        margin = self.goal_cfg['spawn_margin']
        
        # Calculate safe limits for goal
        g_x_min, g_x_max = wb['x_min'] + margin, wb['x_max'] - margin
        g_y_min, g_y_max = wb['y_min'] + margin, wb['y_max'] - margin
        
        # Ensure bounds are valid (avoid crash if world is too small)
        if g_x_max <= g_x_min or g_y_max <= g_y_min:
            raise ValueError("World bounds too small for goal margin!")

        # Fixed spawn modality:
        if self.goal_cfg['spawn_mode'] == 'fixed':
            # Load fixed goal
            fg = self.goal_cfg['fixed_goal']
            self.goal = np.array(fg, dtype=np.float32)
            
            # Validate bounds
            if not (g_x_min <= self.goal[0] <= g_x_max and g_y_min <= self.goal[1] <= g_y_max):
                 # We warn but don't crash, in case you intentionally want a hard goal near a wall
                 print(f"Warning: Fixed goal {self.goal} is outside safe spawn margin.")
        
        # Random spawn modality:
        else:
            self.goal = np.array([
                self.np_random.uniform(g_x_min, g_x_max),
                self.np_random.uniform(g_y_min, g_y_max),
                self.np_random.uniform(-np.pi, np.pi)
            ], dtype=np.float32)



    def render(self):
        """
        Renders the environment using OpenCV.
        Returns a numpy array (image) representing the state.
        """
        # 1. Define Image Parameters
        height, width = 500, 500
        scale = 50  # Pixels per meter (Zoom level)
        center_x, center_y = width // 2, height // 2
        
        # Create a white background
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Helper: Transform World (meters) -> Image (pixels)
        def to_pixel(x, y):
            # Centering the robot in the middle of the screen? 
            # OR Fixed map? Let's do Fixed Map centered at (0,0) for simplicity.
            px = int(center_x + x * scale)
            py = int(center_y - y * scale) # Y-axis inverted in images
            return (px, py)

        # 2. Draw Obstacles (Red Polygons)
        for vertices in self.obstacle_manager.get_all_vertices():
            points = np.array([to_pixel(v[0], v[1]) for v in vertices], dtype=np.int32)
            cv2.fillPoly(canvas, [points], (0, 0, 255))  # BGR: Red
            cv2.polylines(canvas, [points], True, (0, 0, 150), 2)  # Dark red outline

        # 3. Draw Goal (Green Circle)
        gx, gy = to_pixel(self.goal[0], self.goal[1])
        cv2.circle(canvas, (gx, gy), int(0.2 * scale), (0, 255, 0), -1)
        cv2.putText(canvas, "GOAL", (gx + 10, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)

        # 4. Draw Robot (Blue Circle + Heading Line)
        rx, ry, theta = self.state
        rpx, rpy = to_pixel(rx, ry)
        robot_rad_px = int(self.rob_cfg['radius'] * scale)
        
        # Body
        cv2.circle(canvas, (rpx, rpy), robot_rad_px, (255, 0, 0), -1)
        
        # Heading (Line indicating direction)
        end_x = rpx + int(robot_rad_px * np.cos(theta))
        end_y = rpy - int(robot_rad_px * np.sin(theta)) # Minus because Y is flipped
        cv2.line(canvas, (rpx, rpy), (end_x, end_y), (0, 0, 0), 2)

        # 5. Draw Trajectory (if set) ---
        if self.trajectory_buffer is not None and len(self.trajectory_buffer) > 0:
            # Convert world points to pixel points
            path_pixels = []
            for pt in self.trajectory_buffer:
                # pt is assumed to be [x, y, ...]
                path_pixels.append(to_pixel(pt[0], pt[1]))
            
            # Draw as a polyline (Cyan color)
            pts = np.array(path_pixels, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(255, 255, 0), thickness=2)

        # 5. Display (Optional: Only works if you have a GUI/X11)
        # If running in headless Docker, you might want to return the array instead.
        if self.render_mode == "human":
            cv2.imshow("Unicycle Nav", canvas)
            cv2.waitKey(1)  # 1ms delay to process events
            
        return canvas
    

    def set_render_trajectory(self, path):
        """
        Sets a trajectory (list of [x, y] or [x, y, theta]) to be visualized.
        Call this from your main script after calculating the path.
        """
        self.trajectory_buffer = path