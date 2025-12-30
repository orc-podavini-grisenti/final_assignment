import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math

from obstacle_handler import ObstacleManager

class UnicycleEnv(gym.Env):
    """
    A custom Gymnasium environment for a unicycle robot.
    
    State: [x, y, theta] (global frame)
    Action: [v, omega] (linear and angular velocity)
    Observation: Ego-centric vector 
                 [rho_goal, alpha_goal, theta_goal_rel, d_obs_1, phi_obs_1, ...]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config=None):
        super(UnicycleEnv, self).__init__()

        # --- Configuration ---
        if config is None:
            config = {}

        # --- Configuration ---
        self.dt = 0.05  # Time step (s)
        self.max_steps = 500
        
        # Robot Constraints
        self.v_min, self.v_max = 0.0, 1.0   # m/s
        self.w_min, self.w_max = -1.5, 1.5  # rad/s
        self.robot_radius = 0.3             # meters
        
        # Environment Config

        # World bounds for obstacle generation
        self.world_bounds = config.get('world_bounds', {
            'x_min': 0.5, 'x_max': 3.5,
            'y_min': -2.0, 'y_max': 2.0
        })

        self.n_obstacles = 3    # Number of obstacle to spawn
        self.lidar_range = 3.0  # Max range to detect obstacles
        self.goal_threshold = 0.2
        
        # --- Action Space ---
        # FIX: Define the arrays explicitly as float32 to match Gymnasium's expectation
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # --- Observation Space ---
        obs_dim = 3 + (2 * self.n_obstacles)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

        # Internal State (Initialize with float32 for consistency)
        self.state = np.zeros(3, dtype=np.float32) 
        self.goal = np.zeros(3, dtype=np.float32) 
        self.obstacles = []
        self.current_step = 0

        # --- Modules ---
        self.obstacle_manager = ObstacleManager(lidar_range=self.lidar_range)
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Initialize Robot State (x, y, theta)
        self.state = np.array([0.0, 0.0, 0.0])
        
        # 2. Random Goal Generation (e.g., within a 5x5 box)
        self.goal = np.array([
            self.np_random.uniform(2.0, 4.0),
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(-np.pi, np.pi)
        ])
        
        # 3. Random Obstacle Generation
        self.obstacle_manager.generate_random_obstacles(
            num_obstacles=self.n_obstacles,
            bounds=self.world_bounds,
            robot_pos=self.state,
            goal_pos=self.goal,
            min_clearance=0.5,
            np_random=self.np_random
        )
        
        self.current_step = 0
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Unpack and Scale Actions
        # Map [-1, 1] to physical limits
        v_raw, w_raw = action
        v = np.interp(v_raw, [-1, 1], [self.v_min, self.v_max])
        w = np.interp(w_raw, [-1, 1], [self.w_min, self.w_max])
        
        # 2. Apply Kinematics (Unicycle Model)
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        # theta_dot = w
        x, y, theta = self.state
        
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + w * self.dt
        
        # Normalize theta to [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        
        # Update the state to the new one
        self.state = np.array([x_new, y_new, theta_new])
        self.current_step += 1
        
        # 3. Calculate Rewards and Checks
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
        
        # Check Collision
        collision = self.obstacle_manager.check_collision(self.state, self.robot_radius)
        
        # Check Goal Reached
        reached_goal = distance_to_goal < self.goal_threshold
        
        # Reward Function (Basic version - can be moved to rewards.py)
        reward = 0.0
        
        # Progress reward (change in distance)
        prev_dist = np.linalg.norm([x - self.goal[0], y - self.goal[1]])
        reward += (prev_dist - distance_to_goal) * 10.0
        
        # Time penalty
        reward -= 0.05
        
        if collision:
            reward -= 20.0
            terminated = True
        elif reached_goal:
            reward += 20.0
            terminated = True
        else:
            terminated = False
            
        # Truncation (Time limit)
        truncated = self.current_step >= self.max_steps
        
        info = {
            "is_success": reached_goal,
            "collision": collision
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """
        Constructs the observation vector.
        Vector: [rho, alpha, d_theta,  d_obs1, alpha_obs1, ...]
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
        obs_data = self.obstacle_manager.get_closest_obstacles(self.state, self.n_obstacles)
        
        # Flatten obstacle data
        flat_obs = []
        for dist, angle in obs_data:
            flat_obs.extend([dist, angle])
                
        return np.array(obs_vec + flat_obs, dtype=np.float32)

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
        robot_rad_px = int(self.robot_radius * scale)
        
        # Body
        cv2.circle(canvas, (rpx, rpy), robot_rad_px, (255, 0, 0), -1)
        
        # Heading (Line indicating direction)
        end_x = rpx + int(robot_rad_px * np.cos(theta))
        end_y = rpy - int(robot_rad_px * np.sin(theta)) # Minus because Y is flipped
        cv2.line(canvas, (rpx, rpy), (end_x, end_y), (0, 0, 0), 2)

        # 5. Display (Optional: Only works if you have a GUI/X11)
        # If running in headless Docker, you might want to return the array instead.
        if self.render_mode == "human":
            cv2.imshow("Unicycle Nav", canvas)
            cv2.waitKey(1)  # 1ms delay to process events
            
        return canvas