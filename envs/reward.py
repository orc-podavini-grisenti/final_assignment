import numpy as np

class TrajectoryReward:
    def __init__(self, config=None):
        # --- TUNING FOR "ACTIVE" TRACKING ---
        
        # Reduced rho slightly so the robot doesn't panic over small errors
        self.k_rho = 5.0      
        self.w_rho = 1.0      # Reduced from 1.5 (Don't obsess over position)
        
        # Kept alignment importance high so it steers correctly
        self.w_theta = 0.8    
        
        # Smoothness
        self.w_omega = 0.1   

        # --- THE FIX: BOOST PROGRESS REWARD ---
        # Increased from 0.2 to 2.0
        # Now, moving at max speed (v=1.0) gives +2.0 reward.
        # Sitting still gives ~1.8. Moving gives ~3.8.
        # The robot will now WANT to rush forward.
        self.w_vel = 2.0      

    def compute_reward(self, tracking_obs, action):
        rho = tracking_obs[0]
        d_theta = tracking_obs[2]
        v, omega = action

        # 1. Position (Gaussian)
        r_pos = np.exp(-self.k_rho * rho**2)

        # 2. Heading (Cosine)
        r_head = np.cos(d_theta)

        # 3. Smoothness
        r_smooth = -np.abs(omega)

        # 4. Forward Velocity (The Driver)
        r_vel = 0.0
        # Only reward speed if aligned (prevent spinning in circles)
        if abs(d_theta) < 1.0: 
            r_vel = v 
        else:
            # OPTIONAL: Penalize stopping if you are aligned but lazy
            # If angle is good but velocity is low, give a penalty
            if v < 0.1:
                r_vel = -0.5 

        # Total
        reward = (self.w_rho * r_pos) + (self.w_theta * r_head) + \
                 (self.w_omega * r_smooth) + (self.w_vel * r_vel)

        return reward

class NavigationReward:
    def __init__(self, weights=None):
        self.w_goal_dist = 10.0   
        self.w_alignment = 1.0     
        self.w_smoothness = 0.05   
        self.w_obstacle = 5.0      # Increased for a stronger "pessimistic" signal
        self.min_safe_dist = 0.5

        if weights is not None:
            self.w_goal_dist = weights.get('W_GOAL_PROGRESS', self.w_goal_dist)
            self.w_alignment = weights.get('W_ALIGNMENT', self.w_alignment)
            self.w_smoothness = weights.get('W_SMOOTHNESS', self.w_smoothness)
            self.w_obstacle  = weights.get('W_OBSTACLE_DIST', self.w_obstacle)
            self.min_safe_dist = weights.get('W_MIN_SAFE_DIST', self.w_obstacle)
        
    def compute_reward(self, obs, action, collision, reached_goal):
        rho, alpha, d_theta = obs[0:3]
        lidar_scan = obs[3:]
        v, omega = action

        reward = 0.0

        # 1. OBSTACLE AVOIDANCE (The Negative Lower Bound)
        min_lidar = np.min(lidar_scan)
        if min_lidar < self.min_safe_dist:
            # Strong exponential penalty to ensure the agent values safety over all else
            reward -= self.w_obstacle * np.exp(1.0 - min_lidar / self.min_safe_dist)

        # 2. PROGRESS-GATED ALIGNMENT (Prevents Spinning)
        # We only reward facing the goal if the robot is actually moving forward (v > 0)
        if v > 0.1:
            reward += self.w_alignment * np.cos(alpha) * v
        else:
            # Penalize sitting still or moving backward while facing away
            reward -= 0.1

        # 3. SMOOTHNESS 
        # Encourages stable first-order optimization by penalizing jerky rotations
        reward -= self.w_smoothness * np.abs(omega)

        # 4. STEP PENALTY
        # Similar to Atari benchmarks that favor fast learning [cite: 275]
        reward -= 0.05 

        return reward