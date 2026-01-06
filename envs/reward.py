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
        # Navigation weights
        self.w_goal_dist = 10.0    # Progress toward goal
        self.w_alignment = 0.8     # Facing the goal
        self.w_smoothness = 0.1    # Penalize jittery steering
        
        # Obstacle avoidance weights
        # We use an exponential penalty: as distance -> 0, penalty -> infinity
        self.w_obstacle = 2.0      
        self.min_safe_dist = 0.6   # Meters: when to start panicking

        if weights is not None:
            self.w_goal_dist = weights.get('W_GOAL_PROGRESS', self.w_goal_dist)
            self.w_alignment = weights.get('W_ALIGNMENT', self.w_alignment)
            self.w_smoothness = weights.get('W_SMOOTHNESS', self.w_smoothness)
            self.w_obstacle  = weights.get('W_OBSTACLE_DIST', self.w_obstacle)
        
    def compute_reward(self, obs, action, collision, reached_goal):
        """
        obs: [rho, alpha, d_theta, lidar_1 ... lidar_N]
        action: [v, omega]
        """
        # 1. Extract observation components
        rho = obs[0]        # Distance to goal
        alpha = obs[1]      # Angle to goal
        d_theta = obs[2]    # Orientation error
        lidar_scan = obs[3:]
        v, omega = action

        reward = 0.0

        # 2. Obstacle Avoidance (CRITICAL)
        # Find the single closest point detected by LiDAR
        min_lidar = np.min(lidar_scan)
        if min_lidar < self.min_safe_dist:
            # Exponential penalty: becomes very large as the robot gets closer
            # (1 - normalized_dist) creates a value that grows as dist shrinks
            reward -= self.w_obstacle * np.exp(1.0 - min_lidar / self.min_safe_dist)

        # 3. Alignment Reward
        # Reward being pointed toward the goal, especially when moving
        # cos(alpha) is 1.0 when facing goal, -1.0 when facing away
        reward += self.w_alignment * np.cos(alpha) * (v + 0.1)

        # 4. Smoothness
        # Penalize high angular velocity to prevent "spinning"
        reward -= self.w_smoothness * np.abs(omega)

        # 5. Terminal Rewards
        if collision:
            reward -= 100.0
        elif reached_goal:
            reward += 200.0
            
        return reward