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
        # Parameters for the Active Obstacle Zone
        self.min_safe_dist = 0.5    # Deep danger threshold
        self.warning_dist = 0.9     # Border where penalty/reward begins
        
        # Reward Weights
        self.w_goal_dist = 10.0     # Dense progress signal [cite: 136]
        self.w_alignment = 1.0      # Orientation matching [cite: 189]
        self.w_obs_approach = 8.0   # Pessimistic approach penalty 
        self.w_obs_escape = 5.0     # Active escape reward
        self.w_success = 1000.0     # Terminal success bonus [cite: 161]
        
        self.prev_min_dist = None
        # Storage for debugging/plotting components
        self.history = {'total': [], 'progress': [], 'obstacle': [], 'alignment': []}

    def compute_reward(self, obs, action, collision, reached_goal, prev_dist, curr_dist):
        rho, alpha, d_theta = obs[0:3]
        lidar_scan = obs[3:] 
        v, omega = action
        
        # 1. Progress Component (Moved from step logic) [cite: 136]
        progress_rew = (prev_dist - curr_dist) * self.w_goal_dist
        
        # 2. Smart Obstacle Zone Logic
        obs_rew = 0.0
        curr_min_dist = np.min(lidar_scan)
        if curr_min_dist < self.warning_dist:
            # Linear intensity ramp (0 at border to 1 at deep danger)
            intensity = np.clip((self.warning_dist - curr_min_dist) / 
                               (self.warning_dist - self.min_safe_dist), 0, 1)
            
            if self.prev_min_dist is not None:
                delta_obs = curr_min_dist - self.prev_min_dist
                # Heavier penalty for approach than reward for escape to prevent farming
                weight = self.w_obs_escape if delta_obs > 0 else self.w_obs_approach
                obs_rew = delta_obs * weight * intensity
        self.prev_min_dist = curr_min_dist

        # 3. Alignment and Final Orientation [cite: 189]
        align_rew = self.w_alignment * np.cos(d_theta)
        if rho < 0.5: align_rew -= 0.5 * v  # Penalize speed near goal

        # 4. Terminal Events [cite: 139, 161]
        terminal_rew = self.w_success if reached_goal else (-500.0 if collision else 0.0)

        # Combine into Total and log components
        total = progress_rew + obs_rew + align_rew + terminal_rew - 0.1 # Step penalty
        
        self.history['total'].append(total)
        self.history['progress'].append(progress_rew)
        self.history['obstacle'].append(obs_rew)
        self.history['alignment'].append(align_rew)
        
        return total

    def reset_history(self):
        self.prev_min_dist = None
        for key in self.history: self.history[key] = []