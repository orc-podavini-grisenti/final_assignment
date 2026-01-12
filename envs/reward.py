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
        self.min_safe_dist = 0.2    # Deep danger threshold
        self.warning_dist = 0.4     # Border where penalty/reward begins
        self.d_transition = 0.5     # Border where transition between alpha and theta happens


        # Reward Weights
        self.w_goal_dist = 20.0     # Dense progress signal [cite: 136]
        self.w_alpha = 1.0
        self.w_theta = 2.0
        self.w_consistency = 2.0 
        self.w_obs_approach = 8.0   # Pessimistic approach penalty 
        self.w_obs_escape = 5.0     # Active escape reward
        self.w_success = 250.0     # Terminal success bonus [cite: 161]
        
        self.prev_min_dist = None
        # Storage for debugging/plotting components
        self.history = {
          'total': [],
          'progress': [],
          'obstacle': [],
          'nav_alpha': [],
          'pose_theta': [],
          'pose_consistency': [],
          'terminal': [],
        }
        

    import numpy as np


    def compute_reward(self, obs, action, collision, reached_goal, too_far, prev_dist, curr_dist):
        rho, alpha, d_theta = obs[0:3]
        lidar_scan = obs[3:]
        
        dist_change = prev_dist - curr_dist
        
        # --- 1. CONTINUOUS TRANSITION WEIGHT ---
        # Transitions from 0.0 (Far) to 1.0 (Inside Docking Zone)
        # This smooths the boundary at d_near
        docking_weight = np.clip(1.0 - (rho / self.d_transition), 0, 1)
        nav_weight = 1.0 - docking_weight

        # --- 2. CONTINUOUS ALIGNMENT (REPLACES IF-ELSE) ---
        # alignment_score is 1.0 if perfectly aligned, -1.0 if facing away
        alignment_score = np.cos(alpha - d_theta)
        
        # --- 3. NAVIGATION LOGIC (SMOOTHED) ---
        # We apply the nav_weight so this fades out as we enter the cage
        progress_rew_nav = dist_change * self.w_goal_dist
        nav_rew = -self.w_alpha * np.abs(alpha) * nav_weight

        # --- 4. DOCKING CAGE LOGIC (CONTINUOUS) ---
        # Consistency: Instead of (score**2) with a hard cut at 0, 
        # we use a shifted sigmoid or a rectified cosine to keep it positive but smooth.
        # This rewards being in the 'rear quadrant' smoothly.
        consistency_rew = self.w_consistency * np.maximum(0, alignment_score)**2 * docking_weight

        # Progress inside the cage:
        # We want to reward moving closer ONLY if aligned.
        # Instead of hard multipliers, we use a continuous scaling function.
        # (alignment_score + 1) / 2 maps [-1, 1] to [0, 1]
        alignment_multiplier = np.maximum(0, alignment_score) 
        
        if dist_change > 0:
            # Getting closer: Reward is proportional to how well we are aligned
            progress_rew_dock = dist_change * self.w_goal_dist * alignment_multiplier
        else:
            # Moving away: Only penalize if we are NOT in the "back-up" alignment zone
            # This allows the robot to back up to re-align (neutral) but penalizes "escaping"
            penalty_factor = np.clip(1.0 - alignment_score, 0, 2)
            progress_rew_dock = dist_change * self.w_goal_dist * penalty_factor

        # Blend the progress rewards
        progress_rew = (nav_weight * progress_rew_nav) + (docking_weight * progress_rew_dock)

        # --- 5. OBSTACLES (KEEP AS IS) ---
        obs_rew = 0.0
        curr_min_dist = np.min(lidar_scan)
        if curr_min_dist < self.warning_dist:
            intensity = np.clip((self.warning_dist - curr_min_dist) / 
                              (self.warning_dist - self.min_safe_dist), 0, 1)
            obs_rew = -self.w_obs_approach * intensity

        # --- 6. TERMINAL EVENTS ---
        terminal_rew = 0.0
        if reached_goal:
            # Add a bonus for alignment quality at the moment of success
            terminal_rew = self.w_success * (0.5 + 0.5 * alignment_score)
        elif collision:
            terminal_rew = -200.0
        elif too_far:
            terminal_rew = -100.0
        
        # Constant step penalty to encourage speed
        total = progress_rew + consistency_rew + terminal_rew + obs_rew + nav_rew - 0.05

        # --- HISTORY LOGGING (KEEP UNCHANGED) ---
        self.history['total'].append(total)
        self.history['progress'].append(progress_rew)
        self.history['nav_alpha'].append(nav_rew)
        self.history['pose_consistency'].append(consistency_rew)
        self.history['obstacle'].append(obs_rew)
        self.history['terminal'].append(terminal_rew)
        
        return total



    def reset_history(self):
        self.prev_min_dist = None
        for key in self.history: self.history[key] = []