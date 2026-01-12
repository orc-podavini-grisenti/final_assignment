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
        v, omega = action

        reward = 0.0

        # 1. ENCOURAGE SLOWING DOWN NEAR GOAL
        # If the robot is very close (rho < threshold), we penalize high velocity
        # This forces the robot to 'stabilize' rather than 'shoot through'
        if rho < 0.5:
            reward -= 0.5 * v  # Penalty for moving too fast when close

        # 2. ALIGNMENT REWARD (Dynamic)
        # Instead of just cos(alpha), reward matching the FINAL d_theta
        # This gives a signal even if the robot is sitting on the goal
        reward += self.w_alignment * np.cos(d_theta)

        # 3. THE "GOLDEN" SUCCESS BONUS
        # The paper uses high rewards for success in benchmarks[cite: 161, 162].
        # Reaching the goal with the correct orientation should be 
        # the single largest reward the robot can ever receive.
        if reached_goal:
            reward += 1000.0  # Massive bonus for satisfying both constraints
        elif rho < 0.3:
            # "Participation trophy" for getting the position right, 
            # but much smaller than the full success.
            reward += 10.0 

        # 4. STEP PENALTY (Efficiency)
        # This prevents the robot from 'dancing' around the goal to farm alignment points.
        reward -= 0.1 

        return reward