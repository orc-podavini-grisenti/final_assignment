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