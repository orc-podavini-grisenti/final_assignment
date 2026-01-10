import numpy as np

class TrajectoryTrackingReward:
    def __init__(self):
        # --- 1. CONTINUOUS TRACKING WEIGHTS (The "Driver" Logic) ---
        # Position (Gaussian bell curve width)
        self.k_rho = 5.0      
        # Importance weights
        self.w_rho = 1.0      
        self.w_theta = 0.8    
        self.w_omega = 0.1   
        self.w_vel = 2.0      

        # --- 2. DISCRETE/EVENT REWARDS (The "Game" Logic) ---
        self.R_GOAL = 200.0
        self.R_COLLISION = -200.0
        self.R_OFF_PATH = -200.0
        self.R_MAX_STEPS = -100.0
        self.R_CHECKPOINT = 10.0  # Reward per path index advanced

    def compute_reward(self, tracking_obs, action, checkpoints_cleared=0, terminal_reason=None):
        """
        Calculates the total reward for the current step.
        
        Args:
            tracking_obs (array): [rho, alpha, d_theta]
            action (array): [v, omega]
            checkpoints_cleared (int): How many path indices were advanced in this step.
            terminal_reason (str): One of ['goal', 'collision', 'off_path', 'timeout', None]
        """
        
        # Unpack observations
        rho = tracking_obs[0]      # Cross-track error
        d_theta = tracking_obs[2]  # Heading error
        v, omega = action

        # --- A. CALCULATE DENSE REWARD ---
        
        # 1. Position Reward (Gaussian: 1.0 at rho=0, decays as rho increases)
        # Replacing the simple linear penalty from the notebook with this smoother gradient
        r_pos = np.exp(-self.k_rho * rho**2)

        # 2. Heading Reward (Cosine: 1.0 at aligned, -1.0 at reverse)
        r_head = np.cos(d_theta)

        # 3. Smoothness Penalty
        r_smooth = -np.abs(omega)

        # 4. Velocity Reward (Encourage forward motion only if aligned)
        r_vel = 0.0
        if abs(d_theta) < 1.0: 
            r_vel = v 
        else:
            # Penalize being lazy (stopped) even if aligned
            if v < 0.1:
                r_vel = -0.5 

        # Weighted Sum of Dense Rewards
        step_reward = (self.w_rho * r_pos) + \
                      (self.w_theta * r_head) + \
                      (self.w_omega * r_smooth) + \
                      (self.w_vel * r_vel)

        # --- B. CALCULATE EVENT REWARD ---
        
        # 5. Checkpoint Progress (Breadcrumbs)
        if checkpoints_cleared > 0:
            step_reward += (self.R_CHECKPOINT * checkpoints_cleared)

        # 6. Terminal Events
        if terminal_reason == "goal":
            step_reward += self.R_GOAL
        elif terminal_reason == "collision":
            step_reward += self.R_COLLISION
        elif terminal_reason == "off_path":
            step_reward += self.R_OFF_PATH
        elif terminal_reason == "timeout":
            step_reward += self.R_MAX_STEPS

        return step_reward