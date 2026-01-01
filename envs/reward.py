import numpy as np

class TrajectoryReward:
    def __init__(self, config=None):
        # Adjusted weights for Positive Reinforcement
        self.k_rho = 5.0      # Sharpness of the position kernel (Higher = strictly requires precision)
        self.w_rho = 1.5      # Max reward for perfect position
        self.w_theta = 0.8    # Max reward for perfect alignment
        self.w_omega = 0.05   # Weight for smoothness penalty
        self.w_vel = 0.2      # Weight for forward progress

    def compute_reward(self, tracking_obs, action):
        rho = tracking_obs[0]      # Distance
        d_theta = tracking_obs[2]  # Heading Error
        v, omega = action

        # --- 1. Position: Gaussian/Exponential Kernel (The Fix) ---
        # Instead of -rho (linear), we use exp(-rho^2).
        # Result: +1.0 when rho=0, decays to 0.0 when far away.
        # This is bounded [0, 1] and creates a stable "attractor" to the path.
        r_pos = np.exp(-self.k_rho * rho**2)

        # --- 2. Heading: Cosine Alignment ---
        # Instead of -abs(d_theta), we use cos(d_theta).
        # Result: +1.0 when aligned, -1.0 when opposite.
        # This is smoother and differentiable everywhere.
        r_head = np.cos(d_theta)

        # --- 3. Smoothness Penalty ---
        # Keep this negative to discourage shaking
        r_smooth = -np.abs(omega)

        # --- 4. Forward Velocity Bonus ---
        # Only reward speed if we are roughly aligned (prevent running away fast)
        r_vel = 0.0
        if abs(d_theta) < 1.0: 
            r_vel = v  # Reward moving forward

        # TOTAL REWARD
        # Theoretical Max per step: 1.5 (pos) + 0.8 (head) + 0.2 (vel) = ~2.5
        # Theoretical Min per step: 0.0 + (-0.8) + (-omega) = Negative
        reward = (self.w_rho * r_pos) + (self.w_theta * r_head) + (self.w_omega * r_smooth) + (self.w_vel * r_vel)

        return reward