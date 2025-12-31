import numpy as np

''' 
    PART 1: Feedback Controller for Trajectory Tracking
    --------------------------------------------------
    Goal: Replace the analytical Lyapunov controller with a learned RL policy.
    This reward function incentivizes precise tracking and stable motion.
    
    COMPONENTS:
    1. Positional Error (R_pos): 
       Minimizes Euclidean distance (rho) between the robot and target waypoint.
       Formula: -w_rho * rho
       
    2. Heading Error (R_head): 
       Ensures the robot's orientation (theta) aligns with the path.
       Formula: -w_theta * |delta_theta|
       
    3. Smoothness (R_smooth): 
       Penalizes high angular velocity (omega) to prevent jittery oscillations.
       Formula: -w_omega * |omega|
       
    Note: Obstacle avoidance (Part 2) is excluded from this formulation.
'''
class TrajectoryReward:
    def __init__(self, config=None):
        # Weights for Part 1: Trajectory Tracking
        self.w_rho = 2.0      # Weight for distance error
        self.w_theta = 1.0    # Weight for heading error
        self.w_omega = 0.1    # Weight for smoothness (penalize steering effort)
        self.w_vel = 0.5      # Reward for maintaining speed (optional)

    def compute_reward(self, tracking_obs, action):
        """
        Calculates reward based on the Tracking Observation (not the global goal).
        
        Args:
            tracking_obs: [rho, alpha, d_theta] relative to the current waypoint.
            action: [v, omega] physical action taken.
        """
        rho = tracking_obs[0]      # Distance to waypoint
        alpha = tracking_obs[1]    # Bearing to waypoint
        d_theta = tracking_obs[2]  # Orientation error relative to waypoint path
        v, omega = action

        reward = 0.0

        # 1. Minimize Position Error (The main objective)
        # We use an exponential kernel so the reward is higher (near 0) when close, 
        # and decays to -1 when far away. This is more stable than linear -rho.
        reward -= self.w_rho * rho

        # 2. Minimize Heading Error
        # Crucial for "Trajectory" tracking, otherwise it just touches points sideways
        reward -= self.w_theta * abs(d_theta)

        # 3. Smoothness Penalty (Part 1 requirement)
        # Penalize large steering commands to prevent oscillation
        reward -= self.w_omega * abs(omega)

        # 4. Progress Bonus (Optional)
        # Reward moving forward if we are roughly aligned
        # This prevents the robot from sitting still to minimize omega penalty
        # if abs(alpha) < 0.5:  # If looking at target
        #     reward += self.w_vel * v

        return reward