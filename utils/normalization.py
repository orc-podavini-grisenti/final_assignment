import numpy as np

class ObservationNormalizer:
    def __init__(self, max_dist=3.0, lidar_range=3.0):
        """
        max_dist: Sensitivity limit for distance.
                  Distances beyond this will be clipped to 1.0.
        """
        self.max_dist = max_dist
        self.max_angle = np.pi
        self.lidar_range = lidar_range

    def normalize_tt(self, obs):
        """
        Normaliza the TTNetwork input
        Input: obs [rho, alpha, d_theta]
        Output: normalized_obs roughly in range [-1, 1]
        """
        rho, alpha, d_theta = obs
        
        # 1. Normalize Distance with clipping (0 to 1)
        n_rho = np.clip(rho / self.max_dist, 0.0, 1.0)
        
        # 2. Normalize Angles (-1 to 1)
        n_alpha = alpha / self.max_angle
        n_d_theta = d_theta / self.max_angle
        
        return np.array([n_rho, n_alpha, n_d_theta], dtype=np.float32)
    

    def normalize(self, obs):
        """
        Normalizes the PPO input vector.
        Input: [rho, alpha, d_theta, ray_1, ..., ray_N]
        Output: Normalized vector where distances are [0, 1] and angles are [-1, 1]
        """
        # Split goal observations from lidar observations
        goal_obs = obs[:3]
        lidar_obs = obs[3:]
        
        rho, alpha, d_theta = goal_obs
        
        # 1. Normalize Goal Distance (0 to 1)
        n_rho = np.clip(rho / self.max_dist, 0.0, 1.0)
        
        # 2. Normalize Angles (-1 to 1)
        n_alpha = alpha / self.max_angle
        n_d_theta = d_theta / self.max_angle
        
        # 3. Normalize LiDAR rays (0 to 1)
        # 0.0 = immediate collision, 1.0 = clear path/max range
        n_lidar = np.clip(lidar_obs / self.lidar_range, 0.0, 1.0)
        
        return np.concatenate([
            [n_rho, n_alpha, n_d_theta], 
            n_lidar
        ]).astype(np.float32)