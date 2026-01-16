import numpy as np

class ObservationNormalizer:
    def __init__(self, max_dist=3.0):
        """
        max_dist: Sensitivity limit for distance.
                  Distances beyond this will be clipped to 1.0.
        """
        self.max_dist = max_dist
        self.max_angle = np.pi

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