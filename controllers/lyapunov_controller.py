import numpy as np
import math

class LyapunovParams:
    def __init__(self, K_P=1.0, K_THETA=1.0, DT=0.05):
        self.K_P = K_P
        self.K_THETA = K_THETA
        self.DT = DT

class LyapunovController:
    
    def __init__(self, params=None):
        if params is None:
            # Default tuning
            params = LyapunovParams(K_P=2.0, K_THETA=4.0)
            
        self.K_P = params.K_P
        self.K_THETA = params.K_THETA
        
        # Logging
        self.log_v = []
        self.log_w = []



    def get_action(self, obs, v_ref=0.0, omega_ref=0.0):
        """
        Args:
            obs: [rho, alpha, d_theta]
            v_ref: Feed-forward linear velocity (default 0 for parking)
            omega_ref: Feed-forward angular velocity (default 0 for parking)
        """
        rho = obs[0]
        alpha = obs[1]
        d_theta = obs[2]

        exy = rho 
        etheta = -d_theta 

        # If we are tracking a moving target (Dubins), use the provided v_ref
        # If we are parking (Static), calculate a slowing-down profile
        if v_ref == 0.0:
             v_d = 0.5 * np.tanh(rho) # Approach logic
        else:
             v_d = v_ref # Tracking logic

        omega_d = omega_ref 

        # --- Control Law ---
        dv = self.K_P * exy * np.cos(alpha)
        
        arg_sin = (alpha + np.pi) + 0.5 * etheta
        cos_half_etheta = np.cos(etheta / 2.0)
        if abs(cos_half_etheta) < 1e-3: cos_half_etheta = 1e-3
            
        tracking_term = -v_d * exy * (1.0 / cos_half_etheta) * np.sin(arg_sin)
        stabilization_term = -self.K_THETA * np.sin(etheta)
        
        domega = tracking_term + stabilization_term

        v = v_d + dv
        omega = omega_d + domega

        # Normalize for Gym (assuming limits v=[0,1], w=[-1.5, 1.5])
        v_norm = np.interp(v, [0.0, 1.0], [-1.0, 1.0])
        w_norm = np.interp(omega, [-1.5, 1.5], [-1.0, 1.0])

        return np.array([v_norm, w_norm], dtype=np.float32)