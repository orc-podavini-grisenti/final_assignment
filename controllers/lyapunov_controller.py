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
        rho = obs[0]
        alpha = obs[1]
        d_theta = obs[2]

        exy = rho 
        etheta = -d_theta 

        # 1. Determine Feed-Forward Velocity
        if v_ref == 0.0:
             v_d = 0.5 * np.tanh(rho)
        else:
             v_d = v_ref 

        omega_d = omega_ref 

        # 2. Control Law (Keep your original logic!)
        # We KEEP the braking logic here so the robot "wants" to slow down
        dv = self.K_P * exy * np.cos(alpha)
        
        arg_sin = (alpha + np.pi) + 0.5 * etheta
        cos_half_etheta = np.cos(etheta / 2.0)
        
        # Avoid division by zero
        if abs(cos_half_etheta) < 1e-3: 
            cos_half_etheta = 1e-3 * np.sign(cos_half_etheta)
            
        tracking_term = -v_d * exy * (1.0 / cos_half_etheta) * np.sin(arg_sin)
        stabilization_term = -self.K_THETA * np.sin(etheta)
        
        domega = tracking_term + stabilization_term

        v = v_d + dv
        omega = omega_d + domega

        # 3. CRITICAL FIX: The Minimum Speed Floor
        # If the controller wants to stop or reverse (v <= 0), we force a 
        # tiny positive velocity (e.g., 0.05 or 10% of v_ref).
        # This keeps the robot moving just enough for 'omega' to turn it.
        
        min_velocity = 0.05  # Adjust this (e.g., 0.05 m/s)
        
        if v < min_velocity:
            v = min_velocity

        # Normalize for Gym (assuming limits v=[0,1], w=[-1.5, 1.5])
        v_norm = np.interp(v, [0.0, 1.0], [-1.0, 1.0])
        w_norm = np.interp(omega, [-1.5, 1.5], [-1.0, 1.0])

        return np.array([v_norm, w_norm], dtype=np.float32)