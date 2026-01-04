import numpy as np

def compute_metrics(run, v_ref=0.4):
    """
    Analyzes simulation data to evaluate controller performance.
    
    Parameters:
    - run: Dictionary containing time-series data (cte, angular_vel, linear_vel)
    - v_ref: The target linear velocity the vehicle was supposed to maintain.
    """
    
    # Convert lists to NumPy arrays for vectorized mathematical operations
    cte = np.array(run["cte"])            # Cross-Track Error (distance from path)
    omega = np.array(run["angular_vel"])  # Rotational speed (yaw rate)
    v = np.array(run["linear_vel"])       # Forward speed

    return {
        # --- PATH ACCURACY METRICS ---
        # RMS_CTE: Measures the average magnitude of error. 
        # Squaring the error penalizes large deviations more heavily than small ones.
        # High RMS usually indicates oscillations or poor tuning.
        "RMS_CTE": np.sqrt(np.mean(cte ** 2)),
        
        # Mean_CTE: Indicates systematic bias. 
        # If this is non-zero, the vehicle might be consistently hugging one side 
        # of the track (e.g., failing to compensate for a steady-state curve).
        "Mean_CTE": np.mean(cte),
        
        # Max_CTE: Represents the "worst-case scenario." 
        # Crucial for safety; tells you how close the vehicle came to going off-track.
        "Max_CTE": np.max(cte),


        # --- EFFICIENCY & STABILITY METRICS ---
        # Completion_Time: Total duration of the test.
        # Lower is usually better, provided the vehicle remains stable.
        "Completion_Time": run["sim_time"],
        
        # Control_Jitter: Measures the 'smoothness' of steering.
        # High standard deviation in angular velocity suggests 'chatter' or 
        # high-frequency corrections that can damage hardware or cause instability.
        "Control_Jitter": np.std(omega),
        
        # Control_Energy: Proxies how much 'work' the steering actuator did.
        # High values indicate the controller is working very hard (aggressive steering),
        # which can lead to faster battery drain or mechanical wear.
        "Control_Energy": np.mean(np.abs(omega)),


        # --- SPEED TRACKING ---
        # Speed_Error: Measures how well the vehicle adhered to the target speed.
        "Mean_Spead": np.mean(np.abs(v)),
    }