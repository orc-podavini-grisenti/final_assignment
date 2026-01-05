import numpy as np
import math

class DubinsPlanner:
    """
    Computes the shortest path between two poses (x, y, theta) 
    using Dubins curves (Standard formulation).
    """
    def __init__(self, curvature=1.0, step_size=0.05):
        self.k_max = curvature
        self.step_size = step_size
        
        # Primitives definitions
        self.primitives = [self._LSL, self._RSR, self._LSR, self._RSL, self._RLR, self._LRL]
        self.ksigns = [
            [ 1,  0,  1],  # LSL
            [-1,  0, -1],  # RSR
            [ 1,  0, -1],  # LSR
            [-1,  0,  1],  # RSL
            [-1,  1, -1],  # RLR
            [ 1, -1,  1],  # LRL
        ]

    def get_path(self, start, goal):
        """
        Generates a discretized path from start to goal.
        
        Args:
            start: [x, y, theta]
            goal:  [x, y, theta]
            
        Returns:
            path: np.array of shape (N, 3) containing [x, y, theta] waypoints
        """
        x0, y0, th0 = start
        xf, yf, thf = goal
        
        # 1. Compute optimal Dubins parameters (lengths and curvature signs)
        lengths, k_signs = self._solve_dubins(x0, y0, th0, xf, yf, thf)
        
        if lengths is None:
            return None # No solution found
            
        # 2. Discretize the path
        return self._generate_points(start, lengths, k_signs)

    def _solve_dubins(self, x0, y0, th0, xf, yf, thf):
        # Scale to standard problem
        dx, dy = xf - x0, yf - y0
        d = math.hypot(dx, dy)
        
        if d < 1e-6: # Start and Goal are too close
            return [0, 0, 0], [0, 0, 0]

        theta = math.atan2(dy, dx)
        lambda_ = d / 2.0
        sc_k_max = self.k_max * lambda_
        
        # Normalized angles
        sc_th0 = self._mod2pi(th0 - theta)
        sc_thf = self._mod2pi(thf - theta)
        
        best_cost = float('inf')
        best_lengths = None
        best_idx = -1
        
        # Try all 6 primitives
        for i, primitive in enumerate(self.primitives):
            success, s1, s2, s3 = primitive(sc_th0, sc_thf, sc_k_max)
            if success:
                cost = s1 + s2 + s3
                if cost < best_cost:
                    best_cost = cost
                    best_lengths = [s1 * lambda_, s2 * lambda_, s3 * lambda_]
                    best_idx = i
                    
        if best_idx == -1:
            return None, None
            
        # Get physical curvatures
        k_signs = [k * self.k_max for k in self.ksigns[best_idx]]
        
        return best_lengths, k_signs

    def _generate_points(self, start, lengths, curvatures):
        """
        Generates path points using exact trigonometric interpolation 
        instead of Euler integration to avoid drift.
        """
        x_current, y_current, theta_current = start
        path = [[x_current, y_current, theta_current, 0.0]]
        
        # Loop through the 3 segments
        for length, k in zip(lengths, curvatures):
            if length < 1e-6: continue

            n_steps = int(math.ceil(length / self.step_size))
            if n_steps == 0: continue
            
            # 1. Store the start of this segment
            x0, y0, th0 = x_current, y_current, theta_current
            
            ds = length / n_steps
            
            for i in range(1, n_steps + 1):
                dist = i * ds
                
                # --- Exact Integration ---
                if abs(k) < 1e-6:
                    # Straight Motion (Linear)
                    x = x0 + dist * math.cos(th0)
                    y = y0 + dist * math.sin(th0)
                    theta = th0
                else:
                    # Circular Motion (Arc)
                    # Radius r = 1/k. Center of rotation formulas.
                    r = 1.0 / k
                    
                    # Change in angle
                    d_theta = dist * k
                    theta = self._mod2pi(th0 + d_theta)
                    
                    # Chord length formula for displacement
                    # dx = r * sin(d_theta)
                    # dy = r * (1 - cos(d_theta))
                    # rotated by th0
                    
                    chord = 2.0 * r * math.sin(d_theta / 2.0)
                    # The angle of the chord relative to start heading is d_theta/2
                    
                    # Global position
                    x = x0 + chord * math.cos(th0 + d_theta / 2.0)
                    y = y0 + chord * math.sin(th0 + d_theta / 2.0)

                # Append
                path.append([x, y, theta, k])
            
            # Update current pose for the next segment exactly
            # (Use the exact logic again for the full length to prevent float noise)
            if abs(k) < 1e-6:
                x_current = x0 + length * math.cos(th0)
                y_current = y0 + length * math.sin(th0)
                theta_current = th0
            else:
                d_theta_total = length * k
                theta_current = self._mod2pi(th0 + d_theta_total)
                r = 1.0 / k
                chord = 2.0 * r * math.sin(d_theta_total / 2.0)
                x_current = x0 + chord * math.cos(th0 + d_theta_total / 2.0)
                y_current = y0 + chord * math.sin(th0 + d_theta_total / 2.0)

        return np.array(path)

    # --- Helpers ---
    def _mod2pi(self, theta):
        return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))

    def _sinc(self, t):
        if abs(t) < 0.002:
            return 1 - (t**2)/6 * (1 - (t**2)/20)
        return math.sin(t)/t

    # --- Primitives (LSL, RSR, etc.) ---
    # These operate on the scaled standard problem
    
    def _LSL(self, alpha, beta, K):
        invK = 1.0 / K
        C = math.cos(beta) - math.cos(alpha)
        S = 2 * K + math.sin(alpha) - math.sin(beta)
        temp1 = math.atan2(C, S)
        s1 = invK * self._mod2pi(temp1 - alpha)
        temp2 = 2 + 4 * K**2 - 2 * math.cos(alpha - beta) + 4 * K * (math.sin(alpha) - math.sin(beta))
        if temp2 < 0: return False, 0, 0, 0
        s2 = invK * math.sqrt(temp2)
        s3 = invK * self._mod2pi(beta - temp1)
        return True, s1, s2, s3

    def _RSR(self, alpha, beta, K):
        invK = 1.0 / K
        C = math.cos(alpha) - math.cos(beta)
        S = 2 * K - math.sin(alpha) + math.sin(beta)
        temp1 = math.atan2(C, S)
        s1 = invK * self._mod2pi(alpha - temp1)
        temp2 = 2 + 4 * K**2 - 2 * math.cos(alpha - beta) - 4 * K * (math.sin(alpha) - math.sin(beta))
        if temp2 < 0: return False, 0, 0, 0
        s2 = invK * math.sqrt(temp2)
        s3 = invK * self._mod2pi(temp1 - beta)
        return True, s1, s2, s3

    def _LSR(self, alpha, beta, K):
        invK = 1.0 / K
        C = math.cos(alpha) + math.cos(beta)
        S = 2 * K + math.sin(alpha) + math.sin(beta)
        temp1 = math.atan2(-C, S)
        temp3 = 4 * K**2 - 2 + 2 * math.cos(alpha - beta) + 4 * K * (math.sin(alpha) + math.sin(beta))
        if temp3 < 0: return False, 0, 0, 0
        s2 = invK * math.sqrt(temp3)
        temp2 = -math.atan2(-2, s2 * K)
        s1 = invK * self._mod2pi(temp1 + temp2 - alpha)
        s3 = invK * self._mod2pi(temp1 + temp2 - beta)
        return True, s1, s2, s3

    def _RSL(self, alpha, beta, K):
        invK = 1.0 / K
        C = math.cos(alpha) + math.cos(beta)
        S = 2 * K - math.sin(alpha) - math.sin(beta)
        temp1 = math.atan2(C, S)
        temp3 = 4 * K**2 - 2 + 2 * math.cos(alpha - beta) - 4 * K * (math.sin(alpha) + math.sin(beta))
        if temp3 < 0: return False, 0, 0, 0
        s2 = invK * math.sqrt(temp3)
        temp2 = math.atan2(2, s2 * K)
        s1 = invK * self._mod2pi(alpha - temp1 + temp2)
        s3 = invK * self._mod2pi(beta - temp1 + temp2)
        return True, s1, s2, s3

    def _RLR(self, alpha, beta, K):
        invK = 1.0 / K
        C = math.cos(alpha) - math.cos(beta)
        S = 2 * K - math.sin(alpha) + math.sin(beta)
        temp1 = math.atan2(C, S)
        temp2 = 0.125 * (6 - 4 * K**2 + 2 * math.cos(alpha - beta) + 4 * K * (math.sin(alpha) - math.sin(beta)))
        if abs(temp2) > 1: return False, 0, 0, 0
        s2 = invK * self._mod2pi(2 * math.pi - math.acos(temp2))
        s1 = invK * self._mod2pi(alpha - temp1 + 0.5 * s2 * K)
        s3 = invK * self._mod2pi(alpha - beta + K * (s2 - s1))
        return True, s1, s2, s3

    def _LRL(self, alpha, beta, K):
        invK = 1.0 / K
        C = math.cos(beta) - math.cos(alpha)
        S = 2 * K + math.sin(alpha) - math.sin(beta)
        temp1 = math.atan2(C, S)
        temp2 = 0.125 * (6 - 4 * K**2 + 2 * math.cos(alpha - beta) - 4 * K * (math.sin(alpha) - math.sin(beta)))
        if abs(temp2) > 1: return False, 0, 0, 0
        s2 = invK * self._mod2pi(2 * math.pi - math.acos(temp2))
        s1 = invK * self._mod2pi(temp1 - alpha + 0.5 * s2 * K)
        s3 = invK * self._mod2pi(beta - alpha + K * (s2 - s1))
        return True, s1, s2, s3