import numpy as np

    
class NavigationReward:
    def __init__(self, max_v, dt, weights=None):
        
        # Max Step Distance, parameter to normalize the closer reward
        self.max_step_dist = max_v * dt
        self.prev_dist = None

        # Parameters for the Active Obstacle Zone
        self.min_safe_dist = 0.2    # Deep danger threshold
        self.warning_dist = 0.5     # Border where penalty/reward begins
        self.prev_min_dist = None
        
        # Transition treshold between zone 1 and zone 2
        self.distance_transition = 1.5    

        # Dense Reward Weights
        self.w_allignement = 2.0
        self.w_closer =  1.0 
        self.w_pointing = 1.0
        self.w_obstacle = 1.0

        # Terminal Reward Weights
        self.w_success = 200
        self.w_collision = -200
        self.w_truncated = -100
        
        # Storage for debugging/plotting components
        self.history = {
          'total': [],
          'alignment_reward': [],
          'closer_reward': [],
          'pointing_reward': [],
          'obstacle_reward': [],
          'nav_component': [],
          'dock_component': [],
        }
        

    def compute_reward(self, obs, action, info, truncated):
        
        # Observation Description:
        # - rho = linear distance to the goal
        # - alpha = angular diallignment betwen robot orientation ang goal position ( in radiant )
        # - d_theta = difference between the robot and goal orientation ( in radiant )
        # - lidar_scan = array of 20 rays scan ( no obstacle = lidar_scan = 3.0 )
        rho, alpha, d_theta = obs[0:3]
        lidar_scan = obs[3:]
        

        # NB: All the reward must be normalized between -1 and 1 
        

        # --- CONTINUOUS TRANSITION WEIGHT ---
        # Since the task is very complex we decide it to devide in two zone:
        #   Zone 1: Far from the goal: the objective is to point the goal and reduce the ditance
        #   Zone 2: Near the goal: the objective is to get the coorect goal orientation
        
        # The switching point is self.distance_transition: 
        #   if rho > self.distance_transition   ->  rho / self.distance_transition > 0  -> Zone 1
        #   if rho < self.distance_transition   ->  rho / self.distance_transition < 0  -> Zone 2 
        zone = rho / self.distance_transition 
        
        # Calcoliamo il fattore di transizione puro (da 0 a 1)
        # 0 = Lontano, 1 = Vicinissimo
        transition_factor = np.clip(1.0 - zone, 0, 1)

        # Docking: Parte da una base di 0.2 e aggiunge fino a 0.6 in base alla vicinanza.
        # - Lontano (factor 0): 0.2 + 0.0 = 0.2 (Minimo garantito)
        # - Vicino  (factor 1): 0.2 + 0.6 = 0.8 (Massimo)
        docking_weight = 0.2 + (transition_factor * 0.6)
        
        # Nav: È il complementare a 1.0
        # - Lontano: 1.0 - 0.2 = 0.8 (Massimo)
        # - Vicino:  1.0 - 0.8 = 0.2 (Minimo garantito)
        nav_weight = 1.0 - docking_weight

        # print('DEBUG: transition_factor: ', transition_factor, " docking_weight: ", docking_weight)



        # --- 1. ALIGNMENT REWARD ---
        # Encourages smooth trajectories by aligning the pointing error (alpha) with 
        # the final orientation error (d_theta); this rewards approaching the goal 
        # from behind along the ideal curvature for seamless docking.
        alignment_reward = np.cos(alpha - d_theta)
        

        # --- 2. CLOSER REWARD ---
        # We want to reward if the distance to the goal is reduced respect ot the privius step:
        # dist_change = delta of the rho:
        #   - positive if prev_dist > curr_dist  -> the distance is reduced, robot nearest tot he goal
        #   - negative if prev_dist < curr_dist  -> the distance is increase, robot is going away
        dist_change = 0.
        if self.prev_dist is not None:
            dist_change = self.prev_dist - rho
        
        # Normalizzazione tra -1 e 1
        closer_reward = np.clip(dist_change / self.max_step_dist, -1, 1)
        self.prev_dist = rho
        

        # --- 3. POINTING REWARD ---
        # We want to reward if the robot is pointing the direction of the goal ( not orientation )
        # aplha = 0             ->  orientation point the goal  -> cos(0)  = 1
        # alpha = pi            ->  orientation opposit         -> cos(pi) = -1
        pointing_reward = np.cos(alpha)


        # --- 4. GET AWAY OBSTACLES REWARD ---
        obstacle_reward = 0.0
        curr_min_dist = np.min(lidar_scan)

        if curr_min_dist < self.warning_dist and self.prev_min_dist is not None:
            dist_from_obs_change = curr_min_dist - self.prev_min_dist  
            obstacle_reward = np.clip(dist_from_obs_change / self.max_step_dist, -1, 1)
        
        self.prev_min_dist = curr_min_dist



        # Applay Reward Weights:
        alignment_reward = alignment_reward * self.w_allignement 
        closer_reward = closer_reward * self.w_closer
        pointing_reward = pointing_reward * self.w_pointing
        obstacle_reward = obstacle_reward * self.w_obstacle


        # --- APPLY THE ZONE LOGIC ---
        # In Zone 1 (Far): Focus su avvicinamento (closer) e puntamento (pointing)
        # In Zone 2 (Near): Focus su allineamento finale (alignment)
        nav_component = (pointing_reward + closer_reward) * nav_weight
        dock_component = (alignment_reward * closer_reward) * docking_weight

        # Sommiamo i componenti principali
        # Nota: obstacle_reward rimane fuori dai pesi di zona perché la sicurezza 
        # è prioritaria sia lontano che vicino al goal.
        dense_reward = nav_component + dock_component + obstacle_reward - 0.2

       
        # --- TERMINAL EVENTS ---
        terminal_reward = 0.0
        if info['is_success']:
            terminal_reward = self.w_success 
        elif info['collision']:
            terminal_reward = -200.0
        elif truncated:
            terminal_reward = -100.0
        
        # Constant step penalty to encourage speed
        reward = dense_reward + terminal_reward


        # --- HISTORY LOGGING ( for plotting utility stuff) ---
        self.history['total'].append(reward)
        self.history['alignment_reward'].append(alignment_reward)
        self.history['closer_reward'].append(closer_reward)
        self.history['pointing_reward'].append(pointing_reward)
        self.history['obstacle_reward'].append(obstacle_reward)
        self.history['nav_component'].append(nav_component)
        self.history['dock_component'].append(dock_component)
        

        return reward


    # reset method for the start of a new episode
    def reset(self):
        # Previus parameter rest
        self.prev_dist = None
        self.prev_min_dist = None

        # Reset hystory 
        for key in self.history: 
            self.history[key] = []