import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from models.trajectory_tracking_network import TTNetwork

'''
PART 1: Feedback Controller for Trajectory Tracking
--------------------------------------------------
Goal: Replace an analytical Lyapunov controller with a learned RL policy.
The network outputs a Gaussian distribution (mean and std) for continuous control.
'''
class Reinforcment:
    def __init__(self, device, obs_dim, action_dim, lr=1e-3, gamma=0.99):
        self.device = device
        self.policy = TTNetwork(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
        # Episodic memory: cleared after every policy update
        self.log_probs = []
        self.rewards = []

    def get_action(self, obs):
        """Samples an action from the Gaussian distribution and tracks its log-probability."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get distribution parameters from the network
        mean, std = self.policy(obs_t)
        dist = Normal(mean, std)
        
        # Sample using the reparameterization trick (implicit in Normal.sample)
        action = dist.sample()
        
        # Calculate log-probability of the chosen action. 
        # We sum across action dims assuming independence (Multivariate Gaussian with diagonal covariance).
        log_prob = dist.log_prob(action).sum(dim=1)
        self.log_probs.append(log_prob)
        
        # Convert to NumPy and clip to the valid actuator range [-1, 1]
        return np.clip(action.cpu().detach().numpy()[0], -1.0, 1.0)

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        """Performs the Monte Carlo Policy Gradient (REINFORCE) update."""
        R = 0
        returns = []
        
        # 1. Compute discounted returns G_t = r_t + γ*r_{t+1} + ... 
        # We iterate backwards for efficiency.
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.device)
        
        # 2. Whiten returns (Standardization) 
        # High variance in returns is a common REINFORCE issue; this stabilizes training.
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        # 3. Compute the Policy Gradient Loss
        # Formula: -E[log_prob * G_t]. Negative sign turns maximization into minimization.
        policy_loss = []
        for log_prob, G_t in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G_t)
            
        # Total loss is the sum of weighted log-probabilities over the episode
        loss = torch.stack(policy_loss).sum()
        
        # 4. Gradient Descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 5. Clear trajectory memory for the next episode
        self.log_probs = []
        self.rewards = []