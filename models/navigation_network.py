import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class NavActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=255): # Reduced to 64 per paper 
        super(NavActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(), # Tanh used in paper 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        # Log standard deviation initialization 
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

    def get_action(self, x, deterministic=False):
        """Helper to sample an action or return the mean."""
        mu, std = self.forward(x)
        if deterministic:
            return mu
        
        dist = Normal(mu, std)
        return dist.sample()


class NavCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64): # Matched to 64 per paper
        super(NavCritic, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(), # Tanh used for robotics tasks
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x):
        return self.net(x)
    

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class NavAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, device="cpu"):
        super(NavAgent, self).__init__()
        self.device = device
        
        # The paper suggests MLP with two hidden layers of 64 units 
        # and tanh nonlinearities for continuous control tasks.
        self.common = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(device)
        
        # 1. The Actor (Policy Head)
        # Maps hidden features to action means 
        self.actor_head = nn.Linear(hidden_dim, action_dim).to(device)
        
        # Variable standard deviations for the Gaussian distribution 
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)).to(device)
        
        # 2. The Critic (Value Head)
        # Maps hidden features to a single scalar state-value V(s) [cite: 121, 129]
        self.critic_head = nn.Linear(hidden_dim, 1).to(device)
        
    def get_action(self, obs, deterministic=False):
        """
        High-level interface for the training loop.
        Follows Algorithm 1: Run policy in environment for T timesteps[cite: 141].
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Forward pass through shared trunk
        features = self.common(obs_t)
        
        # Policy output 
        mu = torch.tanh(self.actor_head(features)) # Keep means in [-1, 1]
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)
        
        # Value output for advantage estimation [cite: 121]
        value = self.critic_head(features)
        
        if deterministic:
            action = mu
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        action_np = action.cpu().detach().numpy()[0]
        
        # We return value and entropy for the surrogate objective in Eq (9) 
        return action_np, log_prob, value.squeeze(0), dist.entropy().sum(dim=-1)

    def evaluate_actions(self, obs_batch, action_batch):
        """
        Used during the multiple epochs of minibatch SGD[cite: 8, 140].
        Computes terms for the combined loss L^{CLIP+VF+S}.
        """
        features = self.common(obs_batch)
        
        mu = torch.tanh(self.actor_head(features))
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)
        
        # Log probability for the ratio r_t(theta) in Eq (6) [cite: 60]
        log_prob = dist.log_prob(action_batch).sum(dim=-1)
        
        # State-value for the squared-error loss L^{VF} in Eq (9) [cite: 129]
        value = self.critic_head(features)
        
        # Entropy bonus S to ensure sufficient exploration [cite: 125, 127]
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(-1), entropy