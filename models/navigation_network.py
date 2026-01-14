import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import numpy as np


class NavActor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=256): 
        super(NavActor, self).__init__()

        # obs_dim = 3 + # lidar raies
        # action_dim = 2 
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Log standard deviation initialization 
        # Initialize to roughly 0.6 std (exp(-0.5))
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) * -0.5)

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std


    def get_action(self, x, deterministic=False, seed=None):
        """
        Helper to sample an action or return the mean.
        Now supports a 'seed' argument for reproducible sampling.
        """
        mu, std = self.forward(x)
        
        if deterministic:
            return mu
        
        # If a seed is provided, use a local generator for reproducibility
        if seed is not None:
            # Create a generator on the same device as the input
            gen = torch.Generator(device=x.device)
            gen.manual_seed(seed)
            
            # torch.normal allows passing a generator
            return torch.normal(mu, std, generator=gen)
        
        # Default behavior (uses global PyTorch random state)
        dist = Normal(mu, std)
        return dist.sample()




class NavCritic(nn.Module):

    # It should return a value function ( a scalr )
    def __init__(self, obs_dim, hidden_dim=256): # Matched to 64 per paper
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
    