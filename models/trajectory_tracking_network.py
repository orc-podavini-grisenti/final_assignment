import torch
import torch.nn as nn

'''
PART 1: Feedback Controller for Trajectory Tracking
--------------------------------------------------
Goal: Replace an analytical Lyapunov controller with a learned RL policy.
The network outputs a Gaussian distribution (mean and std) for continuous control.
'''
class TTNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(TTNetwork, self).__init__()
        
        # MLP maps state observations to the mean of the action distribution
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),                  # Tanh is preferred for control tasks (smooth gradients)
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)   # Output layer: one value per action dimension
        )
        
        # Learnable log_std allows the RL agent to adapt its exploration noise over time.
        # Initialized to 0 so that exp(0) = 1.0 (standard deviation).
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        # The 'mean' represents the agent's best guess for the optimal action
        action_mean = self.net(x)
        
        # Transform log_std to a positive standard deviation and match batch size
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
        return action_mean, action_std