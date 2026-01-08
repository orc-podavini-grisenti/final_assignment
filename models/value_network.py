import torch
import torch.nn as nn

'''
PART 1.2: Feedback Controller for Trajectory Tracking + Baseline
--------------------------------------------------
Goal: Add a estimated value function as baseline to reduce variance
'''
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)