import torch
import torch.nn as nn
from torch.distributions import Normal

class NavActor(nn.Module):
    """
    The Actor Network: Maps observations to actions.
    Outputs the mean (mu) of a Gaussian distribution for v and omega.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(NavActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Keeps action means in [-1, 1]
        )
        
        # Log standard deviation: start with high exploration
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
    """
    The Critic Network: Maps observations to a scalar Value (V).
    Estimates the 'goodness' of a state to help the Actor calculate Advantage.
    """
    def __init__(self, obs_dim, hidden_dim=256):
        super(NavCritic, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Outputs a single scalar value
        )

    def forward(self, x):
        return self.net(x)
    

class NavAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, device="cpu"):
        super(NavAgent, self).__init__()
        self.device = device
        
        # 1. The Actor (The Controller)
        self.actor = NavActor(obs_dim, action_dim, hidden_dim).to(device)
        
        # 2. The Critic (The Judge)
        self.critic = NavCritic(obs_dim, hidden_dim).to(device)
        
    def get_action(self, obs, deterministic=False):
        """
        High-level interface for the training loop.
        Input: Raw observation from env (as numpy array)
        Output: action (numpy), log_prob (torch), value (torch)
        """
        # Convert to tensor and send to device
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get distribution parameters and state value
        mu, std = self.actor(obs_t)
        value = self.critic(obs_t)
        
        dist = Normal(mu, std)
        
        if deterministic:
            action = mu
        else:
            action = dist.sample()
            
        # Calculate log_prob (summed across action dimensions)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Prepare for environment (numpy)
        action_np = np.clip(action.cpu().detach().numpy()[0], -1.0, 1.0)
        
        return action_np, log_prob, value.squeeze(0), dist.entropy().sum(dim=-1)

    def evaluate_actions(self, obs_batch, action_batch):
        """
        Used during the PPO update step to recalculate log_probs and values.
        """
        mu, std = self.actor(obs_batch)
        value = self.critic(obs_batch)
        
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(-1), entropy