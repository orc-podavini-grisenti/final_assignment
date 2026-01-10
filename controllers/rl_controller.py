import torch
import numpy as np
from models.trajectory_tracking_network import TTNetwork

class RLController:
    def __init__(self, model_path, obs_dim=3, action_dim=2):
        # 1. Initialize the architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TTNetwork(obs_dim, action_dim).to(self.device)

        # 2. Load the .pth file
        # We use weights_only=True for security if using newer torch versions
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # 3. Set to Evaluation Mode
        # This is critical! It ensures the model is not in training mode.
        self.model.eval()
        print(f"RL Controller loaded successfully on {self.device}")

    def get_action(self, obs, v_ref=0.0, omega_ref=0.0):
        """
        Maps [rho, alpha, d_theta] to [v_norm, w_norm] using the RL policy.
        """
        # Convert observation to float32 tensor and add batch dimension
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # The forward pass returns (mean, std)
            action_mean, _ = self.model(obs_tensor)
            
            # Convert back to numpy and remove batch dimension
            # .cpu() ensures it works even if trained on GPU
            action = action_mean.squeeze(0).cpu().numpy()
            
        return action