import numpy as np
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

# class RBFNetwork(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, num_centers=600, env_width=1, device='cuda'):
#         """Initialization."""
#         super(Network, self).__init__()
#         self.centers = torch.FloatTensor(np.linspace(0,env_width, num_centers)).to(device)
#         self.var = env_width/num_centers
#
#         self.layers = nn.Sequential(
#             nn.Linear(num_centers, 64),
#             nn.Linear(64, out_dim),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward method implementation."""
#         input = torch.exp(x - self.centers)/self.var
#         return self.layers(x)