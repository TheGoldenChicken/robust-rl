import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal

class Network(nn.Module):
    def __init__(self, env):
        """Initialization."""
        super(Network, self).__init__()

        # self.layers = nn.Sequential(
        #     nn.Linear(in_dim, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),  # Dropout layer after the first ReLU
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),  # Dropout layer after the second ReLU
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),  # Dropout layer after the third ReLU
        #     nn.Linear(128, out_dim)
        # )
        #
        self.layers = nn.Sequential(
            nn.Linear(env.OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.ACTION_DIM)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)
    
class RadialNetwork2d(nn.Module):
    
    def __init__(self, env):
        super(RadialNetwork2d, self).__init__()
        
        self.env = env
        
        x_min, y_min, x_max, y_max = self.env.BOUNDS
        in_dim = len(torch.arange(x_min, x_max+1, self.env.r_basis_diff)) * \
                    len(torch.arange(y_min, y_max+1, self.env.r_basis_diff))
        
        self.layers = nn.Sequential(nn.Linear(in_dim, env.ACTION_DIM))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        basis = self.radial_basis(x)
        
        return self.layers(basis)

    def radial_basis(self, position):
        x_min, y_min, x_max, y_max = self.env.BOUNDS
        x, y = torch.meshgrid(torch.arange(x_min, x_max+1, self.env.r_basis_diff), torch.arange(y_min, y_max+1, self.env.r_basis_diff))
        points = torch.column_stack((x.ravel(), y.ravel()))

        # Evaluate 2D Gaussian distribution for each point in points
        sigma = torch.eye(2) * self.env.r_basis_var
        basis = torch.zeros(position.shape[0], points.shape[0])
        for i in range(points.shape[0]):
            m = MultivariateNormal(points[i], sigma)
            basis[:, i] = m.log_prob(position).exp()
        
        basis.requires_grad = False
        
        return basis


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