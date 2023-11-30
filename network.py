import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal

class Network(nn.Module):
    def __init__(self, env):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.OBS_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, env.ACTION_DIM)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)
    
class RadialNetwork2d(nn.Module):
    
    def __init__(self, env):
        super(RadialNetwork2d, self).__init__()
        
        self.env = env
        
        x_min, y_min, x_max, y_max = self.env.BOUNDS
        x_range = torch.arange(x_min, x_max, self.env.radial_basis_dist)
        y_range = torch.arange(y_min, y_max, self.env.radial_basis_dist)

        self.in_dim = x_range.shape[0] * y_range.shape[0]
        
        x, y = torch.meshgrid(x_range, y_range)

        self.grid = torch.stack((x,y), axis = 2)

        self.sigma_inv = (torch.eye(2)*self.env.radial_basis_var).inverse()

        # repeat the inverse matrixx*y times so the final shape is x*y*2*2
        self.sigma_inv = self.sigma_inv.unsqueeze(0).unsqueeze(0).repeat(self.grid.shape[0], self.grid.shape[1], 1, 1)

        self.layers = nn.Sequential(nn.Linear(self.in_dim, env.ACTION_DIM))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        basis = self.radial_basis(x)
        
        return self.layers(basis)

    def radial_basis(self, position):
        # if position.shape[0] == 1, add a batch dimension
        if position.shape == (2,):
            position = position.unsqueeze(0)
        
        # broadcast the position to the shape of the radial basis. Current shape is b*2
        # so the final shape is b*x*y*2 where b is the batch size
        positions = position.unsqueeze(1).unsqueeze(1).repeat(1, self.grid.shape[0], self.grid.shape[1], 1)

        # matrix multiply the last two dimentions of the sigma_inv and position tensors in a batch fashion
        # so the final shape is b*x*y*2
        result = torch.matmul(self.sigma_inv, (positions - self.grid).unsqueeze(4)).squeeze(4)
        # result = torch.matmul(self.sigma_inv, (positions - self.grid).unsqueeze(3)).squeeze()

        # matrix multiply the result with the position tensor in a batch fashion
        # so the final shape is b*x*y
        result = torch.matmul(result.unsqueeze(3), (positions - self.grid).unsqueeze(4)).squeeze((3,4))

        # Flatten the last two dimentions
        basis = torch.exp(-0.5*result).flatten(start_dim=1)

        basis.requires_grad = False

        return basis

class RadialNonLinearNetwork2d(RadialNetwork2d):
    
    def __init__(self, env):
        super(RadialNonLinearNetwork2d, self).__init__(env)

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, env.ACTION_DIM)
            )
      