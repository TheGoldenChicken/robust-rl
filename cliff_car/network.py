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
        x_range = torch.arange(x_min, x_max, self.env.r_basis_diff)
        y_range = torch.arange(y_min, y_max, self.env.r_basis_diff)

        self.in_dim = x_range.shape[0] * y_range.shape[0]

        x, y = torch.meshgrid(x_range, y_range)
        points = torch.column_stack((x.ravel(), y.ravel()))
        self.points = points
        sigma = torch.eye(2) * self.env.r_basis_diff
        self.sigma = sigma
        self.mNormal = []
        for i in range(points.shape[0]):
            m = MultivariateNormal(points[i], sigma)
            self.mNormal.append(m)

        self.layers = nn.Sequential(nn.Linear(self.in_dim, env.ACTION_DIM))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        basis = self.radial_basis(x)

        return self.layers(basis)

    def fast_forward(self, x):
        # We dont use sigma, since its just all ones anyway
        basis = torch.exp(-((self.points - x)**2/2)).float()
        return self.layers(basis.T)

    def radial_basis(self, position):
        basis = torch.zeros(position.shape[0], self.in_dim, device=position.device)

        for i in range(self.in_dim):
            basis[:,i] = self.mNormal[i].log_prob(position).exp()

        basis.requires_grad = False

        return basis

class RadialNonlinearNetwork2d(nn.Module):

    def __init__(self, env):
        super(RadialNonlinearNetwork2d, self).__init__()

        self.env = env

        x_min, y_min, x_max, y_max = self.env.BOUNDS
        x_range = torch.arange(x_min, x_max, self.env.r_basis_diff)
        y_range = torch.arange(y_min, y_max, self.env.r_basis_diff)

        self.in_dim = x_range.shape[0] * y_range.shape[0]

        x, y = torch.meshgrid(x_range, y_range)
        points = torch.column_stack((x.ravel(), y.ravel()))

        sigma = torch.eye(2) * self.env.r_basis_diff
        self.mNormal = []
        for i in range(points.shape[0]):
            m = MultivariateNormal(points[i], sigma)
            self.mNormal.append(m)

        self.layers = nn.Sequential(nn.Linear(self.in_dim, 128),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128),
                                    nn.Linear(128, env.ACTION_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        basis = self.radial_basis(x)

        return self.layers(basis)

    def radial_basis(self, position):
        basis = torch.zeros(position.shape[0], self.in_dim, device=position.device)

        for i in range(self.in_dim):
            basis[:,i] = self.mNormal[i].log_prob(position).exp()

        basis.requires_grad = False

        return basis