import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from matplotlib import pyplot as plt

x_min = -3
y_min = -2
x_max = 18
y_max = 12
diff = 1

x, y = torch.meshgrid(torch.arange(x_min, x_max+1, diff), torch.arange(y_min, y_max+1, diff))
points = torch.column_stack((x.ravel(), y.ravel()))

sigma = torch.eye(2) * diff
mNormal = []
for i in range(points.shape[0]):
    m = MultivariateNormal(points[i], sigma)
    mNormal.append(m)

layers = nn.Sequential(nn.Linear(x.shape[0]*x.shape[1], 5))

def radial_basis(position):
    basis = torch.zeros(position.shape[0],points.shape[0])
    
    for i in range(points.shape[0]):
        basis[:,i] = mNormal[i].log_prob(position).exp()
    
    basis.requires_grad = False
    
    return basis


# position = torch.tensor([[1,1],[2,2],[3,3]])
position = torch.tensor([[1,1]])

basis = radial_basis(position)

# basis = layers(basis)

print(basis.shape)

# Plot a grid of points at the x and y coordinates and put a text field with the value of the basis function
fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1])
for i in range(points.shape[0]):
    ax.text(points[i,0], points[i,1], str(round(basis[0,i].item(), 4)))
plt.show()
