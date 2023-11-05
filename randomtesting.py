import numpy as np
import scipy.stats
import torch
import time
from scipy.stats import multivariate_normal

test_numbers = 20000

noise_mean = np.array([0,0])
noise_var = np.array([[0.5,0],[0,0.5]])

torch_mean = torch.tensor(noise_mean, dtype=torch.float)
torch_var = torch.tensor(noise_var, dtype=torch.float)

start_time_np = time.time()


for i in range(test_numbers):
    noise = np.random.multivariate_normal(noise_mean, noise_var)

print(noise)
print(time.time() - start_time_np )

tsart_time_torch = time.time()

multivariate = torch.distributions.MultivariateNormal(torch_mean, torch_var)
bo = True
for i in range(test_numbers):
    if bo is True:
        noise = multivariate.sample((1,)).squeeze().numpy()

print(noise)
print(time.time() - tsart_time_torch)

scipy_start_time = time.time()

normal = multivariate_normal(mean=noise_mean, cov=noise_var)

for i in range(test_numbers):
    noise = normal.rvs(1)

print(noise)
print(time.time() - scipy_start_time)
