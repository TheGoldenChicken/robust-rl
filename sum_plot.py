import matplotlib.pyplot as plt
import numpy as np
import torch
# states = torch.FloatTensor(np.linspace(0, 1000, 5000)).reshape(-1, 1).to('cuda')
# print(states)
# Load the .npy file
for delta in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]:

    data = np.load(f'sumo/test_results/seed_4242_robust_factor_-1/{delta}-q_vals.npy')

    # Extract the columns
    # column1 = data[:, 0]
    # column2 = data[:, 1]
    # column3 = data[:, 2]
    #
    # #
    # # Plot the columns
    # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(column1)
    # plt.title(f'Action 0, {delta}')
    # plt.subplot(3, 1, 2)
    # plt.plot(column2)
    # plt.title('Action 1')
    # plt.subplot(3, 1, 3)
    # plt.plot(column3)
    # plt.title('Action 2')
    # plt.tight_layout()
    # plt.show()

betas = np.load(f'sumo/test_results/seed_4242_robust_factor_-1/{0.1}-betas.npy')
diffs = [abs(betas[i] - betas[i+1]) for i, r in enumerate(betas[:-1])]
print(betas)
plt.plot(np.arange(len(betas[:-1:20])),diffs[::20])
plt.show()
