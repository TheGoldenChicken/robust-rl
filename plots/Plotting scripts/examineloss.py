import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
delta_vals = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.001, 0.005, 0.01, 0.05, 0.5, 1, 2, 3, 5]
delta_vals = [1, 2, 3, 5]

root_folder = 'sumo/test_results'
test_results = 'newoptim-linear-False-test_seed_5318008_robust_factor_-1/0.01-train_data.csv'

full_path = os.path.join(root_folder, test_results)

das_data = pd.read_csv(full_path)

delta_vals = [0.01,0.5]

seeds = [99, 22, 23, 10]
for delta in delta_vals[:2]:
    for seed in seeds:

        test_results = f'newoptim-linear-False-test_seed_{seed}_robust_factor_-1/{delta}-train_score_data.csv'
        full_path = os.path.join(root_folder, test_results)
        das_data = pd.read_csv(full_path)
        plt.plot(das_data['scores'])
        plt.xlabel("Episode")
        plt.ylabel('Return')

        plt.title(f'Episode Return during training, delta = {delta}, seed = {seed}')
        plt.savefig(f'lossplots/Score-{delta}-{seed}.png')
        plt.clf()