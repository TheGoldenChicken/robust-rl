import sys
from tqdm import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
#delta_vals = [0.1, 0.001, 0.005, 0.01, 0.05, 0.5, 1, 2, 3, 5]

delta_vals = [0.1, 1, 2, 3, 5, 0.5, 0.4, 0.2]#, 3, 4, 5]
delta_vals += [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5]



root_folder = 'sumo/test_results/DQN_sumo-8000-frames'
root_folder = 'cliff_car/test_results/SlowCliffcar-newoptim-linear-False-test_seed_7777_robust_factor_-1'

times = []
for delta in delta_vals:
    try:
        time_data = pd.read_csv(os.path.join(root_folder, f'{delta}-time_data.csv'))
    except:
        pass
    times.append(time_data['training_time'][0])

arr = [1403.10,1462.05, 1473.08, 1456.27, 1445.29, 1446.38, 1445.29, 1456.27, 1447.88, 1477.99]
print(times)
print(np.mean(arr), np.std(arr))

#arr = [368.00, 362.38, 367.63, 367.80, 366.47, 308.60, 296.41, 263.85, 248.79, 225.66]
#print(np.mean(arr), np.std(arr))

# das_dat = np.load('itspersec.npy')
# plt.plot(das_dat)
# plt.show()
# #
# total_iterations = 100000
#
# # Create a file to save the IPS stats
# log_file = open('ips_log.txt', 'w')
# some_arr = np.array([1])
# ipss = []
# # Create a tqdm progress bar
# with tqdm(total=total_iterations) as pbar:
#     start_time = datetime.now()
#     for i in range(total_iterations):
#         # Perform your iteration logic here
#         # Update the progress bar
#         pbar.update(1)
#
#         # Calculate iterations per second
#         current_time = datetime.now()
#         some_arr = np.append(some_arr, 1)
#         elapsed_time = (current_time - start_time).total_seconds()
#         ips = (i + 1) / elapsed_time
#
#         # Save the IPS stat to the log file
#         # log_file.write(f"Iteration: {i + 1}, IPS: {ips:.2f}\n")
#         ipss.append(ips)
#         # log_file.write(ips)
#         # log_file.flush()
#
# ipss = np.array(ipss)
# np.save('itspersec', ipss)
#
# # Close the log file
# log_file.close()