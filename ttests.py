import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sumo.sumo_agent import SumoAgent
from sumo.sumo_pp import SumoPPEnv
import torch
import sys
import scipy.stats


# def compare_with_dqn(seeds, delta_vals, linear=True):
#     # Load Q-values for DQN agent
#     dqn_q_vals = [np.load(f'sumo/test_results/DQN_sumo-8000-frames/seed-{seed}-q_vals.npy')
#                   for seed in seeds]
#
#     # Load Q-values for all robust agents
#     paths = [[f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy'
#                      for seed in seeds] for delta in delta_vals]
#
#     all_q_vals = [[np.load(path) for path in delta] for delta in paths]
#
#     return dqn_q_vals, all_q_vals
#
# def get_SAR(seeds, delta_vals, linear=True):
#     dqn_SAR = [np.load(f'sumo/test_results/DQN_sumo-8000-frames/seed-{seed}-test_data.npy')
#                   for seed in seeds]
#
#     # Load Q-values for all robust agents
#     paths = [[f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-test_data.npy'
#                      for seed in seeds] for delta in delta_vals]
#
#     all_SAR = [[np.load(path) for path in delta] for delta in paths]
#
#     return dqn_SAR, all_SAR

def make_t_test_SAR(dqn_sar, all_sar, deltas, seeds, linear=True, with_dqn=True):

    dqn_sar = [np.load(f'sumo/test_results/DQN_sumo-8000-frames/seed-{seed}-test_data.npy') for seed in seeds]

    # Make test for difference with DQN performance
    if with_dqn:
        for i in deltas:
            current_sar = [np.load(f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{i}-test_data.npy') for seed in seeds]
            dqn_mean_state = [np.nanmean(dqn_sar[r][:, :, 0]) for r, seed in enumerate(dqn_sar)]
            robust_mean_state = [np.nanmean(current_sar[r][:, :, 0]) for r, seed in enumerate(current_sar)]
            stats, p_value = scipy.stats.ttest_ind(a=dqn_mean_state, b=robust_mean_state, alternative='greater')
            print(f"{i}: p_value: {p_value}, stats: {stats}, mean_dqn: {np.mean(dqn_mean_state)}, mean_{i}: {np.mean(robust_mean_state)}, var_dqn: {np.std(dqn_mean_state)}, var_{i}: {np.std(robust_mean_state)}")

    else:
        for i in range(1, len(deltas[:])):
            current_delta_sar = [np.load(f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{deltas[i]}-test_data.npy') for seed in seeds]
            last_delta_sar = [np.load(f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{deltas[i-1]}-test_data.npy') for seed in seeds]

            current_delta_sar = [np.nanmean(current_delta_sar[r][:, :, 0]) for r, seed in enumerate(current_delta_sar)]
            last_delta_sar = [np.nanmean(last_delta_sar[r][:, :, 0]) for r, seed in enumerate(last_delta_sar)]

            stats, p_value = scipy.stats.ttest_ind(a=last_delta_sar, b=current_delta_sar, alternative='greater')
            print(f"{deltas[i-1]} greater {deltas[i]}: p_value: {p_value}, stats: {stats}")



    # Make test for difference in two previous delta values


def make_t_test_q_vals(dqn_q_vals, all_q_vals):

    # Get first time left > right or greater than NoOp
    dqn_first_point = [np.where((dqn_q_vals[i][239:, 2] > dqn_q_vals[i][239:, 1]) | (dqn_q_vals[i][239:, 0] > dqn_q_vals[i][239:, 1]))[0][0] for i, r
     in enumerate(dqn_q_vals)]

    all_q_vals_first_point = [[np.where((vals[i][239:, 2] > vals[i][239:, 1]) | (vals[i][239:, 0] > vals[i][239:, 1]))[0] for i, r
     in enumerate(vals)] for vals in all_q_vals[:6]]

    all_q_vals_first_point = [
        [np.where((vals[i][239:, 2] > vals[i][239:, 1]) | (vals[i][239:, 0] > vals[i][239:, 1]))[0] for i, r
         in enumerate(vals)] for vals in all_q_vals[:]]

    list_of_deltas = []
    for i, delta in enumerate(all_q_vals_first_point):
        list_of_seeds = []
        for r, seed in enumerate(delta):
            if any(seed):
                list_of_seeds.append(all_q_vals_first_point[i][r][0] + 240)

            else:
                list_of_seeds.append(0 + 240)
        list_of_deltas.append(list_of_seeds)



delta_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5]
# delta_vals = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]

# dqn_sar, all_sar = get_SAR(seeds, delta_vals, linear=True)
make_t_test_SAR(None, None, delta_vals, seeds, linear=False, with_dqn=True)

