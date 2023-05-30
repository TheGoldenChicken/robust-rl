import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sumo.sumo_agent import SumoAgent
from sumo.sumo_pp import SumoPPEnv
import torch
import sys

from cliff_car.cliff_car_again import CliffCar
from cliff_car.cliff_car_agent import CliffCarAgent

# states = torch.FloatTensor(np.linspace(0, 1000, 5000)).reshape(-1, 1).to('cuda')
# print(states)
# Load the .npy file

root_folder = 'sumo/test_results/test_seed_420_robust_factor_-1'
root_folder = 'sumo/test_results/Truelinear-test_seed_6969_robust_factor_-1'
root_folder = 'sumo/test_results/newoptim-linear-True-test_seed_5318008_robust_factor_-1'
sys.path.append(os.getcwd() + '/sumo')
sys.path.append('cliff_car')

def get_and_plot_csv_data(delta=None, path=None):
    if delta is not None:
        dat = pd.read_csv(os.path.join(root_folder, str(delta))+'-train_data.csv')
        score_dat = pd.read_csv(os.path.join(root_folder, str(delta)) + '-train_score_data.csv')

    elif path is not None:
        dat = pd.read_csv(path + '-train_data.csv')
        score_dat = pd.read_csv(path + '-train_score_data.csv')

    else:
        raise NotImplemented

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(score_dat['scores'])
    plt.title(f'Scores, Delta: {delta}')
    plt.subplot(132)
    plt.plot(dat['losses'])
    plt.title(f'Losses, Delta: {delta}')
    plt.subplot(133)
    plt.plot(dat['epsilons'])
    plt.title(f'Epsilons, Delta: {delta}')
    # plt.tight_layout()
    plt.show()

def get_q_vals(delta=None, path=None):
    if delta is not None:
        q_vals = np.load(os.path.join(root_folder, str(delta)) + '-q_vals.npy')

    elif path is not None:
        q_vals = np.load(path)
        # q_vals = np.load(path + '-q_vals.npy')

    else:
        raise NotImplemented

    return q_vals


def plot_q_vals(q_vals, delta=None, same_plot=True, vertical_lines=False, linear=None, save_folder=None):


    column1 = q_vals[:, 0]
    column2 = q_vals[:, 1]
    column3 = q_vals[:, 2]

    if not same_plot:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(column1)
        plt.title(f'Action 0, {delta}')
        plt.subplot(132)
        plt.plot(column2)
        plt.title('Action 1')
        plt.subplot(133)
        plt.plot(column3)
        plt.title('Action 2')
        plt.tight_layout()
    else:
        plt.plot(column1, color='red', label='0: Noop')
        plt.plot(column2, color='blue', label='1: Right (towards cliff)')
        plt.plot(column3, color='green', label='2: Left (away from cliff)')
        plt.legend()
        plt.title(f'Q-values for converged agent with Delta {delta} with linear={linear}')

    if vertical_lines:
        # Add lines to indicate where right becomes better action than left
        indices = np.where(column3 > column2)[0]
        indices1 = np.where(column1 > column2)[0]

        plt.axvline(x=indices1[0], color='green', linestyle='--')
        plt.axvline(x=indices[0], color='blue', linestyle='--')
        plt.axvline(x=1000, color='purple', linestyle='--')
        plt.axvline(x=240, color='red', linestyle='-')

        plt.text(500, 0.2, f'Left > Right: {indices[0]}\nNoOp > Right: {indices1[0]}\nCliff: {1000}\nStart Pos: 240',
                 rotation=0, va='bottom')

    if save_folder:
        plt.savefig(os.path.join(save_folder, f'q-vals-{delta}-avg{len(seeds)}seeds-{linear}.png'))

    plt.show()

# TODO: ADD Functionality FOR GETTING MEAN BETA_MAX VALUE FOR EACH UPDATE OF ROBUST ESTIMATOR
def get_and_plot_betas(delta, rolling_mean=10):
    betas = np.load(os.path.join(root_folder,str(delta)) + '-betas.npy')

    if rolling_mean:
        betas = [np.mean(betas[i:i+rolling_mean]) for i in range(0,len(betas)-10)]
    plt.plot(betas)
    plt.title(f"Beta_max values Delta: {delta}")
    plt.show()

def load_and_test_agent(delta, test_games, render_games):
    path = os.path.join(root_folder, str(delta)) + '-model'
    env = SumoPPEnv()
    # We define a barebones agent for the qualitaive test - Possible since select_action and test are never interfered with
    agent = SumoAgent(env, replay_buffer=None, epsilon_decay=None, model_path=path)
    test_results = agent.test(test_games=test_games, render_games=render_games)
    return test_results

def load_and_test_agent_cliff_car(delta, test_games, render_games):
    path = os.path.join('cliff_car', 'test_results', 'Cliffcar-newoptim-linear-False-test_seed_6969_robust_factor_-1', '0.5-model')

    env = CliffCar()
    # We define a barebones agent for the qualitaive test - Possible since select_action and test are never interfered with
    agent = CliffCarAgent(env, replay_buffer=None, epsilon_decay=None, model_path=path)
    agent.test(test_games=test_games, render_games=render_games)


def plot_average_of_seeds(seeds, delta_vals, linear=True, save_folder=None):

    paths = [[f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy'
                     for seed in seeds] for delta in delta_vals]

    for i in range(len(paths)):
        q_vals = np.array([get_q_vals(path=path) for path in paths[i]])
        plot_q_vals(np.mean(q_vals, axis=0), delta=delta_vals[i], vertical_lines=True,
                    linear=linear, save_folder=save_folder)

def plot_sar_stats(seeds, delta_vals, linear=True):

    paths = [[f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-test_data.npy'
                     for seed in seeds] for delta in delta_vals]

    for i in range(len(paths)):
        sar_data = np.array([np.load(path) for path in paths[i]])
        # State Hist plotting
        plt.hist(sar_data[:,:,:,0].flatten(), bins=100)
        plt.title(f"States visited for delta value {delta_vals[i]}")
        plt.show()

        # Return Stats
        returns = np.sum(sar_data[:,:,:,2],axis=2) # Returns for each game should be 1000 - 10 seeds, 100 games each
        returns = returns[~np.isnan(returns)]# Remove NAN values - Don't know why they come...
        mean_return, var_return = np.mean(returns), np.var(returns)
        plt.hist(returns, bins=30)
        plt.title(f"Episode Returns for delta value {delta_vals[i]}")
        plt.axvline(x=mean_return, color='red', linestyle='-')
        plt.show()
        
def plot_sumo_states(seeds, delta_vals, linear=True):
    
    paths = [[f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-test_data.npy'
                     for seed in seeds] for delta in delta_vals]
    
    # Combine (not average) the states from all seeds
    for i in range(len(paths)):
        sar_data = np.array([np.load(path) for path in paths[i]])
        sar_data = np.concatenate(sar_data, axis=0)
        sar_data = sar_data[:,:,0].flatten()
        
        plt.hist(sar_data, bins=100)
        plt.title(f"States visited for delta value {delta_vals[i]}")
    plt.legend()
    plt.show()

# plot_sumo_states([6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10], [0.001, 0.005, 0.01], True)

        # sar_data = np.mean(sar_data, axis=0) # Average over seeds

# seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
# delta_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]
# save_folder = 'plots/q-vals'
# plot_average_of_seeds(seeds, delta_vals, False, save_folder=save_folder)
#
# plot_sar_stats(seeds, delta_vals, True)

# load_and_test_agent_cliff_car(0, 10, 10)
# load_and_test_agent(0.01, 10, 10)
#
# paths_linear = [[f'sumo/test_results/newoptim-linear-True-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy'
#                 for seed in seeds] for delta in delta_vals]
# paths_non_linear = [[f'sumo/test_results/newoptim-linear-False-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy'
#                 for seed in seeds] for delta in delta_vals]
#
# test_linear = [[f'sumo/test_results/newoptim-linear-True-test_seed_{seed}_robust_factor_-1/{delta}-test_data.npy'
#                 for seed in seeds] for delta in delta_vals]
#
# for i in range(len(paths_linear)):
#     q_vals = np.array([get_q_vals(path=path) for path in paths_linear[i]])
#     plot_q_vals(np.mean(q_vals, axis=0), delta=delta_vals[i], vertical_lines=True)
#

#
# # get_and_plot_q_vals(0.01, same_plot=True, vertical_lines=True)
# q_vals = get_q_vals(0.05)
# plot_q_vals(q_vals, 0.05, vertical_lines=True)
# get_and_plot_betas(0.05, rolling_mean=100)
# get_and_plot_csv_data(delta=0.05)
# # load_and_test_agent(delta=0.05, test_games=10, render_games=10)

# for delta in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]:
#     get_and_plot_csv_data(delta)
#     get_and_plot_q_vals(delta, same_plot=True, vertical_lines=True)
#     get_and_plot_betas(delta, rolling_mean=100)
#
# # #
