import sys
import os

# Add current folder to path
sys.path.append('..')
# Set the current working directory to the folder this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import random
import numpy as np
from cliff_car_again import CliffCar
from sumo.replay_buffer import TheCoolerReplayBuffer
from cliff_car_robust_agent import RobustCliffCarAgent
import time
import pandas as pd

def seed_everything(seed_value):
    """
    Thanks to Viktor Tolsager for showing me some other guy who's made this
    """

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# Old seed function
# seed = 777
# def seed_torch(seed):
#     torch.manual_seed(seed)
#     if torch.backends.cudnn.enabled:
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
# np.random.seed(seed)
# seed_torch(seed)



if __name__ == "__main__":

    # We seed before initializing environment
    seed_everything(6969)
    # environment
    # line_length = 1000 # Use env default val
    env = CliffCar()

    # Replay buffer parameters - Should not be changed!
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    batch_size = 40
    fineness = 100
    ripe_when = None
    state_max, state_min = np.array(env.max_min[0]), np.array(env.max_min[1])
    ready_when = 10
    num_neighbours = 2
    bin_size = 500

    # Should have converged somewhat at this point
    num_frames = 40000 # 12000

    # Agent parameters - Should not be changed!
    state_dim = 1
    grad_batch_size = 10 # 10
    replay_buffer_size = 1000 #1000
    max_min = [[env.max_min[0]],[env.max_min[1]]]
    epsilon_decay = 1/20000 # default: 1/5000


    # Seeds
    # seeds = [6969, 4242, 6942, 123, 420, 5318008,  23, 22, 99, 10]
    seeds = [7777]

    # seeds = [9000,9001,9002,9003,9004,9005,9006,9007,9008,9009]
        # seeds = [10000]

    # Delta values to test
    delta_vals = [0.1, 1, 2]#, 3, 4, 5]
    # delta_vals = [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]
    # delta_vals = [0.001]
    # Whether to add or subtract robust estimator from reward
    factors = [-1]
    # Whether to use linear or quadratic approximation
    linear_only = [False]
    #delta_vals = [0.01,0.1,0.05,1]
    for seed in seeds:
        print("Started training on seed: ", seed, "...")
        for linear in linear_only:
            for factor in factors:
                # TODO: Fix ugly formatting here, not really becoming of a serious researcher
                test_name = f'Cliffcar-newoptim-linear-{linear}-test_seed_{seed}_robust_factor_{factor}'

                if not os.path.isdir(f'test_results/{test_name}'):
                    os.mkdir(f'test_results/{test_name}',)
                with open(f'test_results/{test_name}/hyperparams.txt', 'w') as f:
                    f.write(f'''\
            Batch size, Fineness, ripe_when, state_max, state_min, ready_when, num_neighbours, bin_size, num_frames,\
            grad_batch_size, replay_buffer_size, max_min, epsilon_decay \n\
            {batch_size}\n{fineness}\n{ripe_when}\n{state_max}\n{state_min}\n{ready_when}\n{num_neighbours}\n{bin_size}\n\
            {num_frames}\n{grad_batch_size}\n{replay_buffer_size}\n{max_min}\n{epsilon_decay}\n{seed}\
                    ''')

                for delta in delta_vals:
                    seed_everything(seed)

                    env = CliffCar()

                    replay_buffer = TheCoolerReplayBuffer(obs_dim=obs_dim, bin_size=bin_size, batch_size=batch_size,
                                                          fineness=fineness, num_actions=action_dim, state_max=state_max,
                                                          state_min=state_min, ripe_when=ripe_when, ready_when=ready_when,
                                                          num_neighbours=num_neighbours, tb=True)

                    agent = RobustCliffCarAgent(env=env, replay_buffer=replay_buffer, grad_batch_size=grad_batch_size,
                                            delta=delta, epsilon_decay=epsilon_decay, max_epsilon=1.0, min_epsilon=0.1,
                                            gamma=0.99, robust_factor=factor, linear_only=linear)

                    train_start = time.time()
                    train_data = agent.train(num_frames, plotting_interval=999999)
                    train_end = time.time()
                    test_start = train_end
                    # test_data = agent.test(test_games=100, render_games=0)
                    test_end = time.time()

                    # States to extract q-values from
                    # Here, we have to make a grid of q vals
                    X, Y = np.mgrid[0:1:150j, 0:1:150j]
                    xy = np.vstack((X.flatten(), Y.flatten())).T
                    states = torch.FloatTensor(xy).to(agent.device)

                    q_vals = agent.get_q_vals(states)

                    test_columns = ['states_actions_rewards']
                    train_columns = ['scores', 'losses', 'epsilons']
                    time_columns = ['training_time', 'testing_time']
                    time_data = [train_end - train_start, test_end - test_start]

                    train_scores = pd.DataFrame({train_columns[0]: train_data[0]}) # Scores
                    train_df = pd.DataFrame({train_columns[i]: train_data[i] for i in range(1, len(train_data))}) # Losses, epsilons
            #        test_df = pd.DataFrame({test_columns[i]: test_data[i] for i in range(len(test_data))})
                    time_df = pd.DataFrame({time_columns[i]: [time_data[i]] for i in range(len(time_data))}) # Training test, testing time

                    train_scores.to_csv(f'test_results/{test_name}/{delta}-train_score_data.csv')
                    train_df.to_csv(f'test_results/{test_name}/{delta}-train_data.csv')
                    # np.save(f'test_results/{test_name}/{delta}-test_data.npy', test_data)
                    np.save(f'test_results/{test_name}/{delta}-q_vals.npy', q_vals)
                    np.save(f'test_results/{test_name}/{delta}-betas.npy', np.array(agent.betas))
             #       test_df.to_csv(f'test_results/{test_name}/{delta}-test_data.csv')
                    time_df.to_csv(f'test_results/{test_name}/{delta}-time_data.csv')
                    agent.save_model(f'test_results/{test_name}/{delta}-model')

                    torch.cuda.empty_cache()
