import sys
import os

# Add current folder to path
sys.path.append('..')
# Set the current working directory to the folder this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import random
import numpy as np
from cliff_car_env import CliffCar
from replay_buffer import TheCoolerReplayBuffer
from cliff_car_robust_agent import RobustCliffCarAgent
from network import Network, RadialNetwork2d
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
    obs_dim = env.OBS_DIM
    action_dim = env.ACTION_DIM
    batch_size = 100
    fineness = 10
    ripe_when = None
    state_max, state_min = np.array(env.max_min[0]), np.array(env.max_min[1])
    ready_when = 10
    num_neighbours = 2
    bin_size = 500

    # Should have converged somewhat at this point
    train_frames = 5000 # 12000

    # Agent parameters - Should not be changed!
    grad_batch_size = 10 # 10
    replay_buffer_size = 1000 #1000
    max_min = [[env.max_min[0]],[env.max_min[1]]]
    epsilon_decay = 1/(train_frames*1.25) # default: 1/5000

    # Seeds
    seeds = [1]
    seed_id = 1000 # used for easily change seed
    
    delta_vals = [0.01, 1, 3]
    
    factors = [-1, 1] # Whether to add or subtract robust estimator from reward
    
    linear_only = [True, False] # Whether to use linear or quadratic approximation
    
    seeds = [seed * seed_id for seed in seeds]
    
    for seed in seeds:
        for linear in linear_only:
            for factor in factors:
                # TODO: Fix ugly formatting here, not really becoming of a serious researcher
                test_name = rf'linear={linear}-seed={seed}-factor={factor}'

                if not os.path.exists(rf'test_results\{test_name}'):
                    os.makedirs(rf'test_results\{test_name}')
                    
                with open(rf'test_results\{test_name}\hyperparams.txt', 'w') as f:
                    f.write(f'''\
            Batch size, Fineness, ripe_when, state_max, state_min, ready_when, num_neighbours, bin_size, train_frames,\
            grad_batch_size, replay_buffer_size, max_min, epsilon_decay \n\
            {batch_size}\n{fineness}\n{ripe_when}\n{state_max}\n{state_min}\n{ready_when}\n{num_neighbours}\n{bin_size}\n\
            {train_frames}\n{grad_batch_size}\n{replay_buffer_size}\n{max_min}\n{epsilon_decay}\n{seed}\
                    ''')

                for delta in delta_vals:
                    
                    print(f"Started training: seed: {seed}, linear: {linear}, factor: {factor}, delta: {delta}")
                    seed_everything(seed)

                    env = CliffCar(noise_var = 0.05)

                    replay_buffer = TheCoolerReplayBuffer(obs_dim=obs_dim, bin_size=bin_size, batch_size=batch_size,
                                                          fineness=fineness, num_actions=action_dim, state_max=state_max,
                                                          state_min=state_min, ripe_when=ripe_when, ready_when=ready_when,
                                                          num_neighbours=num_neighbours, tb=True)

                    agent = RobustCliffCarAgent(env=env, replay_buffer=replay_buffer, network = RadialNetwork2d,
                                            grad_batch_size=grad_batch_size,
                                            delta=delta, epsilon_decay=epsilon_decay, max_epsilon=1.0, min_epsilon=0.1,
                                            gamma=0.99, robust_factor=factor, linear_only=linear)

                    train_start = time.time()
                    train_data = agent.train(train_frames = train_frames,
                                             test_interval = 1000,
                                             test_games = 20,
                                             do_test_plots = True,
                                             test_name_prefix = "test-delete-me")
                    train_end = time.time()
                    test_start = train_end
                    # test_data = agent.test(test_games=100, render_games=0)
                    test_end = time.time()

                    # States to extract q-values from
                    # Here, we have to make a grid of q vals
                    X, Y = np.mgrid[env.BOUNDS[0]:env.BOUNDS[2]:150j, env.BOUNDS[1]:env.BOUNDS[3]:150j]
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

                    train_scores.to_csv(rf'test_results\{test_name}\{delta}-train_score_data.csv')
                    train_df.to_csv(rf'test_results\{test_name}\{delta}-train_data.csv')
                    # np.save(rf'test_results\{test_name}\{delta}-test_data.npy', test_data)
                    np.save(rf'test_results\{test_name}\{delta}-q_vals.npy', q_vals)
                    np.save(rf'test_results\{test_name}\{delta}-betas.npy', np.array(agent.betas))
             #       test_df.to_csv(rf'test_results\{test_name}\{delta}-test_data.csv')
                    time_df.to_csv(rf'test_results\{test_name}\{delta}-time_data.csv')
                    agent.save_model(rf'test_results\{test_name}\{delta}-model')

                    torch.cuda.empty_cache()
