import sys
import os

# Add current folder to path
sys.path.append('..')
# Set the current working directory to the folder this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import random
import string
import numpy as np
from cliff_car_env import CliffCar
from replay_buffer import TheCoolerReplayBuffer
from cliff_car_robust_agent import RobustCliffCarAgent
from network import Network, RadialNetwork2d, RadialNonlinearNetwork2d
import time
import pandas as pd
from itertools import product

import wandb
wandb.login(key = "ec26ff6ba9b98d017cdb3165454ce21496c12c35")

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


if __name__ == "__main__":

    experiment_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    seed_everything(6969)

    env = CliffCar()

    # Replay buffer parameters - Should not be changed!
    obs_dim = env.OBS_DIM
    action_dim = env.ACTION_DIM
    state_max, state_min = np.array(env.max_min[0]), np.array(env.max_min[1])
    max_min = [[env.max_min[0]],[env.max_min[1]]]
    
    robust_batch_size = 75 # 100
    fineness = 5
    ripe_when = None
    ready_when = 10
    num_neighbours = 2
    bin_size = 500
    train_frames = 30000 # 12000
    noise_var = [[0.05, 0],[0, 0.05]]
    noise_mean = [0, 0]

    # Agent parameters - Should not be changed!
    grad_batch_size = 10 # 10
    replay_buffer_size = 500 #1000
    epsilon_decay = 1/5000 # default: 1/5000

    # Seeds
    seeds = [2002, 1003]
    delta_vals = [0.01]#, 0.5, 1]
    # factors = [-1, 1] # Whether to add or subtract robust estimator from reward
    # factors = [1, -1]
    factors = [1,-1]
    # linear_only = [False, True] # Whether to use linear or quadratic approximation
    linear_only = [False]
    finenesses = [2]#[2, 5, 10]
    # epsilon_decays = [1/2500, 1/1000, 1/500, 1/100]
    epsilon_decays = [1/15000] # default 1/3000
    bin_sizes = [200]
    
    # r_basis_diffs = [2, 1]
    r_basis_diffs = [1.5] #  default 2
    r_basis_vars = [7]
    
    for seed in seeds:
        for linear in linear_only:
            for factor in factors:
                for fineness in finenesses:
                    for epsilon_decay in epsilon_decays:
                        for bin_size in bin_sizes:
                            for r_basis_diff in r_basis_diffs:
                                for r_basis_var in r_basis_vars:
                                    test_name = rf'linear={linear}-seed={seed}-factor={factor}-fineness={fineness}-epsilon_decay={epsilon_decay}-bin_size={bin_size}-r_basis_diff={r_basis_diff}-r_basis_var={r_basis_var}'
                                    
                                    if not os.path.exists(rf'test_results_{experiment_id}\{test_name}'):
                                        os.makedirs(rf'test_results_{experiment_id}\{test_name}')
                                        
                                    with open(rf'test_results_{experiment_id}\{test_name}\hyperparams.txt', 'w') as f:
                                        f.write(f'''\
                                Batch size, Fineness, ripe_when, state_max, state_min, ready_when, num_neighbours, bin_size, train_frames,\
                                grad_batch_size, replay_buffer_size, max_min, epsilon_decay \n\
                                {robust_batch_size}\n{fineness}\n{ripe_when}\n{state_max}\n{state_min}\n{ready_when}\n{num_neighbours}\n{bin_size}\n\
                                {train_frames}\n{grad_batch_size}\n{replay_buffer_size}\n{max_min}\n{epsilon_decay}\n{seed}\
                                        ''')

                                    for delta in delta_vals:
                                        

                                        wandb.init(
                                            project="robust-rl-cliff-car", 
                                            name=f"delta={delta}-linear={linear}-factor={factor}-seed={seed}- \
                                                fineness={fineness}-epsilon_decay={epsilon_decay}-bin_size={bin_size}-id={experiment_id}", 
                                            config={
                                                "delta" : delta,
                                                "linear" : linear,
                                                "factor" : factor,
                                                "seed" : seed,
                                                "robust_batch_size" : robust_batch_size,
                                                "fineness" : fineness,
                                                "ripe_when" : ripe_when,
                                                "ready_when" : ready_when,
                                                "num_neighbours": num_neighbours,
                                                "bin_size" : bin_size,
                                                "train_frames" : train_frames,
                                                "grad_batch_size" : grad_batch_size,
                                                "replay_buffer_size" : replay_buffer_size,
                                                "epsilon_decay" : epsilon_decay,
                                                "noise_var" : noise_var,
                                                "noise_mean" : noise_mean
                                            })
                                        
                                        print(f"Started training: seed: {seed}, linear: {linear}, factor: {factor}, delta: {delta}")
                                        seed_everything(seed)

                                        env = CliffCar(noise_var = noise_var, noise_mean = noise_mean, r_basis_diff = r_basis_diff, r_basis_var = r_basis_var)

                                        replay_buffer = TheCoolerReplayBuffer(obs_dim=obs_dim, bin_size=bin_size, batch_size=robust_batch_size,
                                                                            fineness=fineness, num_actions=action_dim, state_max=state_max,
                                                                            state_min=state_min, ripe_when=ripe_when, ready_when=ready_when,
                                                                            num_neighbours=num_neighbours, tb=True)

                                        agent = RobustCliffCarAgent(env=env, replay_buffer=replay_buffer, network = RadialNetwork2d,
                                                                grad_batch_size=grad_batch_size,
                                                                delta=delta, epsilon_decay=epsilon_decay, max_epsilon=0.99, min_epsilon=0.05,
                                                                gamma=0.99, robust_factor=factor, linear_only=linear)

                                        train_start = time.time()
                                        train_data = agent.train(train_frames = train_frames,
                                                                test_interval = 1000,   
                                                                test_games = 50,
                                                                do_test_plots = True,
                                                                test_name_prefix = "-seed-" + str(seed) + "-linear-" + str(linear) + "-delta-" + str(delta) + "-factor-" + str(factor) + "-r_basis_diff-" + str(r_basis_diff) + "-r_basis_var-" + str(r_basis_var))
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

                                        train_scores.to_csv(rf'test_results_{experiment_id}\{test_name}\{delta}-train_score_data.csv')
                                        train_df.to_csv(rf'test_results_{experiment_id}\{test_name}\{delta}-train_data.csv')
                                        # np.save(rf'test_results_{experiment_id}\{test_name}\{delta}-test_data.npy', test_data)
                                        np.save(rf'test_results_{experiment_id}\{test_name}\{delta}-q_vals.npy', q_vals)
                                        np.save(rf'test_results_{experiment_id}\{test_name}\{delta}-betas.npy', np.array(agent.betas))
                                #       test_df.to_csv(rf'test_results_{experiment_id}\{test_name}\{delta}-test_data.csv')
                                        time_df.to_csv(rf'test_results_{experiment_id}\{test_name}\{delta}-time_data.csv')
                                        agent.save_model(rf'test_results_{experiment_id}\{test_name}\{delta}-model')
                                        
                                        # wandb.save(rf'test_results_{experiment_id}\{test_name}\{delta}-model')

                                        torch.cuda.empty_cache()
                                        
                                        wandb.finish()
