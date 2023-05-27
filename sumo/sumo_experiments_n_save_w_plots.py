import sys
import os

# Add current folder to path
sys.path.append('.')

import torch
import random
import numpy as np
from sumo_pp import SumoPPEnv
from replay_buffer import TheCoolerReplayBuffer
from robust_sumo_agent import RobustSumoAgent
import time
import pandas as pd
import matplotlib.pyplot as plt

def seed_everything(seed_value):
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


    seed_everything(6969)
    # environment
    # line_length = 1000 # Use env default val
    env = SumoPPEnv()

    # Replay buffer parameters - Should not be changed!
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    batch_size = 40
    fineness = 100
    ripe_when = None
    state_max, state_min = np.array([env.max_min[0]]), np.array([env.max_min[1]])
    ready_when = 10
    num_neighbours = 2
    bin_size = 1000

    # Should have converged somewhat at this point
    num_frames = 20000

    # Agent parameters - Should not be changed!
    state_dim = 1
    grad_batch_size = 10
    replay_buffer_size = 500
    max_min = [[env.max_min[0]],[env.max_min[1]]]
    epsilon_decay = 1/1000

    seed = 999
    for seed in [999]:#, 6942, 420, 123, 5318008, 23, 22, 99, 10]:
        #delta_vals = [0.5]
        # delta_vals =[0.01, 0.1, 0.5]#, 1, 5]
        # delta_vals = [1]
        delta_vals = [0.01,0.05,0.1,0.5,1,2]
        # delta = 1

        factor = -1

        # 3x2 grid of plots with distance between the plots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for i, delta in enumerate(delta_vals):
            seed_everything(seed)

            env = SumoPPEnv()

            replay_buffer = TheCoolerReplayBuffer(obs_dim=obs_dim, bin_size=bin_size, batch_size=batch_size, fineness=fineness,
                                                    num_actions=action_dim, state_max=state_max, state_min=state_min,
                                                    ripe_when=ripe_when, ready_when=ready_when, num_neighbours=num_neighbours,
                                                    tb=True)


            agent = RobustSumoAgent(env=env, replay_buffer=replay_buffer, grad_batch_size=grad_batch_size, delta=delta,
                                    epsilon_decay=epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, robust_factor=factor)

            # # Change
            # train_start = time.time()
            train_data = agent.train(num_frames, plotting_interval=999999)
            # train_end = time.time()
            # test_start = train_end
            test_data = agent.test(test_games=100, render_games=0)
            # test_end = time.time()

            states = torch.FloatTensor(np.linspace(0,1,1200)).reshape(-1,1).to(agent.device)

            q_vals = agent.get_q_vals(states)
            
            

            # Plot the q values for each action (dim(q_vals)=(len(states), action_dim))
            for j in range(action_dim):
                axs[i//2, i%2].plot(states[:,0], q_vals[:,j], label=f"Action {j}")
                axs[i//2, i%2].set_title(f"Delta = {delta}")
                axs[i//2, i%2].set_xlabel("State")
                axs[i//2, i%2].set_ylabel("Q Value")
                axs[i//2, i%2].legend()
                
            indices = np.where(q_vals[:,2] > q_vals[:,1])[0]
            if len(indices) != 0:
                print(f'Right > Left: {indices[0]/1200 }\nCliff: {0.84}')
                axs[i//2, i%2].axvline(x=indices[0]/1200, color='black', linestyle='--')
                axs[i//2, i%2].text(0.5, 0.2, f'Right > Left: {indices[0]/1200 }\nCliff: {0.84}',rotation=0, va='bottom')

            axs[i//2, i%2].axvline(x=0.84, color='purple', linestyle='--')

            
            torch.cuda.empty_cache()

        plt.show()

# def robust_agent_testing(seed, testing_runs,)