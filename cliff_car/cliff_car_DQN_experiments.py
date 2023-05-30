import torch
import random
import numpy as np
import os
from cliff_car_again import CliffCar
from sumo.replay_buffer import ReplayBuffer
from cliff_car_DQN import DQNCliffCarAgent
import time
import pandas as pd

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
    env = CliffCar()

    # Replay buffer parameters - Should not be changed!
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    batch_size = 40
    state_max, state_min = np.array([env.max_min[0]]), np.array([env.max_min[1]])
    ready_when = 300 # Just about the same as robust agent with 3/10
    size = 10000

    # Should have converged somewhat at this point
    num_framess = [2500,5000,8000,10000,12000]
    for frames in num_framess:
        num_frames = frames

        # Agent parameters - Should not be changed!
        state_dim = 1
        grad_batch_size = 10
        replay_buffer_size = 500
        max_min = [[env.max_min[0]],[env.max_min[1]]]
        epsilon_decay = 1/2000
        target_update = 300

        seed = 6969

        # TODO: Fix ugly formatting here, not really becoming of a serious researcher
        test_name = f'DQN_cliffcar-{num_frames}-frames'

        if not os.path.isdir(f'test_results/{test_name}'):
            os.mkdir(f'test_results/{test_name}',)

        with open(f'test_results/{test_name}/hyperparams.txt', 'w') as f:
            f.write(f'''\
    Batch size, state_max, state_min, ready_when, num_frames,\
    replay_buffer_size, max_min, epsilon_decay\n\
    {batch_size}\n{state_max}\n{state_min}\n{ready_when}\n\
    {num_frames}\n{grad_batch_size}\n{replay_buffer_size}\n{max_min}\n{epsilon_decay}\
            ''')

        seeds = [6969, 4242, 6942, 420, 123, 5318008, 23, 22, 99, 10]

        for seed in seeds:

            seed_everything(seed)

            replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=size, batch_size=32, ready_when=300)
            agent = DQNCliffCarAgent(env=env, replay_buffer=replay_buffer, target_update=300, epsilon_decay=epsilon_decay,
                                 max_epsilon=1.0, min_epsilon=0.1, gamma=0.99)

            train_start = time.time()
            train_data = agent.train(num_frames, plotting_interval=999999)
            train_end = time.time()
            test_start = train_end
            test_data = agent.test(test_games=100, render_games=0)
            test_end = time.time()

            # States to extract q-values from
            # states = torch.FloatTensor(np.linspace(0,1, 1200)).reshape(-1,1).to(agent.device)
            # q_vals = agent.get_q_vals(states)

            test_columns = ['states_actions_rewards']
            train_columns = ['scores', 'losses', 'epsilons']
            time_columns = ['training_time', 'testing_time']
            time_data = [train_end - train_start, test_end - test_start]

            train_scores = pd.DataFrame({train_columns[0]: train_data[0]}) # Scores
            train_df = pd.DataFrame({train_columns[i]: train_data[i] for i in range(1, len(train_data))}) # Losses, epsilons
    #        test_df = pd.DataFrame({test_columns[i]: test_data[i] for i in range(len(test_data))})
            time_df = pd.DataFrame({time_columns[i]: [time_data[i]] for i in range(len(time_data))}) # Training test, testing time

            train_scores.to_csv(f'test_results/{test_name}/seed-{seed}-train_score_data.csv')
            train_df.to_csv(f'test_results/{test_name}/seed-{seed}-train_data.csv')
            np.save(f'test_results/{test_name}/seed-{seed}-test_data.npy', test_data)
            # np.save(f'test_results/{test_name}/seed-{seed}-q_vals.npy', q_vals)
     #       test_df.to_csv(f'test_results/{test_name}/{delta}-test_data.csv')
            time_df.to_csv(f'test_results/{test_name}/seed-{seed}-time_data.csv')
            agent.save_model(f'test_results/{test_name}/seed-{seed}-model')

            torch.cuda.empty_cache()
