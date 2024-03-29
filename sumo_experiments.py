import torch
import random
import numpy as np
import os
from sumo.sumo_pp import SumoPPEnv
from sumo.replay_buffer import TheCoolerReplayBuffer
from sumo.robust_sumo_agent import RobustSumoAgent
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


    seed_everything(6969)
    # environment
    line_length = 1000
    env = SumoPPEnv(line_length=line_length)

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
    num_frames = 2500

    # Agent parameters - Should not be changed!
    state_dim = 1
    grad_batch_size = 10
    replay_buffer_size = 500
    max_min = [[env.max_min[0]],[env.max_min[1]]]
    epsilon_decay = 1/1000

    # Don't really know the good name to call it
    # TODO: Fix ugly formatting here, not really becoming of a serious researcher
    test_name = 'test_2_robust_positive'
    if not os.path.isdir(f'sumo/test_results/{test_name}'):
        os.mkdir(f'sumo/test_results/{test_name}',)
    with open('sumo/test_results/test_1/hyperparams.txt', 'w') as f:
        f.write(f'''\
Batch size, Fineness, ripe_when, state_max, state_min, ready_when, num_neighbours, bin_size, num_frames,\
grad_batch_size, replay_buffer_size, max_min, epsilon_decay \n\
{batch_size}\n{fineness}\n{ripe_when}\n{state_max}\n{state_min}\n{ready_when}\n{num_neighbours}\n{bin_size}\n\
{num_frames}\n{grad_batch_size}\n{replay_buffer_size}\n{max_min}\n{epsilon_decay}\
        ''')

    delta_vals =[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]
    # delta = 1

    for delta in delta_vals:

        seed_everything(6969)
        replay_buffer = TheCoolerReplayBuffer(obs_dim=obs_dim, bin_size=bin_size, batch_size=batch_size, fineness=fineness,
                                              num_actions=action_dim, state_max=state_max, state_min=state_min,
                                              ripe_when=ripe_when, ready_when=ready_when, num_neighbours=num_neighbours,
                                              tb=True)


        agent = RobustSumoAgent(env=env, replay_buffer=replay_buffer, grad_batch_size=grad_batch_size, delta=delta,
                                epsilon_decay=epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None)



        train_start = time.time()
        train_data = (scores, losses, epsilons) = agent.train(num_frames, plotting_interval=999999)
        train_end = time.time()
        test_start = train_end
        test_data = agent.test(test_games=100, render_games=0)
        test_end = time.time()

        test_columns = ['Test Scores', 'Episode mean cliff dists']
        train_columns = ['Scores', 'Losses', 'Epsilons']
        time_columns = ['Training Time', 'Testing Time']
        time_data = [train_end - train_start, test_end - test_start]

        train_scores = pd.DataFrame({train_columns[0]: train_data[0]}) # Scores
        train_df = pd.DataFrame({train_columns[i]: train_data[i] for i in range(1, len(train_data))}) # Losses, epsilons
        test_df = pd.DataFrame({test_columns[i]: [test_data[i]] for i in range(len(test_data))}) # Losses, epsilons
        time_df = pd.DataFrame({time_columns[i]: [time_data[i]] for i in range(len(time_data))}) # Training test, testing time

        train_scores.to_csv(f'sumo/test_results/{test_name}/{delta}-train_score_data.csv')
        train_df.to_csv(f'sumo/test_results/{test_name}/{delta}-train_data.csv')
        test_df.to_csv(f'sumo/test_results/{test_name}/{delta}-test_data.csv')
        time_df.to_csv(f'sumo/test_results/{test_name}/{delta}-time_data.csv')
        agent.save_model(f'sumo/test_results/{test_name}/{delta}-model')

        torch.cuda.empty_cache()
# def robust_agent_testing(seed, testing_runs,)