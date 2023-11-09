import sys
import os

# # Add current folder to path
# sys.path.append('..')
# # Set the current working directory to the folder this file is in
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import random
import numpy as np
from cliff_car_env import CliffCar
from replay_buffer import TheCoolerReplayBuffer, ReplayBuffer
from cliff_car_robust_agent import RobustCliffCarAgent
from cliff_car_dqn_agent import DQNCliffCarAgent
from network import RadialNetwork2d, RadialNonLinearNetwork2d
import time
import pandas as pd
import argparse
import wandb

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

def create_directories(seed, delta, identifier):
    test_name_body = rf'delta-{delta}_seed-{seed}'
    test_name = test_name_body

    # If the test exists, add a counter to the name
    counter = 0
    while os.path.exists(test_name):
        counter += 1
        test_name = test_name_body + f'-{counter}'

    path = os.path.join("results", identifier, test_name)
    # Experiment directory
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join("results", identifier, test_name, "training")
    # Training plots directory
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join("results", identifier, test_name, "evaluation")
    # Evaluation plots directory
    if not os.path.exists(path):
        os.makedirs(path)

    return ["results", identifier, test_name] # Return path to test directory

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Robust RL on Cliff Car')

    # Parameters changed most often
    parser.add_argument('--seed', type=int, nargs='+',
                        help='Seed. Either None (random), int or list',
                        required=True)
    parser.add_argument('--delta', type=float, nargs='+',
                        help = 'Delta value for robustness. Either int or list',
                        default=0.01)
    
    # Wandb: Weights and Biases
    parser.add_argument('--wandb_key', type = str, default = None, required=False)

    parser.add_argument('--train_identifier', default = "unnamed", help= "Used to identify the training run.", required=False)

    # Agent and environment type
    parser.add_argument('--robust_agent', default=False, required=False, action='store_true') # Robust or DQN agent
    parser.add_argument('--discrete_env', default=False, required=False, action='store_true')
    parser.add_argument('--non_linear', default=True, required=False, action='store_true') # Non-linear or linear network

    # Replay buffer parameters
    parser.add_argument('--fineness', type=int, default=2, required=False)
    parser.add_argument('--ripe_when', type=int, default=None, required=False)
    parser.add_argument('--ready_when', type=int, default=10, required=False)
    parser.add_argument('--num_neighbours', type=int, default=2, required=False)
    parser.add_argument('--replay_buffer_size', type=int, default=500, required=False)
    parser.add_argument('--bin_size', type=int, default=200, required=False)

    parser.add_argument('--train_frames', type=int, default=1000000, required=False)
    parser.add_argument('--test_interval', type=int, default=10000, required=False)
    parser.add_argument('--test_games', type=int, default=100, required=False)

    # Training parameters
    parser.add_argument('--robust_batch_size', type=int, default=75, required=False)
    parser.add_argument('--grad_batch_size', type=int, default=32, required=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001, required=False)
    parser.add_argument('--weight_decay', type=float, default=0, required=False)

    # Epsilon values
    parser.add_argument('--epsilon_decay', type=float, default=1/10000, required=False)
    parser.add_argument('--max_epsilon', type=float, default=1, required=False)
    parser.add_argument('--min_epsilon', type=float, default=0.1, required=False)
    parser.add_argument('--gamma', type=float, default=0.95, required=False)

    # Environment parameters
    parser.add_argument('--noise_var', type=float, default=[[0.5, 0],[0, 0.5]], required=False)
    parser.add_argument('--noise_mean', type=float, default=[0, 0], required=False)

    # Network parameters
    parser.add_argument('--radial_basis_dist', type=float, default=1, required=False)
    parser.add_argument('--radial_basis_var', type=float, default=7, required=False)
    
    parser.add_argument('--silence_tqdm', type=bool, default=False, required=False)

    args = parser.parse_args()

    # Set up seed
    if type(args.seed) == int:
        args.seed = [args.seed]

    # Change to list if not already
    if type(args.delta) == float:
        args.delta = [args.delta]

    print(">>>>>>>>>>>>>>>>>>>>>> " + str(args.seed) + " " + str(type(args.seed)))

    # Set up wandb
    if args.wandb_key != None:
        try:
            # "ec26ff6ba9b98d017cdb3165454ce21496c12c35"
            wandb.login(key = args.wandb_key)
            wandb_active = True
        except:
            print("Invalid wandb key. Continuing without wandb...")
    else:
        wandb_active = False

    # Set up train identifier
    args.train_identifier += f"-{time.ctime()}" # Add date for uniqueness
    args.train_identifier = args.train_identifier.replace(':', '-').replace(' ','_') # Reformat to avoid path issues

    for seed in args.seed:
        for delta in args.delta:

            path_components = create_directories(seed = seed,
                                                 delta = delta,
                                                 identifier = args.train_identifier)

            path = os.path.join(*path_components, "hyperparams.txt")
            # Print hyperparameters to file for reproducibility
            with open(path, 'w') as f:
                f.write(f"Seed: {seed}\nDelta: {delta}\nFineness: {args.fineness}\n \
                        Ripe when: {args.ripe_when}\nReady when: {args.ready_when}\n \
                        Num neighbours: {args.num_neighbours}\nBin size: {args.bin_size}\n \
                        Replay buffer size: {args.replay_buffer_size}\nRobust batch size: {args.robust_batch_size}\n \
                        Train frames: {args.train_frames}\nGrad batch size: {args.grad_batch_size}\n \
                        Epsilon decay: {args.epsilon_decay}\n Max epsilon: {args.max_epsilon}\n\
                        Min epsilon: {args.min_epsilon}\n Learning rate: {args.learning_rate}\n \
                        Weight decay: {args.weight_decay}\nNoise var: {args.noise_var}\n \
                        Noise mean: {args.noise_mean}\nRadial basis dist: {args.radial_basis_dist}\n \
                        Radial basis var: {args.radial_basis_var}")

            # Set up weights and biases if key is provided
            if wandb_active:
                wandb.init(
                    project="robust-rl-cliff-car", 
                    name=f"{path_components[1]}-{path_components[2]}", 
                    config={
                        "seed": seed,
                        "delta": delta,
                        "fineness": args.fineness,
                        "ripe_when": args.ripe_when,
                        "ready_when": args.ready_when,
                        "num_neighbours": args.num_neighbours,
                        "bin_size": args.bin_size,
                        "replay_buffer_size": args.replay_buffer_size,
                        "robust_batch_size": args.robust_batch_size,
                        "train_frames": args.train_frames,
                        "grad_batch_size": args.grad_batch_size,
                        "epsilon_decay": args.epsilon_decay,
                        "max_epsilon": args.max_epsilon,
                        "min_epsilon": args.min_epsilon,
                        "gamma": args.gamma,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "noise_var": args.noise_var,
                        "noise_mean": args.noise_mean,
                        "radial_basis_dist": args.radial_basis_dist,
                        "radial_basis_var": args.radial_basis_var
                    })
                
            ### SETUP FINISHED - START TRAINING ###
            print("==============================================")
            print(f">>> Started training; seed: {seed}, delta: {delta}")
            seed_everything(seed)

            env = CliffCar(**vars(args))


            if args.non_linear:
                network = RadialNonLinearNetwork2d
            else:
                network = RadialNetwork2d   

            state_max, state_min = np.array(env.max_min[0]), np.array(env.max_min[1])
            if args.robust_agent:
                replay_buffer = TheCoolerReplayBuffer(obs_dim=env.OBS_DIM, num_actions=env.ACTION_DIM,
                                                    state_min=state_min, state_max=state_max,
                                                    batch_size = args.robust_batch_size,
                                                    **vars(args))

                agent = RobustCliffCarAgent(env=env, replay_buffer=replay_buffer, network = network,
                                            **vars(args))
            else: # DQN agent
                replay_buffer = ReplayBuffer(obs_dim=env.OBS_DIM, size=args.bin_size,
                                             batch_size=args.grad_batch_size,
                                             **vars(args))
                agent = DQNCliffCarAgent(env=env, replay_buffer=replay_buffer, network = network,
                                         **vars(args))

            train_start = time.time()
            train_data = agent.train(plot_path = path_components, wandb_active=wandb_active,
                                     **vars(args))
            train_end = time.time()

            print("Training finished. Time: " + str(train_end - train_start))
            ### TRAINING FINISHED - START EVALUATION ###
            print("==============================================")
            print(f">>> Started evaluation; seed: {seed}, delta: {delta}")

            test_start = train_end
            test_data = agent.test(test_games=args.test_games, render_games=0, frame = args.train_frames)
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

            path = os.path.join(*path_components, "training", "train_score_data.csv")
            train_scores.to_csv(path)
            path = os.path.join(*path_components, "training", "train_data.csv")
            train_df.to_csv(path)
            # np.save(rf'test_results_{experiment_id}\{test_name}\{delta}-test_data.npy', test_data)
            path = os.path.join(*path_components, "evaluation", "q_vals.npy")
            np.save(path, q_vals)
            # path = os.path.join(*path_components, "evaluation", "betas.npy")
            # np.save(path, np.array(agent.betas))
        #       test_df.to_csv(rf'test_results_{experiment_id}\{test_name}\{delta}-test_data.csv')
            path = os.path.join(*path_components, "evaluation", "time_data.csv")
            time_df.to_csv(path)
            path = os.path.join(*path_components, "evaluation", "model")
            agent.save_model(path)

            torch.cuda.empty_cache()
            
            wandb.finish()
