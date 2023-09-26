# TODO: WE HAVE A PROBLEM!
# TODO: ACTIONS!!! CAN COME AS NUMPY ARRAYS??? WTFFFF
# TODO: MAKE THE WHOLE THING WORK IN CASE ACTIONS COME AS NUMPY ARRAYS AND NOT INTEGERS, THAT WOULD BE STUPIDDDD
# TODO GRIDKEYS: Some way of adding variable fineness to the different grid keys
# TODO: IMPORTANT: Find out if all types (tuples, np arrays, such)... are correct!
# Must be done in create_grid_keys
# Must change mult_dim og single_dim intepreter til at passe med forskellige fineness
# TODO: Det der skal lægges til INdex af action skal ændres til at være sum(actions) hvis actions er en liste
# TODO: CONSIDER ADDING NUM_NIEGHBOURS (CHECK HOW MANY GRIDS) TO AGENT

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import random
from IPython.display import clear_output
from sumo.network import Network
from tqdm import tqdm
from robust_sumo_agent import RobustSumoAgent
from sumo_pp import SumoPPEnv
import os
import sys
sys.path.append(os.getcwd() + '/sumo')
# finenesss = used for replay buffer
# state_dim = used for replay and network
# action_dim = used for select action
# Batch_size = used for replay_buffer
# replay_buffer_size = used for replay_buffer
# self.max, self.min can be moved to the sumo environment
# ripe_when, moved to replay buffer
# number_neighbours - move to replay buffer (default 2)

class EnsembleSumoTestAgent:
    def __init__(self, env, agents):
        self.env = env
        self.max, self.min = np.array(self.env.max_min[0]), np.array(self.env.max_min[1])

        self.agents = agents

        # Should work for tensors and numpy...
        self.state_normalizer = lambda state: (state - self.min)/(self.max - self.min)


    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        selected_actions = []
        
        for agent in self.agents:
            selected_actions.append(agent.select_action(state))
        
        # Do majority voting between the agents
        selected_action = np.argmax(np.bincount(selected_actions))    
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done

    def test(self, test_games=100, render_games=0, render_speed=60):
        """
        Test the agent
        :param test_games: number of test games to get score from
        :param render_games: number of rendered games to inspect qualitatively
        :param render_speed: frame_rate of the rendered games
        :return:
        """
        all_sar = [] # All state_action_reward

        for i in range(test_games):
            state = self.env.reset()
            done = False

            # NOTE that we also get the last state, action, reward when the environment terminates here...

            sar = [(np.nan, np.nan, np.nan)] * self.env.max_duration

            i = 0
            # Changed here from training, since we play games till the end, not for a certain number of steps (frames)
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                if np.isnan(reward): # Sometimes rewards become nan values... who knows why
                    reward = 0

                # Using item here because they are numpy arrays... stupid
                sar[i] = (state.item(), action.item(), reward)

                i += 1

                state = next_state

            all_sar.append(sar)

        if render_games > 0:
            self.env.init_render()
            self.env.frame_rate = render_speed

        for i in range(render_games):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                self.env.render()

                state = next_state

        return np.array(all_sar)



        return np.array(all_sar)

if __name__ == "__main__":
    env = SumoPPEnv()
    delta_val = 0.001
    seeds = [6969, 4242, 123, 420, 6942, 5318008, 23, 22, 99, 10]
    agents = []
    for seed in seeds:
        print(os.getcwd())
        root_folder = f'test_results/truenonlinear-linear-False-test_seed_{seed}_robust_factor_-1/{delta_val}-model'
        agent = RobustSumoAgent(env=env, replay_buffer=None, grad_batch_size=None,
                                delta=delta_val, epsilon_decay=1/2000, max_epsilon=0, min_epsilon=0,
                                gamma=0.99, robust_factor=-1, linear_only=False)

        try:
            agent.load_model(root_folder)
        except:
            agent.load_model(root_folder + '.zip')
        agents.append(agent)

    ensemble = EnsembleSumoTestAgent(env, agents)
    ensemble.test(test_games=0, render_games=10, render_speed=60)