from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import random
from IPython.display import clear_output
from sumo.network import Network
from tqdm import tqdm


class EnsembleCliffTestAgent:
    def __init__(self, env, agents):
        self.env = env
        self.max, self.min = np.array(self.env.max_min[0]), np.array(self.env.max_min[1])


        self.agents = agents

        # Should work for tensors and numpy...
        self.state_normalizer = lambda state: (state - self.min)/(self.max - self.min)


    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
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



    def test(self, test_games=100):
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
            
            sar = np.zeros([self.env.max_duration, 3,2]) # state, action, reward

            i = 0
            # Changed here from training, since we play games till the end, not for a certain number of steps (frames)
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                # Using item here because they are numpy arrays... stupid
                sar[i] = np.array([state, np.array([action.item(),0]), np.array([reward,0])])

                i += 1

                state = next_state

            all_sar.append(sar)

        return np.array(all_sar)
