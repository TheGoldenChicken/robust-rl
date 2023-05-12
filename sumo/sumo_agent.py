# TODO: WE HAVE A PROBLEM!
# TODO: ACTIONS!!! CAN COME AS NUMPY ARRAYS??? WTFFFF
# TODO: MAKE THE WHOLE THING WORK IN CASE ACTIONS COME AS NUMPY ARRAYS AND NOT INTEGERS, THAT WOULD BE STUPIDDDD
# TODO GRIDKEYS: Some way of adding variable fineness to the different grid keys
# TODO: IMPORTANT: Find out if all types (tuples, np arrays, such)... are correct!
# Must be done in create_grid_keys
# Must change mult_dim og single_dim intepreter til at passe med forskellige fineness
# TODO: Det der skal lægges til INdex af action skal ændres til at være sum(actions) hvis actions er en liste
# TODO: CONSIDER ADDING NUM_NIEGHBOURS (CHECK HOW MANY GRIDS) TO AGENT

# from sumo import sumo_pp
from sumo import sumo_pp
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sumo_utils import normalize_tensor
from replay_buffer import TheCoolerReplayBuffer
import random
import distributionalQLearning2 as distributionalQLearning
from IPython.display import clear_output
from network import Network
from tqdm import tqdm

# SubSymbolic AI? Knowing the effect of action and just calculating the noise instead of the transition probabilities

# finenesss = used for replay buffer
# state_dim = used for replay and network
# action_dim = used for select action
# Batch_size = used for replay_buffer
# replay_buffer_size = used for replay_buffer
# self.max, self.min can be moved to the sumo environment
# ripe_when, moved to replay buffer
# number_neighbours - move to replay buffer (default 2)

class SumoAgent:
    def __init__(self, env, replay_buffer, epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None):
        self.env = env
        self.max, self.min = np.array(self.env.max_min[0]), np.array(self.env.max_min[1])

        # Learning and exploration parameters
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.gamma = gamma

        self.replay_buffer = replay_buffer
        self.training_ready = False

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device) # Just to know which one we're on

        self.dqn = Network(env.obs_dim, env.action_dim).to(self.device)
        if model_path is not None:
            self.dqn.load_state_dict(model_path)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # Should work for tensors and numpy...
        self.state_normalizer = lambda state: (state - self.min)/(self.max - self.min)

    def get_samples(self) -> tuple[dict, ]:
        """
        Should be updated for each individual agent type
        returns: tuple[samples,current_samples] current_samples only if robust agent, samples is list in this case
        """
        raise(NotImplementedError)
        return samples

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Needs to be implemented for each agent"""
        raise(NotImplementedError)
        return loss


    def select_action(self, state: np.ndarray, ) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = random.randint(0, self.env.action_dim-1) # Why is this not inclusive, exclusive??? Stupid
        else:
            select_state = self.state_normalizer(state)
            selected_action = self.dqn(
                torch.FloatTensor(select_state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        # Store transitions
        if not self.is_test:
            # Store current transition
            self.transition += [reward, next_state, done]
            self.replay_buffer.store(*self.transition)

        return next_state, reward, done

    def update_model(self): # -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.get_samples() # Get_samples needs to be set for each subclass

        loss = self._compute_dqn_loss(*samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200, q_val_plotting_interval=200):
        """Train the agent."""
        self.is_test = False
        self.dqn.train()
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        current_episode_score = 0

        for frame_idx in tqdm(range(1, num_frames + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            current_episode_score += reward

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # Update whether we're ready to train
            if not self.training_ready:
                self.training_ready = self.replay_buffer.training_ready()

            else:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)


            # plotting
                if frame_idx % plotting_interval == 0:
                    self._plot(frame_idx, scores, losses, epsilons)
                    # print(frame_idx, loss, self.epsilon, )

                if frame_idx % q_val_plotting_interval == 0:
                    self._plot_q_vals()

        print("Training complete")
        return scores, losses, epsilons

    def save_model(self, model_path):
        try:
            torch.save(self.dqn.state_dict(), model_path)
        except:
            print("ERROR! Could not save model!")

    def test(self, test_games=100, render_games: int=0, render_speed: int=60):
        """
        Test the agent
        :param test_games: number of test games to get score from
        :param render_games: number of rendered games to inspect qualitatively
        :param render_speed: frame_rate of the rendered games
        :return:
        """
        self.is_test = True # Prevent from taking random actions
        all_sar = [] # All state_action_reward
        self.dqn.eval()

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

                sar[i] = (state.item(), action.item(), reward)

                i += 1

                state = next_state

            all_sar.append(sar)

        # If statement necessary, otherwise Pygame opens and stays loitering around
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

    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))


    def get_q_vals(self, states):
        return self.dqn(states).detach().cpu().numpy()


    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()

    def _plot_q_vals(self):
        states = torch.FloatTensor(np.linspace(0, 1, 1000)).reshape(-1, 1).to(self.device)
        data = self.get_q_vals(states)
        action0 = data[:, 0]
        action1 = data[:, 1]
        action2 = data[:, 2]

        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('Action 0')
        plt.plot(action0)
        plt.subplot(132)
        plt.title('Action 1')
        plt.plot(action1)
        plt.subplot(133)
        plt.title('Action 2')
        plt.plot(action2)
        plt.show()
