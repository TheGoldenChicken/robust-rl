import os
from typing import Dict, List, Tuple
from sumo import sumo_pp
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from replay_buffer import ReplayBuffer, TheSlightlyCoolerReplayBuffer
import random
from sumo.sumo_pp import SumoPPEnv
import distributionalQLearning2 as distributionalQLearning
from network import Network


class SumoSlowAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        obs_dim = 1
        action_dim = 3

        # obs_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n

        self.env = env
        self.memory = TheSlightlyCoolerReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = 1 # Only need for important sample
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.action_dim = action_dim

        # device: cpu / gpu
        self.device = torch.device(
            "cpu" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = random.randint(0, self.action_dim-1)
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        k = 100
        important_sample = self.memory.sample_batch(all=False)
        samples = self.memory.sample_batch(action=important_sample['acts'])
        indices = self.get_KNN(important_sample['obs'], samples['obs'], K=k)[:k]
        samples = {i: r[indices] for i, r in samples.items()}

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_KNN(self, s, samples, K):
        """
        Get KNN and return indices of the K-nearest neighbours
        """
        dists = [np.linalg.norm(s - i) for i in samples]
        K_neighbours = np.argpartition(dists, K)
        return K_neighbours


    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            reward = reward * -1
            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                done = False # Perhaps this should be done in the step function?
                scores.append(score)
                score = 0
                current_episode_score = 0

            # if training is ready
            if len(self.memory) >= 500:
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

        # self.env.close()

    # def test(self, video_folder: str="derp"):
    #     """Test the agent."""
    #     self.is_test = True
    #
    #     # for recording a video
    #     naive_env = self.env
    #
    #     state = self.env.reset()
    #     done = False
    #     score = 0
    #
    #     while not done:
    #         action = self.select_action(state)
    #         next_state, reward, done = self.step(action)
    #
    #         state = next_state
    #         score += reward
    #
    #     print("score: ", score)
    #     # self.env.close()
    #     return score
    #     # reset

    def test(self, test_games=100, render_games: int=0, render_speed: int=60):
        self.is_test = True # Prevent from taking random actions
        state = self.env.reset()
        scores = []
        done = False

        for i in range(test_games):
            score = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            scores.append(score)

        self.env.init_render()
        state = self.env.reset()
        self.env.frame_rate = 60
        for i in range(render_games):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                self.env.render()

                state = next_state

        return scores



    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        done = done.detach().cpu().numpy()

        # reward = reward.detach().cpu().numpy()
        # TODO: CHECK IF V(S) ESTIMATE IS BASED ON CURRENT OR NEXT STATE
        Q_vals = self.dqn(state) # Should all have the same action
        curr_q_value = Q_vals.gather(1, action)
        Q_vals = Q_vals.max(dim=1, keepdim=True)[0].detach().cpu().numpy()

        robust_estimator = distributionalQLearning.robust_estimator(X_p=samples['obs'], y_p=samples['next_obs'],
                                                                    X_v=samples['next_obs'], y_v=Q_vals,
                                                                    delta=0.5)

        mask = 1 - done # Remove effect from those that are done
        robust_estimator = reward + self.gamma * robust_estimator * mask


        # calculate dqn loss #TODO: CHECK WHY WE USE SMOOTH_L1_LOSS
        loss = F.smooth_l1_loss(curr_q_value, robust_estimator)

        return loss

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


if __name__ == "__main__":

    # environment
    env = sumo_pp.SumoPPEnv(line_length=500)

    seed = 777


    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


    np.random.seed(seed)
    seed_torch(seed)

    num_frames = 8000

    # parameters
    fineness = 100
    state_dim = 1
    action_dim = 3
    batch_size = 64
    replay_buffer_size = 500
    max_min = [[500],[0]]
    epsilon_decay = 1/2000

    agent = SumoSlowAgent(env=env, memory_size=100000, batch_size=batch_size, epsilon_decay=epsilon_decay)

    agent.train(num_frames)
    scores = agent.test(test_games=10, render_games=20)
