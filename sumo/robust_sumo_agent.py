from sumo_agent import SumoAgent
import torch
import torch.nn.functional as F
import distributionalQLearning4 as distributionalQLearning
import numpy as np
from replay_buffer import TheCoolerReplayBuffer, TheSlightlyCoolerReplayBuffer
from sumo_pp import SumoPPEnv
from typing import Dict, List
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

class RobustSumoAgent(SumoAgent):
    def __init__(self, env, replay_buffer, epsilon_decay, grad_batch_size=30, delta=0.5, max_epsilon=1.0,
                 min_epsilon=0.1, gamma=0.99, model_path=None, robust_factor=1, linear_only=False):
        super().__init__(env, replay_buffer, epsilon_decay, max_epsilon, min_epsilon, gamma, model_path)

        self.robust_factor = robust_factor # Factor multiplied on the robust estimator
        self.grad_batch_size = grad_batch_size # How many samples to compute each loss based off of
        self.delta = delta # For the robust estimator

        # Adding the below as class specific values is kinda stupid, however not having them here would require changing the train function
        self.betas = []
        self.robust_estimators = []
        self.quadratic_approximations = []
        self.linear_only = linear_only

    def get_samples(self) -> tuple[dict, dict]:
        """
        Should be updated for each individual agent type
        """
        samples, current_samples = self.replay_buffer.sample_from_scratch(K=self.replay_buffer.batch_size,
                                                                          nn=None,
                                                                          num_times=self.grad_batch_size,
                                                                          check_ripeness=True)

        return samples, current_samples
    def _compute_dqn_loss(self, samples: List[Dict[str, np.ndarray]],
                          current_samples: List[Dict[str, np.ndarray]] = 0): # -> torch.Tensor: # Default value only so Pycharm doesn't shit itself
        """
        Blablabla compute the loss, this is a good docstring, screw u
        :param samples: Samples to compute robust_estimator by using state, next_state and such
        :param curent_samples: The centre samples to use as reference point wrt. loss between robust estimator and Q(s,a)
        :return:
        """
        # TODO: MAKE ROBUST ESTIMATOR WORK WITH TENSORS...
        device = self.device  # for shortening the following lines

        rewards = torch.FloatTensor(current_samples['rews']).reshape(-1,1).to(device)
        mask = torch.FloatTensor(1 - current_samples['done']).reshape(-1,1).to(device)
        current_sample_obs = torch.FloatTensor(self.state_normalizer(current_samples['obs'])).to(device)
        current_sample_actions = torch.LongTensor(current_samples['acts']).reshape(-1,1).to(device)
        current_q_values = self.dqn(current_sample_obs).gather(1, current_sample_actions)

        robust_estimators = []
        plotting_robust_estimators = [] # For holding output without gamma discounted, reward added robust estimators

        for i, sample in enumerate(samples):
            state = sample["obs"]
            next_state = sample["next_obs"]

            # Normalize state values (very important, trust me!)
            state = self.state_normalizer(state)
            next_state = self.state_normalizer(next_state)

            Q_vals = self.dqn(torch.FloatTensor(next_state).to(device))  # Should all have the same action
            Q_vals = Q_vals.max(dim=1, keepdim=True)[0].detach().cpu().numpy()

            # No reason for calculating robust estimator in the real way if it gets masked anyway
            if mask[i] == 0:
                robust_estimator = 0
                beta_max = np.nan
            else:
                robust_estimator, beta_max = distributionalQLearning.robust_estimator(X_p=state, y_p=next_state,
                                                                                      X_v=next_state, y_v=Q_vals,
                                                                                      delta=self.delta,
                                                                                      linear_only=self.linear_only)
                robust_estimator = robust_estimator

            self.betas.append(beta_max)
            robust_estimators.append(robust_estimator)
            plotting_robust_estimators.append(robust_estimator)

        robust_estimators = torch.FloatTensor(robust_estimators).to(device).reshape(-1, 1) * self.robust_factor # -1 because that works better? #unsqueezing because im stupid and lazy
        robust_estimators = rewards + self.gamma * robust_estimators * mask

        # mask = 1 - done  # Remove effect from those that are done

        # calculate dqn loss
        loss = F.smooth_l1_loss(current_q_values, robust_estimators, reduction='mean') # reduction same as default, no worries

        return loss, robust_estimators, plotting_robust_estimators

    def update_model(self): # -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.get_samples() # Get_samples needs to be set for each subclass

        loss, robust_estimator, _ = self._compute_dqn_loss(*samples)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=2)
        self.optimizer.step()

        return loss.item(), torch.mean(robust_estimator).detach().cpu().numpy()

    def train(self, num_frames: int, plotting_interval: int = 200, q_val_plotting_interval=200):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = [] # For plotting scores
        score = 0 # Current episode score
        robust_estimators = []

        for frame_idx in tqdm(range(1, num_frames + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

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
                loss, robust_estimator = self.update_model()
                losses.append(loss)
                robust_estimators.append(robust_estimator)
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

        print("Training complete")
        return scores, losses, epsilons


if __name__ == "__main__":

    # environment
    env = SumoPPEnv()

    seed = 4949
    # def seed_torch(seed):
    #     torch.manual_seed(seed)
    #     if torch.backends.cudnn.enabled:
    #         torch.backends.cudnn.benchmark = False
    #         torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)
    # seed_torch(seed)

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

    seed_everything(seed)

    obs_dim = env.obs_dim
    action_dim = env.action_dim
    batch_size = 40
    fineness = 100
    ripe_when = None
    state_max, state_min = np.array([env.max_min[0]]), np.array([env.max_min[1]])
    ready_when = 10
    num_neighbours = 2
    bin_size = 1000
    # replay_buffer = TheCoolerReplayBuffer(obs_dim=obs_dim, bin_size=bin_size, batch_size=batch_size, fineness=fineness,
    #                                       num_actions=action_dim, state_max=state_max, state_min=state_min,
    #                                       ripe_when=ripe_when, ready_when=ready_when, num_neighbours=num_neighbours,
    #                                       tb=True)

    replay_buffer = TheSlightlyCoolerReplayBuffer(obs_dim=obs_dim, size=100000, batch_size=batch_size)

    num_frames = 8000

    # parameters
    fineness = 100
    state_dim = 1
    action_dim = 3
    batch_size = 40
    grad_batch_size = 10
    replay_buffer_size = 1000
    max_min = [[env.cliff_position],[0]]
    epsilon_decay = 1/2000
    ripe_when = None # Just batch size
    delta = 0.01 # Should basically be same as DQN with such a small value

    # TODO: PERHAPS JUST PASS A PREMADE REPLAY BUFFER TO THE SUMO AGENT TO AVOID SO MANY PARAMETERS?
    agent = RobustSumoAgent(env=env, replay_buffer=replay_buffer, grad_batch_size=grad_batch_size, delta=delta,
                            epsilon_decay=epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None)
    agent.device = 'cuda'

    agent.train(num_frames)

    states = torch.FloatTensor(np.linspace(0, 1, 1200)).reshape(-1, 1).to(agent.device)
    data = agent.get_q_vals(states)
    column1 = data[:, 0]
    column2 = data[:, 1]
    column3 = data[:, 2]

    # Plot the columns
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(column1)
    plt.title(f'Action 0, {delta}')
    plt.subplot(3, 1, 2)
    plt.plot(column2)
    plt.title('Action 1')
    plt.subplot(3, 1, 3)
    plt.plot(column3)
    plt.title('Action 2')
    plt.tight_layout()
    plt.show()

    scores = agent.test(test_games=100, render_games=5)
    i = 5
    #print(scores, np.mean(scores))
