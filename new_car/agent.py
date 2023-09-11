
import torch
import torch.nn.functional as F
import distributionalQLearning3 as distributionalQLearning
import numpy as np
from rb import CoolerReplayBuffer
from env import CliffCar
from typing import Dict, List
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

class RobustCliffCarAgent:
    def __init__(self, env, replay_buffer, network,
                 grad_batch_size=30,
                 robust_batch_size = 40,
                 delta=0.5,
                 epsilon_decay = 1/1000,
                 max_epsilon=1.0,
                 min_epsilon=0.1,
                 gamma=0.99,
                 model_path=None,
                 robust_factor=1,
                 linear_only=False,
                 is_train = False,
                 device = "cpu"):

        self.env = env
        self.replay_buffer = replay_buffer
        self.dqn = network(env)

        self.robust_factor = robust_factor # Factor multiplied on the robust estimator
        self.grad_batch_size = grad_batch_size # How many samples to compute each loss based off of
        self.robust_batch_size = robust_batch_size
        self.delta = delta # For the robust estimator
        
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.max_epsilon = min_epsilon
        
        self.gamma = gamma

        # Adding the below as class specific values is kinda stupid, however not having them here would require changing the train function
        self.betas = []
        self.robust_estimators = []
        self.quadratic_approximations = []
        self.linear_only = linear_only
        
        self.device = device
    

    def select_action(self, state):
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = random.randint(0, self.env.ACTION_DIM-1) # Why is this not inclusive, exclusive??? Stupid
        else:
            select_state = self.state_normalizer(state)
            selected_action = self.dqn(
                torch.FloatTensor(select_state).to(self.device)
            ).argmax(axis=1)[0]
            selected_action = int(selected_action)

        return selected_action
    
    def step(self, state, action):
        self.env.position = state
        next_state, reward, done, _ = self.env.step(action)
            
        self.replay_buffer.store({"state":state,"action":action,"next_state":next_state,"reward":reward,"done":done})
        
        return next_state, reward, done
    
    def collect_data(self, n):
        state = self.env.reset()
        for i in range(n):
            action = self.select_action(state)
            
            next_state, _, done = self.step(state, action)
            
            if(done):
                state = self.env.reset()
            else:
                state = next_state
            
        
    
    def _compute_dqn_loss(self):
        """
        Blablabla compute the loss, this is a good docstring, screw u
        :param samples: Samples to compute robust_estimator by using state, next_state and such
        :param curent_samples: The centre samples to use as reference point wrt. loss between robust estimator and Q(s,a)
        :return:
        """
       
        robust_estimators = []
       
        grad_batch = self.replay_buffer.draw(self.grad_batch_size)
        
        if grad_batch is None:
            self.collect_data(self.grad_batch_size)
            return self._compute_dqn_loss()
        
        for grad_sample in grad_batch:
            
            if grad_sample["done"]: continue # Do not sample from done positions
            
            robust_batch = self.replay_buffer.knn(self.robust_batch_size, grad_sample["state"], grad_sample["action"], )
            
            if robust_batch is None:
                self.collect_data(self.robust_batch_size)
                return self._compute_dqn_loss()
            
            states = []
            next_states = []
            rewards = []
            dones = []
            for robust_sample in robust_batch:
                states.append(robust_sample["state"])
                
                # TODO: Add mask for done
                next_state, reward, done = self.step(robust_sample["state"],robust_sample["action"])
                
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                
            Q_vals = self.dqn(torch.FloatTensor(next_states).to(self.device))  # Should all have the same action
            Q_vals = Q_vals.max(dim=1, keepdim=True)[0].detach().cpu().numpy()

            robust_estimator, beta_max = distributionalQLearning.robust_estimator(X_p=np.array(states), y_p=np.array(next_states),
                                                                                    X_v=np.array(next_states), y_v=Q_vals,
                                                                                    delta=self.delta,
                                                                                    linear_only=self.linear_only)
            robust_estimators.append(robust_estimator)

        robust_estimators = torch.FloatTensor(robust_estimators).to(self.device).reshape(-1, 1) * self.robust_factor # -1 because that works better? #unsqueezing because im stupid and lazy
        robust_estimators = rewards + self.gamma * robust_estimators * mask

        # calculate dqn loss
        loss = F.smooth_l1_loss(current_q_values, robust_estimators, reduction='mean') # reduction same as default, no worries

        return loss, robust_estimators, plotting_robust_estimators

    def update_model(self): # -> torch.Tensor:
        """Update the model by gradient descent."""
        
        loss, robust_estimator, _ = self._compute_dqn_loss()

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=2)
        self.optimizer.step()

        return loss.item(), torch.mean(robust_estimator).detach().cpu().numpy()

    def train(self, num_frames: int, plotting_interval: int = 200, q_val_plotting_interval=999999):
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
            next_state, reward, done = self.step(state, action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

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
    env = CliffCar()

    seed = 4949

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
    state_max, state_min = np.array(env.max_min[0]), np.array(env.max_min[1]) # Here we don't listify it, contrary to the sumo environmnet, gotta fix that in replay buffer
    ready_when = 10
    num_neighbours = 2
    bin_size = 1000
    replay_buffer = TheCoolerReplayBuffer(obs_dim=obs_dim, bin_size=bin_size, batch_size=batch_size, fineness=fineness,
                                          num_actions=action_dim, state_max=state_max, state_min=state_min,
                                          ripe_when=ripe_when, ready_when=ready_when, num_neighbours=num_neighbours,
                                          tb=True)

    # replay_buffer = TheSlightlyCoolerReplayBuffer(obs_dim=obs_dim, size=100000, batch_size=batch_size)

    num_frames = 2000

    # parameters
    fineness = 100
    state_dim = 2
    action_dim = 5
    batch_size = 40
    grad_batch_size = 10
    replay_buffer_size = 1000
    max_min = env.max_min
    # max_min = [[env.cliff_height,],[0]]
    epsilon_decay = 1/2000
    ripe_when = None # Just batch size
    delta = 0.01 # Should basically be same as DQN with such a small value

    # TODO: PERHAPS JUST PASS A PREMADE REPLAY BUFFER TO THE SUMO AGENT TO AVOID SO MANY PARAMETERS?
    agent = RobustCliffCarAgent(env=env, replay_buffer=replay_buffer, grad_batch_size=grad_batch_size, delta=delta,
                            epsilon_decay=epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None)
    agent.device = 'cuda'

    agent.train(num_frames)

    X,Y = np.mgrid[0:1:150j, 0:1:150j]

    xy = np.vstack((X.flatten(), Y.flatten())).T

    states = torch.FloatTensor(xy).to(agent.device)
    # states1 = torch.FloatTensor(np.linspace(0, 1, 1200)).reshape(-1, 1).to(agent.device)

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
