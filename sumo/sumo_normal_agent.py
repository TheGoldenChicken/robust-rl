import numpy as np
import torch
from sumo import sumo_pp

import os
from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import itertools
from collections import defaultdict

# TODO: MAKE IT SO WILL ONLY CONSIDER ACTIONS THAT ARE OF THE SAME AS CURRENTLY
# SubSymbolic AI? Knowing the effect of action and just calcuting the noise instead of the transition probabilities
# noise = mean(abs(state - next_state) - effect_of_action)
# TODO: FIX CREATE_GRUD_KEYS BY ADDING GET_STATE_DIM AND GET_STATE_MAX_MIN AND GET_STATE_ACTIONS functions
def create_grid_keys(fineness, actions=1):
    """
    Generates positions representing the center of squares used in the grid for NN purposes
    Generates grid with keys equal to finess^state_dim

    returns: List of (state_dim) tuples containing the keys for a grid
    """
    # Should get [[state_dim_1_min,state_dim_1_max], [state_dim_2_min,state_dim_2_max]]
    #max_min = self.env.get_max_min_state()

    # Dummy var - for testing purposes
    max_min = [[-200, 200], [0, 600]]

    all_state_dim_lists = []
    for state_dim in max_min:
        curr_state_dim_list = []
        # From min in state_dim to max in state_dim...
        # With a jump equal to the max distance in each state_dim over the fineness chosen (n squares per dim)
        # Lige overvej Calles integer division her, husk at bruge width!!! //
        width = sum(abs(r) for r in state_dim)/fineness
        for i in range(state_dim[0], state_dim[1], int(sum(abs(r) for r in state_dim)/fineness)):
            curr_state_dim_list.append(i)
        all_state_dim_lists.append(curr_state_dim_list)
    # Sorted sorts in lexiographical order, meaning it sorts for first item first, then second, then third, etc.
    all_combs = sorted(list(itertools.product(*all_state_dim_lists)))

    return np.array(all_combs), np.array(all_state_dim_lists)

class SumoNormalAgent:
    def __init__(self, fineness, obs_dim, replay_size,batch_size, num_actions):
        self.grid_keys, self.grid_list = create_grid_keys(fineness)
        # Dictionary of dictionaries of actions
        # Creates actions * fineness^state_dim different bins
        self.replay_buffer = {r: {i: ReplayBuffer(obs_dim, replay_size, batch_size) for i in self.grid_keys}
                              for r in range(num_actions)}

        self.normal_dists = {i: (0,1) for i in self.replay_buffer.keys()}
    def get_normal_dist(self, action, pos):
        pass
    def fit_value_polynomial(self):
        pass

    def get_closest_grids(self, current_pos, neighbours=0, euclidian=0):
        # Returns [x, y] location of the nearest grid...

        squared_dists = np.sort(np.sum((current_pos - self.grid_keys)**2, axis=1))


        nearest_grid = self.grid_keys[(squared_dists[:neighbours+1])]
        return nearest_grid

# This should work for finding neighbouring grids, but only for 2 dimensions...
    # And, is ugly :((((
# for i in range(-neighbour_grids, neighbour_grids+1 # inclusive,exclusive)
    # for r in range(-neighbour_grids, neighbour_grids+1)
        # next_idx = current_idx + i + r*fineness
        # if next_idx >= 0:
            # if next_idx < len(self.grid_list):
                # all_neighbour_grids.append(next_idx)



    def get_neighbour_grids(self, current_grid, neighbour_grids):
        # Gets neighbouring grid [x, y] locations from current grid [x, y] location

        # Get the index of the current grid_position
        current_grid_idx = np.where(self.grid_list==current_grid)
        neighbour_grid_idxs = []
        for i in

        # Just get all where squared distance to current grid is smaller than given distance

        dim_loc = (self.grid_list[i].index(nearest_grids[i]) for i in self.grid_list)
        max_min_idxs = [(min(0, dim_loc[i] - neighbour_grids), max(len(self.grid_list) - 1, dim_loc[i] + neighbour_grids)) for i indim_loc]

    def get_KNN(self,action, pos, K=0, neighbour_grids=0):
        # Calc squared_dists
        # Return key with lowest squared_dist

        # # Get nearest grid_keys
        # # Then get K nearest neighbours in the r nearest grid_keys from replay_buffer
        # # THEN pass them to the get_normal_dist

        # Get squared_dists to each of the grid_keys
        squared_dists = np.sum((pos - self.grid_keys)**2, axis=1)
        # nearest_grid = self.grid_keys[np.argmin(squared_dists)] # Old code - found only the closest grid location
        nearest_grids = self.grid_keys[np.argpartition(squared_dists, neighbour_grids)]

        # If we don't want to get the K neareset neighbours - and just wanna explore
        # Måske dele op til at have get_nearest_grids
        if K == 0:
            return nearest_grids

        # TODO: Find way to look into r nearest grids in grid_keys (perhaps index grid_keys in a smart way?)
            # Perhaps have grid_keys be a literal grid? With the i'th element being the i // Fineness row and the i % fineness column?
            # Next, get all elements in those grids, get the squared dists and find the elements belonging to the K nearest
            # Pass this to get_normal_dist

        # TODO: Find a not retarded way to do what goes on below
        if nearest_grids != 0:
            # Gets the location of the current grid, useful for getting which partition to look in
            dim_loc_1d = np.where(self.grid_keys == nearest_grids)
            
            dim_loc = (self.grid_list[i].index(nearest_grids[i]) for i in self.grid_list)
            max_min_idxs = [(min(0, dim_loc[i] - neighbour_grids), max(len(self.grid_list)-1, dim_loc[i] + neighbour_grids)) for i in dim_loc]
            all_grid_locations = [self.grid_keys[max_min_idxs[i][0]: max_min_idxs[i][1]] for i in max_min_idxs]
            all_grids_to_search = list(itertools.product(*all_grid_locations))

        # TODO: So what we really want, is getting a single dict with a numpy array for obs, rew, action, next_obs, so on
            # But what we have is a bunch of dicts, each with their own numpy arrays
            # Now, I GUESS that we could just take the numpy arrays from the dicts and concatenate them to a single one of the dicts
            # OR we can remake the replay_buffers to instead of having a replay_buffer for each element in the dict, we have a singel
            # replay_buffer, where the keys just say which pointer range to look at
            # So each grid (assuming rth grid, and i size for replay_buffer), gets pointer space r*i:r*i+i
            # This just begs the question: How do we make tuples link to pointer spaces? A seperate dict?
            # Also, this puts increased presure onn a single object? But removes need for a dict....

        # Dette samler bare indholdet af alle dicts i en
        dd = defaultdict(list)
        for d in (self.replay_buffer[i] for i in all_grids_to_search):
            for key, value in d.items():
                dd[key].append(value)
        # ATTENTION: RIGHT NOW IS JUSTS NORMAL LIST, REMEMBER TO CHANGE TO NUMPY ARRAYS!!!
        # Ignore Pycharm fucking up down here, iz alright
        for key,value in dd.items():
            dd[key] = np.array(dd[key])

        return dd

            # Essentielt dette som de to ovenstående linjer erstatter...
            # x_loc = self.grid_list[0].index(nearest_grids[0])
            # y_loc = self.grid_list[1].index(nearest_grids[1])
            # x_min, x_max = min(0, x_loc-neighbour_grids)
            # y_min, y_max = min(0, y_loc-neighbour_grids)

            all_neighbour_grids = list(itertools.product(*all_state_dim_lists))
        #
        # KNN = list(self.replay_buffer[action].keys())[np.argpartition(squared_dists, K)]
        # return keys[np.argmin(squared_dists)]




class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

class TheCoolerReplayBuffer(ReplayBuffer):
    """
    Replay buffer more optimal for grid-based replay-buffering
    """
    def __init__(self, obs_dim, size, batch_size, fineness):
        super().__init__(obs_dim, size, batch_size)
        self.partitions = fineness^obs_dim
        self.max_size = size*self.partitions
        self.ptr, self.size = [0] * self.partitions, [0] * self.partitions

    def get_from_dim(self, dims):



class RegularDQN:
    def __init__(self, replay_size, state_dim, action_dim, batch_size, env,target_update, epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99):
        self.state_buffer = np.zeros([replay_size, state_dim], dtype=np.float32)
        self.next_state_buffer = np.zeros([replay_size, state_dim], dtype=np.float32)
        self.action_buffer = np.zeros([replay_size], dtype=np.float32)
        self.reward_buffer = np.zeros([replay_size], dtype=np.float32)
        self.done_buffer = np.zeros(replay_size, dtype=np.float32)
        self.max_size, self.batch_size = replay_size, batch_size
        self.pointer, self.size = 0,0

        self.env = env
        self.target_update = target_update # How
        self.epsilon_decay = epsilon_decay
        self.epsilon=max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dqn = Network(state_dim, action_dim).to(self.device)
        self.dqn_target = Network(state_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False


    def store(self, state, action, reward, next_state, done):
        self.state_buffer[self.pointer] = state
        self.next_state_buffer[self.pointer] = next_state
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.done_buffer[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.max_size # Makes the pointer wrap around when max size is reached
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.state_buffer[idxs],
                    next_obs=self.next_state_buffer[idxs],
                    acts=self.action_buffer[idxs],
                    rews=self.reward_buffer[idxs],
                    done=self.done_buffer[idxs])

    def __len__(self) -> int:
        return self.size

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
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
            self.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self) >= self.batch_size:
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

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)

        self.env.close()

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

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




# environment
env_id = "CartPole-v0"
env = gym.make(env_id)

seed = 777

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)
env.seed(seed)

# parameters
num_frames = 10000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000

agent = RegularDQN(env, memory_size, batch_size, target_update, epsilon_decay)

agent.train(num_frames)