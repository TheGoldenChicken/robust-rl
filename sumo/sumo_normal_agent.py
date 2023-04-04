# TODO: WE HAVE A PROBLEM!
# TODO: ACTIONS!!! CAN COME AS NUMPY ARRAYS??? WTFFFF
# TODO: MAKE THE WHOLE THING WORK IN CASE ACTIONS COME AS NUMPY ARRAYS AND NOT INTEGERS, THAT WOULD BE STUPIDDDD
# TODO GRIDKEYS: Some way of adding variable fineness to the different grid keys
    # Must be done in create_grid_keys
    # Must change mult_dim og single_dim intepreter til at passe med forskellige fineness
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
# TODO: Det der skal lægges til INdex af action skal ændres til at være sum(actions) hvis actions er en liste

# SubSymbolic AI? Knowing the effect of action and just calculating the noise instead of the transition probabilities
def create_grid_keys(fineness, max_min, actions=1):
    """
    :param fineness: Number of grid points per dimension
    :param max_min: list of dim lists where each list contains [state_min, state_max] valueus
    :param actions: # TODO SEE IF YOU CAN'T REMOVE THIS
    :param tb# TODO: REMOVE THIS (trash-border): Meant to create a single encompassing border that lumps all out-of-boundary observations into it, agent will have shit data here, but it shouldn't be here anyways
    :return: all grid key combinations (numpy array), numpy array containing dim lists of fineness size with all grid locations along each dimension

    Generates positions representing the center of squares used in the grid for NN purposes
    Generates grid with keys equal to finess^state_dim

    returns: List of (state_dim) tuples containing the keys for a grid
    """
    # Should get [[state_dim_1_min,state_dim_1_max], [state_dim_2_min,state_dim_2_max]]
    # Dummy var - for testing purposes

    max_min = [[-200, 200], [0, 600]]

    all_state_dim_lists = []
    for state_dim in max_min:
        curr_state_dim_list = []

        # From min in state_dim to max in state_dim...
        # With a jump equal to the max distance in each state_dim over the fineness chosen (n squares per dim)
        # Lige overvej Calles integer division her, husk at bruge width!!! //
        width = int(sum(abs(r) for r in state_dim)/fineness)
        for i in range(state_dim[0], state_dim[1], width):
            curr_state_dim_list.append(i)
        all_state_dim_lists.append(curr_state_dim_list)

    # Sorted sorts in lexiographical order, meaning it sorts for first item first, then second, then third, etc.
    all_combs = sorted(list(itertools.product(*all_state_dim_lists)))

    return np.array(all_combs), np.array(all_state_dim_lists)

class SumoNormalAgent:
    def __init__(self, fineness, obs_dim, replay_size, batch_size, num_actions):
        self.fineness = fineness
        self.grid_keys, self.grid_list = create_grid_keys(fineness)
        self.state_min, self.state_max = [-200, 200], [0, 600] # Something like this
        # self.state_min, self.state_max = self.env.get_max_min()

        #self.replay_buffer = TheCoolerReplayBuffer()


    def get_closest_grids(self, current_pos, neighbours=0, euclidian=0):
        # TODO Implement actual euclidian, not this cursed squared distance... doesn't scale well, as you probably know
        # Returns [x, y] location of the nearest grid...

        squared_dists = np.sum((current_pos - self.grid_keys)**2, axis=1)
        #squared_dists = np.sort(np.sum((current_pos - self.grid_keys)**2, axis=1))

        nearest_grid = self.grid_keys[np.argmin(squared_dists)]
        #nearest_grid = self.grid_keys[(squared_dists[:neighbours+1])]
        return nearest_grid


    def get_neighbour_grids(self, current_grid, neighbour_grids):
        # Gets neighbouring grid [x, y] locations from current grid [x, y] location
        # Index, not coordinates!

        # Get the index of the current grid_position
        grid_index = np.where(self.grid_list == current_grid)
        dims = len(self.state_max)

        # TODO: ATTACH FINENESS TO THE CLASS ITSELF
        # TODO: FIND THE WAY TO GET SELF.DIM
        multi_index = single_dim_interpreter(grid_index, fineness=self.fineness, dim=dims)
        neighbours_multi_index = neighbours(P=multi_index, neighbours=neighbour_grids,
                                            maxx=self.fineness-1, minn=0)
        neighbours_single_index = [multi_dim_interpreter(i, fineness=self.fineness, dim=None)
                                   for i in neighbours_multi_index]

        return neighbours_single_index


    def get_KNN(self, pos, action, K=0, neighbour_grids=0):
        """
        Get K-nearest neighbours for a given pos based on state
        :param pos: Current pos (state) to look based on
        :param action: Only look up places in the replay buffer that have an action corresponding to current action
        :param K: K neighbours to find
        :param neighbour_grids: Number of neighbouring grids next to current to look at
        :return:
        """
        # KNN = list(self.replay_buffer[action].keys())[np.argpartition(squared_dists, K)]
        # return keys[np.argmin(squared_dists)]
        # TODO: Make work for multiple actions

        closest_grid = self.get_closest_grids(pos, neighbours=0)
        idxs_to_look = self.get_neighbour_grids(closest_grid, neighbour_grids=neighbour_grids)

        all_points = self.replay_buffer[self.replay_buffer.create_partition_idxs(idxs_to_look, action)]

        all_points_of_interest = all_points['obs']
        # Finally, NN will be KNN among the all_points_of_interest, and we can use these for the purposes of sampling...
        # TODO: look into FAISS for getting KNN among all_points_of_interest

def single_dim_interpreter(idx, fineness, dim):
    """
    Takes an idx that specifies a location on a line, and projects it to dim dimensions depending on estabilshed fineness
    :param idx: Single-dimensional index to interpret to multi-dimensional coordinates
    :param fineness: How many 'grids' are per dimension (row)
    :param dim: Number of dimensions in the target coordinates
    :return: tuple of dim elements corresponding to idx being projected in dim-dimensional space
    """
    # Takes single dim idx that we know codes for a flattened list of dim dimensions
    # With fineness elements per dimension row

    interpreted = [0] * dim
    # Reverse iteration
    for i in range(dim-1,-1, -1):

        current_dim_size = idx // fineness**i
        interpreted[i] = current_dim_size
        idx = idx - current_dim_size * fineness**i

    return interpreted

def multi_dim_interpreter(idxs, fineness, dim=None):
    """
    Takes a multi-dimensional coordinate and places it along a line
    :param idxs: tuple of multi-dimensional coordinates
    :param fineness: number of 'grids' per dimension
    :param dim: inferred, not needed
    :return: single-dimensional coordinate representing idxs along a line
    """

    # Takes a multi dimensional coordinate, transforms it to single dim flattened
    # Technically don't need dim, since this is inferred from len(idxs)

    idx = 0
    for r, i in enumerate(idxs):
        idx += i * fineness**r
    return idx

import random

# Courtesy of https://stackoverflow.com/questions/40292190/calculate-the-neighbours-of-n-dimensional-fields-in-python
def stencil(dim, neighbours, remove_zero=False) -> list:
    """
    Helper function for neighbours function, creates a kind of 'movement grid' stencil for navigating around multi-dim
    :param dim: dimensionality of hypercube to create move stencils for:
    :param neighbours: number of points to out along each dimension:
    :param remove_zero: if we need to remove the starting point (no zero-stencil):
    :return: list: list of tuples that correspond to all ways of combining -neighbours:neighbours moves away from some origin
    """
    stencils = list(itertools.product(list(range(-neighbours, neighbours+1)), repeat=dim))
    if remove_zero:
        zero = ((0,) * dim)
        stencils.remove(zero)
    return stencils

def neighbours(P: tuple, neighbours: int=0, maxx=np.infty, minn=-np.infty) -> list:
    """

    :param P: Point in n-dimensional cube:
    :param neighbours: Number points out along all axis combinatinons to get neighbours:
    :param maxx: Remove neighbours with coordaintes above this value (should be fineness-1)
    :param minn: Remove neighbours with coordiantes below this value (should be 0)
    :return: list of tuples that contain all neighbours
    """
    stencils = stencil(len(P), neighbours)
    positions = [tuple([sum(x) for x in zip(P, s)]) for s in stencils]

    # Remove all smaller than maxx or minn (why the stupid spelling - Python fucks ur ass otherwise)
    if maxx != None:
        positions = [i for i in positions if max(i) <= maxx]
    if minn != None:
        positions = [i for i in positions if min(i) >= minn]
    return positions

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
    # TODO: Make functionality for more actions... You can pretty much just extend the 1-d array
    #  with one extra 1-d array for each action, same length as the whole shebang
    """
    Replay buffer more optimal for grid-based replay-buffering
    """
    def __init__(self, obs_dim, size, batch_size, fineness, num_actions, state_max=np.infty, state_min=-np.infty, tb=True):
        self.partitions = (fineness^obs_dim)*num_actions + 1*tb
        self.max_size = size*self.partitions # Max size per action
        self.max_partition_size = size
        self.state_max, self.state_min = state_max, state_min
        self.tb = tb # Trash-buffer, ensures that states out-of-scope are not thrown together with valuable states *in scope*

        # Have to call super here so ptr won't get replaced...
        super().__init__(obs_dim, self.max_size, batch_size)
        self.ptr, self.size = [0] * self.partitions, [0] * self.partitions

    def __getitem__(self, idxs) -> Dict[str, np.ndarray]:
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.obs_buf[idxs],
                    rews=self.obs_buf[idxs],
                    done=self.obs_buf[idxs])
    def __len__(self):
        """
        Returns total size across all partitions
        """
        return sum(self.size)

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        idx: int
        ):
        """
        Store new observation
        Idx here refers to which partition to place in (based on current observation location)
        :type idx: int
        :param obs:
        :type done: object
        """

        if self.tb and ((obs < self.state_min).any() and (obs > self.state_max).any()):
            idx = len(self.size) - 1 # The trash observation goes in the trash can

        # todo: remove shitty solution here, since right now, we only consider action arrays of single elements
        action = act.tolist().index(1)
        # idx to store at, based
        # TODO: Make a lambda out of this?
        teh_idx = idx * self.max_partition_size + action * self.max_size + self.ptr[idx]

        self.obs_buf[teh_idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.acts_buf[idx] = act
        self.rews_buf[idx] = rew
        self.done_buf[idx] = done
        self.ptr[idx] = (self.ptr[idx] + 1) % self.max_partition_size
        self.size[idx] = min(self.size[idx] + 1, self.max_partition_size)


    def create_partition_idxs(self, idx: list, action: int=0) -> list:
        """
        Creates indices to grab from memory based on partition indices
        :param idx: List of indices corresponding to which parititons to retrieve from
        :param action: Action to grab memory based on, each action grabs indices one self.max_size further ahead
        :return: list of indices to be used in __getitem__
        """

        marker = self.max_partition_size + action*self.max_size
        # Don't worry about the list comprehension outside, that's just to flatten the bastard
        idxs = [item for sublist in [list(range(i * marker, i * marker + self.size[idx])) for i in idx]
                for item in sublist]
        return idxs


    def sample_randomly_idxs(self, size=None):
        """
        Old school sample randomly from all avaliable, no matter grid_location data (or action)
        :param size: how many points to sample
        :return: list of idxs where randomly sampled
        """

        if size==None:
            size=self.batch_size

        # Get all populated indices in 1-d memory array
        pop_idxs = [list(range(i*self.max_partition_size, i*self.max_partition_size+self.size[i]))
                for i in range(self.partitions)]
        assert len(pop_idxs) >= size # No point trying to sample if we don't have enough datapoints...
        idxs = np.random.choice(pop_idxs, size)
        return idxs



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