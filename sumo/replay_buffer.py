import numpy as np
from typing import Dict, List, Tuple
# TODO: CONSIDER MOVING GET_IDX TO THIS FUNCTION
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
    def __init__(self, obs_dim, bin_size, batch_size, fineness, num_actions, state_max=np.infty, state_min=-np.infty, tb=True):
        self.partitions = (fineness^obs_dim)*num_actions + 1*tb
        self.max_size = bin_size*self.partitions # Max size per action
        self.max_partition_size = bin_size
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

    def spec_len(self, idx):
        """
        Return len of a specific grid
        :param idx:
        """
        return self.size[idx]

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        idx: int=0 # Gotta set default value here so PyCharm doesn't shit itself
        ):
        """
        Store new observation
        Idx here refers to which partition to place in (based on current observation location)
        :type idx: int
        :param obs:
        :type done: object
        """

        if self.tb and ((obs < self.state_min).any() or (obs > self.state_max).any()):
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

        if size is None:
            size=self.batch_size

        # Get all populated indices in 1-d memory array
        pop_idxs = [list(range(i*self.max_partition_size, i*self.max_partition_size+self.size[i]))
                for i in range(self.partitions)]
        assert len(pop_idxs) >= size # No point trying to sample if we don't have enough datapoints...
        idxs = np.random.choice(pop_idxs, size)
        return idxs



