import numpy as np
from typing import Dict, List, Tuple
from sumo.sumo_utils import neighbours

# TODO: CONSIDER MOVING GET_IDX TO THIS FUNCTION
# TODO: FIX POTENTIALLY BIG PROBLEM WITH REPLAY BUFFER NOT CORRECTLY WORKING FOR FINENESS = MAX, MIN DIM
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

class TheSlightlyCoolerReplayBuffer(ReplayBuffer):

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        super().__init__(obs_dim, size, batch_size)

    def sample_batch(self, all=True, action=None) -> Dict[str, np.ndarray]:
        if not all:
            idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
            return dict(obs=self.obs_buf[idxs],
                        next_obs=self.next_obs_buf[idxs],
                        acts=self.acts_buf[idxs],
                        rews=self.rews_buf[idxs],
                        done=self.done_buf[idxs])

        if action is not None:
            return dict(obs=self.obs_buf[:self.size],
                        next_obs=self.next_obs_buf[:self.size],
                        acts=self.acts_buf[:self.size],
                        rews=self.rews_buf[:self.size],
                        done=self.done_buf[:self.size])

        else:
            # Get the observations with the same action
            same_action = self.acts_buf[:self.size] == action
            return dict(obs=self.obs_buf[same_action],
                        next_obs=self.next_obs_buf[same_action],
                        acts=self.acts_buf[same_action],
                        rews=self.rews_buf[same_action],
                        done=self.done_buf[same_action])


class TheCoolerReplayBuffer(ReplayBuffer):
    # TODO: Make functionality for more actions... You can pretty much just extend the 1-d array
    #  with one extra 1-d array for each action, same length as the whole shebang
    """
    Replay buffer more optimal for grid-based replay-buffering
    """
    def __init__(self, obs_dim, bin_size, batch_size, fineness, num_actions, state_max=np.infty, state_min=-np.infty, tb=True):
        self.bins_per_action = (fineness**obs_dim)
        self.bins = (fineness**obs_dim)*num_actions + 1*tb
        self.size_per_action = (fineness**obs_dim)*bin_size # Max size per action
        self.total_size = bin_size*self.bins
        self.max_bin_size = bin_size
        self.state_max, self.state_min = state_max, state_min # Dim-wise max, min [max_0, max_1], [min_0, min_1]
        self.tb = tb # Trash-buffer, ensures that states out-of-scope are not thrown together with valuable states *in scope*
        self.fineness = fineness

        # Have to call super here so ptr won't get replaced...
        super().__init__(obs_dim, self.total_size, batch_size)
        self.ptr, self.size = [0] * self.bins, [0] * self.bins
        self.ptr, self.size = [0] * self.bins, [0] * self.bins

    def __getitem__(self, idxs) -> Dict[str, np.ndarray]:
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.obs_buf[idxs],
                    rews=self.obs_buf[idxs],
                    done=self.obs_buf[idxs])
    def __len__(self):
        """
        Returns total size across all bins
        """
        return sum(self.size)

    def spec_len(self, idx):
        """
        Return len of a specific grid
        :param idx:
        """
        return self.size[idx]


    def get_bin_idx(self, s, single_dim=False, test=False):
        """
        Given a location, return the bin in which to place it
        :param s: Some kind of iterable representing the state
        :param single_dim: basically whether or not you use it to store
        :return: an int if not single_dim, otherwise a list
        """
        # TODO: MOVE THIS TO A SELF VARIABLE?
        widths = [i / self.fineness for i in self.state_max]
        idxs = [int(s_val // widths[i]) for i, s_val in enumerate(s)]
        if single_dim:
            idxs = sum([r*self.fineness**i for i, r in enumerate(idxs)])

        s = np.array(s)
        if self.tb and ((s <= self.state_min).any() or (s >= self.state_max).any()):
            idxs = len(self.size) - 1 # The trash observation goes in the trash can

        if test:
            widths = [i / self.fineness for i in self.state_max]
            idxs = [int(s_val // widths[i]) for i, s_val in enumerate(s)]
            if single_dim:
                idxs = sum([r*self.fineness**i for i, r in enumerate(idxs)])

            s = np.array(s)
            if self.tb and ((s <= self.state_min).any() or (s >= self.state_max).any()):
                idxs = len(self.size) - 1 # The trash observation goes in the trash can


        return idxs

    def get_neighbour_bins(self, P, num_neighbours):
        """
        Gets all neighbours bins to a certain bin with multi index
        """
        neighbours_multi_index = neighbours(P=P, neighbours=num_neighbours, maxx=self.fineness-1, minn=0)

        neighbours_single_idx = [sum([r*self.fineness**i for i, r in enumerate(current_neighbour)])
                                 for current_neighbour in neighbours_multi_index]

        return neighbours_single_idx
    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        idx: int=0, # Gotta set default value here so PyCharm doesn't shit itself
        ):
        """
        Store new observation
        Idx here refers to which bin to place in (based on current observation location)
        :type idx: int
        :param obs:
        :type done: object
        """


        # TODO: Since this is in both store and get_bin_idxs, perhaps move to only have it one of places
        if self.tb and ((obs <= self.state_min).any() or (obs >= self.state_max).any()):
            idx = len(self.size) - 1 # The trash observation goes in the trash can
            teh_idx = idx * self.max_bin_size + self.ptr[idx] # We don't care what action it is...

        else:
            # TODO: Make a lambda out of this?

            idx += act * self.bins_per_action
            teh_idx = idx * self.max_bin_size + self.ptr[idx]

        trash_obs = self.rews_buf[teh_idx] != 0

        self.obs_buf[teh_idx] = obs
        self.next_obs_buf[teh_idx] = next_obs
        self.acts_buf[teh_idx] = act
        self.rews_buf[teh_idx] = rew
        self.done_buf[teh_idx] = done
        self.ptr[idx] = (self.ptr[idx] + 1) % self.max_bin_size
        self.size[idx] = min(self.size[idx] + 1, self.max_bin_size)

        return trash_obs

    def create_bin_idxs(self, idx: list, action: int=0) -> list:
        """
        Creates indices to grab all datapoints from bin based on indices of bins
        :param idx: List of indices corresponding to which bins to retrieve from
        :param action: Action to grab memory based on, each action grabs indices one self.max_size further ahead
        :return: list of indices to be used in __getitem__
        """

        marker = action*self.size_per_action
        # Don't worry about the list comprehension outside, that's just to flatten the bastard
        idxs = [item for sublist in
                [list(range(i * self.max_bin_size + marker, i * self.max_bin_size + marker + self.size[i])) for i in idx]
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
        pop_idxs = [list(range(i*self.max_bin_size, i*self.max_bin_size+self.size[i]))
                for i in range(self.bins)]
        assert len(pop_idxs) >= size # No point trying to sample if we don't have enough datapoints...
        idxs = np.random.choice(pop_idxs, size)
        return idxs



