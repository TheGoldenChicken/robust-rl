import numpy as np
from typing import Dict, List, Tuple
from sumo.sumo_utils import neighbours, single_dim_interpreter

# TODO: CONSIDER MOVING GET_IDX TO THIS FUNCTION
# TODO: FIX POTENTIALLY BIG PROBLEM WITH REPLAY BUFFER NOT CORRECTLY WORKING FOR FINENESS = MAX, MIN DIM
# Fix get_sample_idxs_from_bin potentially getting no specific action
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

        self.num_actions = num_actions

        self.ripe_bins = [False] * (self.bins - 0*tb) # We don't let the trash_buffer be ripe # We technically don't, but this is easier
        self.num_neighbours = 1
        self.obs_dim = obs_dim

        self.frame_idx = 0 # Debug value
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

    def sample_from_scratch(self, K, nn, specific_action=None, distance='Euclidian', check_ripeness=True):
        """
        Practically the only interface the agent should have with the replay_buffer (apart from dundermethods)
        The whole sampling shebang based on a random point in the current dataset
        :param K: K nearest neighbours to sample based on (basically a batch_size
        :param nn: number-neighbours how many grids to look in larger -> slower + more accurate approximations
        :param specific_action: the action to sample based on, if not chosen, chooses random action
        :param distance: which distance measure to use when getting KNN
        :param check_ripeness: Whether to only sample from bins we know are in a ripe quarter
        :return:
        """

        if specific_action is None:
            specific_action = np.random.randint(0,self.num_actions)

        current_sample = self[self.sample_randomly_idxs(size=1, check_ripeness=check_ripeness)] # Sample to calc KNN from
        current_bin_idx = self.get_bin_idx(current_sample['obs'], single_dim=False) # Idx of bin of current_sample
        neighbour_bin_idxs = self.get_neighbour_bins(current_bin_idx, num_neighbours=nn) # Idx of neighbour bins of current_sample
        samples = self[self.get_sample_idxs_from_bin(neighbour_bin_idxs, action=specific_action)] # Samples from neighbour_bins

        KNN_samples = self.get_knn( current_sample=current_sample, samples=samples, K=K, distance=distance)

        return KNN_samples

    def get_knn(self, current_sample: dict, samples: dict, K: int, distance='Euclidian') -> dict:
        """
        Given a current sample and bunch of other samples, get K-nearest samples from bunch of other samples
        Includes current sample
        :param current_sample: Current sample to calculate distance from
        :param samples: Other samples to find K nearest neighbours from
        :param K: How many neighbours to find
        :param distance: Distance measure to use
        :return:
        """

        if current_sample is None:
            current_sample = np.random.choice(samples['obs'], 1)

        dists = [np.linalg.norm(current_sample, x) for x in samples['obs']]
        idxs = np.argpartition(dists, K)[:K]
        samples = {r: i[idxs] for r, i in samples.items()}

        return samples

    def get_bin_idx(self, s, single_dim=False, test=False):
        """
        Given a location, return the bin in which to place it
        :param s: Some kind of iterable representing the state
        :param single_dim: basically whether or not you use it to store TRUE IF YOU'RE STORIN'
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

        # TODO: REMOVE THIS WHEN DONE TESTING
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

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool,
              idx=None, # Gotta set default value here so PyCharm doesn't shit itself
              ):
        """
        Store new observation
        Idx here refers to which bin to place in (based on current observation location)
        :type idx: int or None
        :param obs:
        :type done: object
        """

        if idx is None:
            idx = self.get_bin_idx(obs, single_dim=True)

        # TODO: Since this is in both store and get_bin_idxs, perhaps move to only have it one of places
        if self.tb and ((obs <= self.state_min).any() or (obs >= self.state_max).any()):
            idx = len(self.size) - 1 # The trash observation goes in the trash can
            teh_idx = idx * self.max_bin_size + self.ptr[idx] # We don't care what action it is...

        else:
            # TODO: Make a lambda out of this?
            idx += act * self.bins_per_action
            teh_idx = idx * self.max_bin_size + self.ptr[idx]
        
        # Trahs_obs for testing
        trash_obs = self.rews_buf[teh_idx] != 0

        self.obs_buf[teh_idx] = obs
        self.next_obs_buf[teh_idx] = next_obs
        self.acts_buf[teh_idx] = act
        self.rews_buf[teh_idx] = rew
        self.done_buf[teh_idx] = done
        self.ptr[idx] = (self.ptr[idx] + 1) % self.max_bin_size
        self.size[idx] = min(self.size[idx] + 1, self.max_bin_size)
        self.frame_idx+= 1

        # Update ripeness - When updated O(1) when not, adds O(1) best case, O(whatever, i'll find out later)
        if not self.ripe_bins[idx]: # Don't want trash buffer things to contribute to ripeness - do we?
            if self.frame_idx >= 1000:
                i = 4

            if self.size[idx] >= self.batch_size:
                self.ripe_bins[idx] = True

            # The single dim interpreter only checks for neighbours to this one bastard
            elif sum([self.size[i + self.bins_per_action * act] for i in
                      self.get_neighbour_bins(single_dim_interpreter
                      (idx - self.bins_per_action * act, self.fineness, self.obs_dim), self.num_neighbours)]) >= self.batch_size:
                self.ripe_bins[idx] = True

        # Only returned for testing purposes
        return trash_obs

    def get_sample_idxs_from_bin(self, idx: list, action: int=0) -> list:
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

    def sample_randomly_idxs(self, size=None, check_ripeness=True):
        """
        Old school sample randomly from all avaliable, no matter grid_location data (or action)
        :param size: how many points to sample
        :return: list of idxs where randomly sampled
        """

        if size is None:
            size=self.batch_size

        # Get all populated indices in 1-d memory array
        # If we check for ripeness, we only go through ripe arrays
        if check_ripeness:
            pop_idxs = [list(range(i*self.max_bin_size, i*self.max_bin_size+self.size[i]))
                    for i in range(self.bins) if self.ripe_bins[i] is True]

        else:
            pop_idxs = [list(range(i*self.max_bin_size, i*self.max_bin_size+self.size[i]))
                    for i in range(self.bins)]
        assert len(pop_idxs) >= size # No point trying to sample if we don't have enough datapoints...
        idxs = np.random.choice(pop_idxs, size)
        return idxs

    # def update_ripeness(self):
    #     """
    #     Used to determine what bins are ripe for sampling
    #     :return:
    #     """
    #
    #     for i, r in enumerate(self.ripe_bins):
    #         if not r: # If unripe, we check
    #             if self.spec_len(i) >= self.batch_size:
    #                 self.ripe_bins[i] = True
    #                 continue
    #
    #             idx_no_action = r % self.bins_per_action
    #             neighs = self.get_neighbour_bins(single_dim_interpreter(idx_no_action,self.fineness, self.obs_dim),
    #                                                  num_neighbours=self.num_neighbours)
    #
    #             if sum([self.spec_len(i) for i in neighs]) >= self.batch_size:
    #                 self.ripe_bins[i] = True

