import numpy as np
from typing import Dict, List, Tuple
from utils import neighbours, single_dim_interpreter

# TODO: CONSIDER MOVING GET_IDX TO THIS FUNCTION
# TODO: FIX POTENTIALLY BIG PROBLEM WITH REPLAY BUFFER NOT CORRECTLY WORKING FOR FINENESS = MAX, MIN DIM
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, ready_when=500):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.int32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.ready_when = ready_when

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

    def training_ready(self) -> bool:
        return self.size >= self.ready_when

class TheSlightlyCoolerReplayBuffer(ReplayBuffer):

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        super().__init__(obs_dim, size, batch_size)
        self.noise_adder = True # Since this is used for robust agent, we must also add noise

    def __getitem__(self, idxs) -> Dict[str, np.ndarray]:
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):

        if self.noise_adder:
            obs += np.random.normal(loc=0,scale=1e-3, size=len(obs))
            next_obs += np.random.normal(loc=0,scale=1e-3, size=len(next_obs))

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_from_scratch(self, K, nn, num_times=1, specific_action=None, distance='Euclidian', check_ripeness=True):
        """
        Basically like sample_from_scratch for TheCoolerReplayBuffer
        But this is probably pretty slow
        This should be pretty FACKING SLOW
        """
        # TODO
        # I know this is stupid, will fix later, same for TCRP
        KNN_sampless = []
        # Should replace be false here?

        current_idxs = np.random.choice(self.size, size=num_times, replace=False)
        current_samples = self[current_idxs] # Samples to calc KNN from

        for i, r in enumerate(current_samples['obs']):
            dists = [np.linalg.norm(r - x) if self.acts_buf[p] == self.acts_buf[i] else np.infty for p, x in enumerate(self.obs_buf[:self.size])]
            K = min(K, len(dists))
            idxs = np.argpartition(dists, K)[:K]
            KNN_sampless.append(self[idxs])

        return KNN_sampless, current_samples

    def training_ready(self) -> bool:
        return self.size >= self.ready_when

class TheCoolerReplayBuffer(ReplayBuffer):
    """
    Replay buffer more optimal for grid-based replay-buffering
    """
    def __init__(self, obs_dim, bin_size, batch_size, fineness, num_actions, state_max=np.infty,
                 state_min=-np.infty, tb=True, ripe_when=None, ready_when=10, num_neighbours=2, noise_adder = True):
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
        self.num_neighbours = num_neighbours
        self.obs_dim = obs_dim

        if ripe_when is None:
            self.ripe_when = self.batch_size + 1 # Value to decide when a buffer is ripe or not # Fixes dumb issue with argpartition (the +1 that is)
        else:
            self.ripe_when = ripe_when
        self.ready_when = ready_when # Number of ripe bins before we can start training

        self.noise_adder = noise_adder

    def __getitem__(self, idxs) -> Dict[str, np.ndarray]:
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
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

    def sample_from_scratch(self, K, nn=None, num_times=1, specific_action=None, distance='Euclidian', check_ripeness=True):
        """
        Practically the only interface the agent should have with the replay_buffer (apart from dundermethods)
        The whole sampling shebang based on a random point in the current dataset
        :param K: K nearest neighbours to sample based on (basically a batch_size
        :param nn: number-neighbours how many grids to look in larger -> slower + more accurate approximations
        :param num_times: number of times we repeat sampling procedure for different points (meant for batch learning)
        :param specific_action: the action to sample based on, if not chosen, chooses random action
        :param distance: which distance measure to use when getting KNN
        :param check_ripeness: Whether to only sample from bins we know are in a ripe quarter
        :return: KNN samples (list of dicts) the actual samples to be used, current_samples, the reference sample for...
        each point, used for specifying the reward used in computing robust estimator
        """
        # TODO: FIX NOT BEING ABLE TO SPECIFY SPECIFIC_ACTION - WE CANNOT GUARANTEE RIPE REPLAY BUFFERS ON THAT ACTION...

        if nn is None:
            nn = self.num_neighbours

        KNN_sampless = []
        current_samples = self[self.sample_randomly_idxs(size=num_times, check_ripeness=check_ripeness)] # Samples to calc KNN from

        for i, r in enumerate(current_samples['obs']):

            current_action = current_samples['acts'][i]
            current_bin_idx = self.get_bin_idx(current_samples['obs'][i], single_dim=False) # Idx of bin of current_sample
            neighbour_bin_idxs = self.get_neighbour_bins(P=current_bin_idx, num_neighbours=nn) # Idx of neighbour bins of current_sample
            samples = self[self.get_sample_idxs_from_bin(neighbour_bin_idxs, action=current_action)] # Samples from neighbour_bins

            KNN_samples = self.get_knn(current_obs=current_samples['obs'][i], samples=samples, K=K, distance=distance)

            KNN_sampless.append(KNN_samples)

        return KNN_sampless, current_samples

    def get_knn(self, current_obs: np.ndarray, samples: dict, K: int, distance='Euclidian', return_dict=True) -> dict:
        """
        Given a current sample and bunch of other samples, get K-nearest samples from bunch of other samples
        Includes current sample
        :param current_obs: Current obs to calculate distance from
        :param samples: Other samples to find K nearest neighbours from
        :param K: How many neighbours to find
        :param distance: Distance measure to use
        :param return_dict: Whether or not to reutrn a dictionary or just the samples for appending to a list and size guide (not used)
        :return:
        """

        if current_obs is None:
            current_sample = np.random.choice(samples['obs'], 1)

        dists = [np.linalg.norm(current_obs - x) for x in samples['obs']]
        K = min(K, len(samples['obs']))

        # TODO: Debug this so we don't need the min(k, len(samples['obs']) above
        # if K >= len(samples['obs']):
        #     i = 2
        try:
            idxs = np.argpartition(dists, K)[:K]
        except:
            idxs = np.argpartition(dists, min(K, len(dists) - 1))[:min(K, len(dists) - 1)]

        samples = {r: i[idxs] for r, i in samples.items()}

        return samples

    def get_bin_idx(self, s, single_dim=False, test=False):
        """
        Given a location, return the bin in which to place it
        :param s: Some kind of iterable representing the state
        :param single_dim: basically whether or not you use it to store TRUE IF YOU'RE STORIN'
        :return: an int if not single_dim, otherwise a list
        """
        widths = [i / self.fineness for i in self.state_max]
        idxs = [int(s_val // widths[i]) for i, s_val in enumerate(s)]
        if single_dim:
            idxs = sum([r*self.fineness**i for i, r in enumerate(idxs)])

        # Should only really go here if we're storing
        if self.tb and sum((s < self.state_min) + (s > self.state_max)):
            # TODO: Fix problem where this becomes a integer which fails later. Current fix is to clamp state and run again
            s = np.clip(s, self.state_min, self.state_max)
            idxs = self.get_bin_idx(s, single_dim, test)
            # idxs = len(self.size) - 1 # The trash observation goes in the trash can

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
        if self.tb and ((obs < self.state_min).any() or (obs > self.state_max).any()):
            idx = len(self.size) - 1 # The trash observation goes in the trash can
            teh_idx = idx * self.max_bin_size + self.ptr[idx] # We don't care what action it is...

        else:
            # TODO: Make a lambda out of this?
            idx += act * self.bins_per_action
            teh_idx = idx * self.max_bin_size + self.ptr[idx]
        
        # Trahs_obs for testing
        trash_obs = self.rews_buf[teh_idx] != 0

        if self.noise_adder:
            # Stupid stuff to prevent singular matrices from appearing in calculation of the robust estimator
            obs += np.random.normal(loc=0,scale=5e-4, size=self.obs_dim)
            next_obs += np.random.normal(loc=0,scale=5e-4, size=self.obs_dim)

        self.obs_buf[teh_idx] = obs
        self.next_obs_buf[teh_idx] = next_obs
        self.acts_buf[teh_idx] = act
        self.rews_buf[teh_idx] = rew
        self.done_buf[teh_idx] = done
        self.ptr[idx] = (self.ptr[idx] + 1) % self.max_bin_size
        self.size[idx] = min(self.size[idx] + 1, self.max_bin_size)

        # Update ripeness - When updated O(1) when not, adds O(1) best case, O(whatever, i'll find out later)
        if not self.ripe_bins[idx]: # Don't want trash buffer things to contribute to ripeness - do we?

            # DEBUG
            if self.size[idx] >= self.batch_size:
                self.ripe_bins[idx] = True

            # The single dim interpreter only checks for neighbours to this one bastard
            elif sum([self.size[i + self.bins_per_action * act] for i in
                      self.get_neighbour_bins(single_dim_interpreter
                      (idx - self.bins_per_action * act, self.fineness, self.obs_dim), self.num_neighbours)]) >= self.ripe_when:
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
                [list(range(i * self.max_bin_size + marker, i * self.max_bin_size + marker + self.size[i+action*self.bins_per_action])) for i in idx]
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
            # Just to flatten the fucker first - The whole item/sublist thingy... is it faster? Who knows
            poppable_idxs = [item for sublist in [list(range(i*self.max_bin_size, i*self.max_bin_size+self.size[i]))
                    for i in range(self.bins-self.tb) if self.ripe_bins[i] is True] for item in sublist]

        else:
            poppable_idxs = [item for sublist in [list(range(i*self.max_bin_size, i*self.max_bin_size+self.size[i]))
                    for i in range(self.bins-self.tb)] for item in sublist]
        assert len(poppable_idxs) >= size # No point trying to sample if we don't have enough datapoints...
        idxs = np.random.choice(poppable_idxs, size)
        return idxs

    def training_ready(self) -> bool:
        return sum(self.ripe_bins) >= self.ready_when



#
# # Old stuff used to be used in slow replay buffer
#     def sample_batch(self, all=True, action=None) -> Dict[str, np.ndarray]:
#         if not all:
#             idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
#             return dict(obs=self.obs_buf[idxs],
#                         next_obs=self.next_obs_buf[idxs],
#                         acts=self.acts_buf[idxs],
#                         rews=self.rews_buf[idxs],
#                         done=self.done_buf[idxs])
#
#         if action is not None:
#             return dict(obs=self.obs_buf[:self.size],
#                         next_obs=self.next_obs_buf[:self.size],
#                         acts=self.acts_buf[:self.size],
#                         rews=self.rews_buf[:self.size],
#                         done=self.done_buf[:self.size])
#
#         else:
#             # Get the observations with the same action
#             same_action = self.acts_buf[:self.size] == action
#             return dict(obs=self.obs_buf[same_action],
#                         next_obs=self.next_obs_buf[same_action],
#                         acts=self.acts_buf[same_action],
#                         rews=self.rews_buf[same_action],
#                         done=self.done_buf[same_action])
