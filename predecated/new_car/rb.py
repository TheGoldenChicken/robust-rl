import numpy as np
from collections import defaultdict, deque
import itertools
import torch

class Bin:
    
    def __init__(self, size):
        self.size = size
        self.samples = torch.empty(size, dtype=object)
        self.head_ptr = -1
        self.is_full = False
        
    def store(self, sample):
        self.head_ptr += 1
        self.samples[self.head_ptr] = sample
        
        # If the head ptr is at the end list then 
        if (self.head_ptr == self.size - 1):
            self.head_ptr = 0
            self.is_full = True
    
    def get(self, num_samples):
        # Define a tensor of probabilities
        probs = probs = torch.ones(self.head_ptr + 1) / (self.head_ptr + 1)

        # Sample from the multinomial distribution
        samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)

        # Return the samples
        return self.samples[samples]
    
class CoolerReplayBuffer:
    
    def __init__(self, bin_width, bin_size, state_dim, action_dim = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bin_width = bin_width
        self.bin_size = bin_size
        
        # If action_dim is None the actions share the replay buffer
        # (i.e. all actions are stored in the same bins)
        if action_dim is None:
            self.bins = [defaultdict(lambda: Bin(self.bin_size))]
        else: # Each action has its own replay buffer
            self.bins = [defaultdict(lambda: Bin(self.bin_size)) for i in range(action_dim)]
        
    def get_idx(self, state : torch.tensor):
        return torch.round(state / self.bin_width)
       
    def store(self, sample : dict, randomize = False):
        
        # Avoid duplicates by adding random disturbance to the state
        if randomize:
            sample["state"] += np.random.normal(loc=0,scale=5e-4, size=self.state_dim)
            sample["next_state"] += np.random.normal(loc=0,scale=5e-4, size=self.state_dim)
            
        self.bins[sample["action"]][self.get_idx(sample["state"])].store(sample)
    
    def draw(self, n : int, action : int = None , replace : bool = False, min_req_ratio = 3):
        
        bins = []
        if action is None:
            bins = self.bins
        else:
            bins = [self.bins[action]]
        
        samples = []
        for action_bin in bins:
            for bin_ in action_bin.values():
                samples_ = bin_.get_samples()
                if samples_ is not None:
                    samples.append(samples_)
                    
        if samples != []:
            samples = np.concatenate(samples)
            
            if samples.size < n * min_req_ratio: return None
            
            return np.random.choice(samples, n, replace = False)
    
    def knn(self, k, state, action = None, neighbour_bin_dist = 5, min_req_ratio = 1.5, distance = "Euclidean"):
        """
        Get the knn samples from the given state
        
        state: n-dimensional array of floats
        action: int representing the different action. If None then all actions are used.
        k: number of neighbours
        neighbour_bin_dist: How far away we will sample from
        min_req_ratio: Least number of samples required before k-nearest neighbour is calculed
        """
        assert min_req_ratio >= 1, "min_req_ratio must be equal or above 1"
        assert neighbour_bin_dist >= 0, "neighbour_dist must be positive and non-zero"
        assert k >= 1, "At least 1 neighbour most be found"
        
        idx = self.get_idx_array(state)
        idxs = np.array(list(itertools.product(range(-neighbour_bin_dist, neighbour_bin_dist + 1), repeat=self.state_dim)), dtype = np.int32) + idx
        
        bins = []
        if action is None:
            bins = self.bins
        else:
            bins = [self.bins[action]]
        
        # Get all samples in all bins indexed by the idxs and store them as an array
        samples = []
        for action_bin in bins:
            for i in idxs:
                bin_ = action_bin[tuple(i)]
                samples_ = bin_.get_samples()
                if samples_ is not None:
                    samples.append(samples_)
        
        if samples != []:
            samples = np.concatenate(samples)
            
        # At least < k * min_req_ratio > samples are needed
        if samples.size < k * min_req_ratio:
            return None
        
        # Calculate distances between state and all samples
        distances = np.linalg.norm(np.array([sample["state"] for sample in samples]) - state, axis=1)
        
        # Get the indices of the k nearest neighbours
        knn_indices = np.argpartition(distances, k)[:k]
    
        return samples[knn_indices]
    
# Test dataset

# import matplotlib.pyplot as plt
# import time


# n = 100
# dim = 2 # dimention

# rb = CoolerReplayBuffer(dim, 3, 0.1, 100)

# all_data = []
# for loc in [0, 1, 2]:
#     data = np.array([np.random.normal(loc = loc, size = n) for _ in range(dim)]).T
    
#     all_data.append(data)
    
#     samples = [{"state" : d, "action" : loc} for d in data]
    
#     for sample in samples:
#         rb.store(sample)

# all_data = np.concatenate(all_data)

# knn = rb.knn(k = 100, state = np.array([0.5,0.5]), action = 0, neighbour_bin_dist=20, min_req_ratio=1)


# knn_state = np.array([sample["state"] for sample in knn])

# r_state = np.array([sample["state"] for sample in rb.draw(100)])

# plt.scatter(all_data[:,0],all_data[:,1])
# plt.scatter(r_state[:,0], r_state[:,1])
# plt.scatter(knn_state[:,0], knn_state[:,1])
# plt.show()
        
        
        
    
        