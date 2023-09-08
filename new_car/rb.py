import numpy as np
from collections import defaultdict
import itertools

class Bin:
    
    def __init__(self, size):
        
        self.size = size
        self.samples = np.empty((size,), dtype = dict)
        self.is_full = False
        self.head_ptr = -1
        
    def store(self, sample):
        self.head_ptr += 1
        self.samples[self.head_ptr] = sample
        
        # If the head ptr is at the end list then 
        if (self.head_ptr == self.size - 1):
            self.head_ptr = 0
            self.is_full = True
    
    def get_samples(self):
        if self.head_ptr == -1: # if empty
            return None
        if self.is_full: # if full
            return samples
        return self.samples[:self.head_ptr + 1] #else
    
class CoolerReplayBuffer:
    
    def __init__(self, state_dim, action_dim, bin_width, bin_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bin_width = bin_width
        self.bin_size = bin_size
        
        self.bins = [defaultdict(lambda: Bin(self.bin_size)) for i in range(action_dim)]
        
    def get_idx(self, state : np.array):
        return tuple(np.round(state / self.bin_width).astype(np.int32))
    
    def get_idx_array(self, state : np.array):
        return np.round(state / self.bin_width).astype(np.int32)
        
    def store(self, sample : dict):
        self.bins[sample["action"]][self.get_idx(sample["state"])].store(sample)
    
    def draw(self, n : int, action : int = None , replace : bool = False, min_req_ratio = 3, repeat = 1):
        
        batches = []
        
        for _ in range(repeat):
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
                
                batches.append(np.random.choice(samples, n, replace = False))
            
        return batches
    
    def knn(self, k, state, action = None, neighbour_bin_dist = 1, min_req_ratio = 1.5, repeat = 1, distance = "Euclidean"):
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
        
        batches = []
        
        for _ in range(repeat):
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
                print(samples)
                samples = np.concatenate(samples)
                
            # At least < k * min_req_ratio > samples are needed
            if samples.size < k * min_req_ratio:
                return None
            
            # Calculate distances between state and all samples
            distances = np.linalg.norm(np.array([sample["state"] for sample in samples]) - state, axis=1)
            
            # Get the indices of the k nearest neighbours
            knn_indices = np.argpartition(distances, k)[:k]
        
            batches.append(samples[knn_indices])
            
        # Return the k nearest neighbours
        return batches

# Test dataset

import matplotlib.pyplot as plt
import time


n = 100
dim = 2 # dimention

rb = CoolerReplayBuffer(dim, 3, 0.1, 100)

all_data = []
for loc in [0, 1, 2]:
    data = np.array([np.random.normal(loc = loc, size = n) for _ in range(dim)]).T
    
    all_data.append(data)
    
    samples = [{"state" : d, "action" : loc} for d in data]
    
    for sample in samples:
        rb.store(sample)

all_data = np.concatenate(all_data)

knn = rb.knn(k = 100, state = np.array([0.5,0.5]), action = 0, neighbour_bin_dist=20, min_required_ratio=1)


knn_state = np.array([sample["state"] for sample in knn])

r_state = np.array([sample["state"] for sample in rb.draw(100)])

plt.scatter(all_data[:,0],all_data[:,1])
plt.scatter(r_state[:,0], r_state[:,1])
plt.scatter(knn_state[:,0], knn_state[:,1])
plt.show()
        
        
        
    
        