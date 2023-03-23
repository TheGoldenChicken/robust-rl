import itertools
import numpy as np

# def create_grid_keys(fineness, actions=2):
#     """
#     Generates positions representing the center of squares used in the grid for NN purposes
#     Generates grid with keys equal to finess^state_dim
#
#     returns: List of (state_dim) tuples containing the keys for a grid
#     """
#     # Should get [[state_dim_1_min,state_dim_1_max], [state_dim_2_min,state_dim_2_max]]
#     #max_min = self.env.get_max_min_state()
#
#     # Dummy var - for testing purposes
#     max_min = [[-200, 200], [0, 600]]
#
#     all_state_dim_lists = []
#     for state_dim in max_min:
#         curr_state_dim_list = []
#         # From min in state_dim to max in state_dim...
#         # With a jump equal to the max distance in each state_dim over the fineness chosen (n squares per dim)
#         width = sum(abs(r) for r in state_dim)/fineness
#         for i in range(state_dim[0], state_dim[1], int(sum(abs(r) for r in state_dim)/fineness)):
#             curr_state_dim_list.append(i)
#         all_state_dim_lists.append(curr_state_dim_list)
#     all_state_dim_lists.append(list(range(actions)))
#     print(all_state_dim_lists)
#     all_combs = list(itertools.product(*all_state_dim_lists))
#
#     return all_combs
#
# print(len(create_grid_keys(5)))
#
#
# def get_NN(pos, keys):
#     # Calc squared_dists
#     # Return key with lowest squared_dist
#     squared_dists = np.sum((pos - keys)**2, axis=1)
#     return keys[np.argmin(squared_dists)]
#
# #def get_normal_dist():

import itertools

def single_dim_interpreter(idx, fineness, dim):
    # Takes single dim idx that we know codes for a flattened list of dim dimensions
    # With fineness elements per dimension row

    interpreted = [0] * dim
    # Reverse iteration
    for i in range(dim-1,-1, -1):

        current_dim_size = idx // fineness**i
        interpreted[i] = current_dim_size
        idx = idx - current_dim_size * fineness**i

    return interpreted

def multi_dim_interpreter(idxs, fineness, dim):
    # Takes a multi dimensional coordinate, transforms it to single dim flattened
    # Technically don't need dim, since this is inferred from len(idxs)

    idx = 0
    for r, i in enumerate(idxs):
        idx += i * fineness**r
    return idx

import random
dim = 2
fineness = 4
original_list = list(range(fineness**dim))
print("Len of original list: ", len(original_list))

for i in range(10):
    rnd_idx = random.randint(0,len(original_list))
    print("Current idx:", rnd_idx)
    multi_dim = single_dim_interpreter(rnd_idx, fineness, dim)
    print("Multi Dim interpretation:", multi_dim)
    single_dim = multi_dim_interpreter(multi_dim, fineness, dim)
    print("Single Dim interpretation:", single_dim)
def stencil(dim, neighbours):
    stencils = list(itertools.product(list(range(-neighbours, neighbours+1)), repeat=dim))
    zero = ((0,) * dim)
    stencils.remove(zero)
    stencils = [i for i in stencils if not i.contains()]
    return stencils

def neighbours(P, max=None, min=None):
    stencils = stencil(len(P))
    positions = [tuple([sum(x) for x in zip(P, s)]) for s in stencils]

    # Remove all smaller than max or min
    if max != None:
        positions = [i for i in positions if max(i) <= max]
    if min != None:
        positions = [i for i in positions if min(i) >= min]
    return positions
#
# print(len(stencil(2, 1)))
# print(stencil(2, 1))
# print(len(stencil(2, 2)))
# max = 5
# min = -1
# P = (4, 4, 4)