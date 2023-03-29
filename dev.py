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

dim = 2
fineness = 4
original_list = list(range(fineness**dim))
print("Len of original list: ", len(original_list))

for i in range(10):
    rnd_idx = random.randint(0, len(original_list))
    print("Current idx:", rnd_idx)
    multi_dim = single_dim_interpreter(rnd_idx, fineness, dim)
    print("Multi Dim interpretation:", multi_dim)

    print("All neighbours ", neighbours(multi_dim, neighbours=1, maxx=4, minn=0))

    single_dim = multi_dim_interpreter(multi_dim, fineness, dim)
    print("Single Dim interpretation:", single_dim)

print(len(stencil(2, 1)))
print(stencil(2, 1))
print(len(stencil(2, 2)))
max = 5
min = -1
P = (4, 4, 4)

# The old get_KNN - For posterity
# def get_KNN(self, action, pos, K=0, neighbour_grids=0):
#     # Calc squared_dists
#     # Return key with lowest squared_dist
#
#     # # Get nearest grid_keys
#     # # Then get K nearest neighbours in the r nearest grid_keys from replay_buffer
#     # # THEN pass them to the get_normal_dist
#
#     # Get squared_dists to each of the grid_keys
#     squared_dists = np.sum((pos - self.grid_keys) ** 2, axis=1)
#     # nearest_grid = self.grid_keys[np.argmin(squared_dists)] # Old code - found only the closest grid location
#     nearest_grids = self.grid_keys[np.argmin(squared_dists, neighbour_grids)]
#
#     # If we don't want to get the K neareset neighbours - and just wanna explore
#     # Måske dele op til at have get_nearest_grids
#     if K == 0:
#         return nearest_grids
#
#     # TODO: Find way to look into r nearest grids in grid_keys (perhaps index grid_keys in a smart way?)
#     # Perhaps have grid_keys be a literal grid? With the i'th element being the i // Fineness row and the i % fineness column?
#     # Next, get all elements in those grids, get the squared dists and find the elements belonging to the K nearest
#     # Pass this to get_normal_dist
#
#     # TODO: Find a not retarded way to do what goes on below
#     if nearest_grids != 0:
#         # Gets the location of the current grid, useful for getting which partition to look in
#         dim_loc_1d = np.where(self.grid_keys == nearest_grids)
#
#         dim_loc = (self.grid_list[i].index(nearest_grids[i]) for i in self.grid_list)
#         max_min_idxs = [
#             (min(0, dim_loc[i] - neighbour_grids), max(len(self.grid_list) - 1, dim_loc[i] + neighbour_grids)) for i in
#             dim_loc]
#         all_grid_locations = [self.grid_keys[max_min_idxs[i][0]: max_min_idxs[i][1]] for i in max_min_idxs]
#         all_grids_to_search = list(itertools.product(*all_grid_locations))
#
#     # TODO: So what we really want, is getting a single dict with a numpy array for obs, rew, action, next_obs, so on
#     # But what we have is a bunch of dicts, each with their own numpy arrays
#     # Now, I GUESS that we could just take the numpy arrays from the dicts and concatenate them to a single one of the dicts
#     # OR we can remake the replay_buffers to instead of having a replay_buffer for each element in the dict, we have a singel
#     # replay_buffer, where the keys just say which pointer range to look at
#     # So each grid (assuming rth grid, and i size for replay_buffer), gets pointer space r*i:r*i+i
#     # This just begs the question: How do we make tuples link to pointer spaces? A seperate dict?
#     # Also, this puts increased presure onn a single object? But removes need for a dict....
#
#     # Dette samler bare indholdet af alle dicts i en
#     dd = defaultdict(list)
#     for d in (self.replay_buffer[i] for i in all_grids_to_search):
#         for key, value in d.items():
#             dd[key].append(value)
#     # ATTENTION: RIGHT NOW IS JUSTS NORMAL LIST, REMEMBER TO CHANGE TO NUMPY ARRAYS!!!
#     # Ignore Pycharm fucking up down here, iz alright
#     for key, value in dd.items():
#         dd[key] = np.array(dd[key])
#
#     return dd
#
#     # Essentielt dette som de to ovenstående linjer erstatter...
#     # x_loc = self.grid_list[0].index(nearest_grids[0])
#     # y_loc = self.grid_list[1].index(nearest_grids[1])
#     # x_min, x_max = min(0, x_loc-neighbour_grids)
#     # y_min, y_max = min(0, y_loc-neighbour_grids)
#
#     all_neighbour_grids = list(itertools.product(*all_state_dim_lists))
# #
# # KNN = list(self.replay_buffer[action].keys())[np.argpartition(squared_dists, K)]
# # return keys[np.argmin(squared_dists)]
#

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
    current_grid_idx = np.where(self.grid_list == current_grid)
    neighbour_grid_idxs = []
    for i in

    # Just get all where squared distance to current grid is smaller than given distance

    dim_loc = (self.grid_list[i].index(nearest_grids[i]) for i in self.grid_list)
    max_min_idxs = [(min(0, dim_loc[i] - neighbour_grids), max(len(self.grid_list) - 1, dim_loc[i] + neighbour_grids))
                    for i indim_loc]
