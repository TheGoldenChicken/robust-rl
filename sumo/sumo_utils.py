import numpy as np
import itertools


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

    #max_min = [[-200, 200], [0, 600]]

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

a, b = create_grid_keys(5, max_min=[[-1000,1000], [0, 500]])
print(a)


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
