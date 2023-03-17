import itertools
import numpy as np

def create_grid_keys(fineness, actions=2):
    """
    Generates positions representing the center of squares used in the grid for NN purposes
    Generates grid with keys equal to finess^state_dim

    returns: List of (state_dim) tuples containing the keys for a grid
    """
    # Should get [[state_dim_1_min,state_dim_1_max], [state_dim_2_min,state_dim_2_max]]
    #max_min = self.env.get_max_min_state()

    # Dummy var - for testing purposes
    max_min = [[-200, 200], [0, 600]]

    all_state_dim_lists = []
    for state_dim in max_min:
        curr_state_dim_list = []
        # From min in state_dim to max in state_dim...
        # With a jump equal to the max distance in each state_dim over the fineness chosen (n squares per dim)
        width = sum(abs(r) for r in state_dim)/fineness
        for i in range(state_dim[0], state_dim[1], int(sum(abs(r) for r in state_dim)/fineness)):
            curr_state_dim_list.append(i)
        all_state_dim_lists.append(curr_state_dim_list)
    all_state_dim_lists.append(list(range(actions)))
    print(all_state_dim_lists)
    all_combs = list(itertools.product(*all_state_dim_lists))

    return all_combs

print(len(create_grid_keys(5)))


def get_NN(pos, keys):
    # Calc squared_dists
    # Return key with lowest squared_dist
    squared_dists = np.sum((pos - keys)**2, axis=1)
    return keys[np.argmin(squared_dists)]

#def get_normal_dist():