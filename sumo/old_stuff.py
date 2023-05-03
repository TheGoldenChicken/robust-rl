
class SumoNormalAgent:
    def __init__(self, fineness, obs_dim, replay_size, batch_size, num_actions):
        self.fineness = fineness
        self.grid_keys, self.grid_list = create_grid_keys(fineness)
        self.state_min, self.state_max = [-200, 200], [0, 600] # Something like this
        # self.state_min, self.state_max = self.env.get_max_min()

        #self.replay_buffer = TheCoolerReplayBuffer()


    def get_closest_grids(self, current_pos, neighbours=0, euclidian=0):
        # TODO Implement actual euclidian, not this cursed squared distance... doesn't scale well, as you probably know
        # Returns [x, y] location of the nearest grid...

        squared_dists = np.sum((current_pos - self.grid_keys)**2, axis=1)
        #squared_dists = np.sort(np.sum((current_pos - self.grid_keys)**2, axis=1))

        nearest_grid = self.grid_keys[np.argmin(squared_dists)]
        #nearest_grid = self.grid_keys[(squared_dists[:neighbours+1])]
        return nearest_grid


    def get_neighbour_grids(self, current_grid, neighbour_grids):
        # Gets neighbouring grid [x, y] locations from current grid [x, y] location
        # Index, not coordinates!

        # Get the index of the current grid_position
        grid_index = np.where(self.grid_list == current_grid)
        dims = len(self.state_max)

        # TODO: ATTACH FINENESS TO THE CLASS ITSELF
        # TODO: FIND THE WAY TO GET SELF.DIM
        multi_index = single_dim_interpreter(grid_index, fineness=self.fineness, dim=dims)
        neighbours_multi_index = neighbours(P=multi_index, neighbours=neighbour_grids,
                                            maxx=self.fineness-1, minn=0)
        neighbours_single_index = [multi_dim_interpreter(i, fineness=self.fineness, dim=None)
                                   for i in neighbours_multi_index]

        return neighbours_single_index


    def get_KNN(self, pos, action, K=0, neighbour_grids=0):
        """
        Get K-nearest neighbours for a given pos based on state
        :param pos: Current pos (state) to look based on
        :param action: Only look up places in the replay buffer that have an action corresponding to current action
        :param K: K neighbours to find
        :param neighbour_grids: Number of neighbouring grids next to current to look at
        :return:
        """
        # KNN = list(self.replay_buffer[action].keys())[np.argpartition(squared_dists, K)]
        # return keys[np.argmin(squared_dists)]
        # TODO: Make work for multiple actions

        closest_grid = self.get_closest_grids(pos, neighbours=0)
        idxs_to_look = self.get_neighbour_grids(closest_grid, neighbour_grids=neighbour_grids)

        all_points = self.replay_buffer[self.replay_buffer.create_partition_idxs(idxs_to_look, action)]

        all_points_of_interest = all_points['obs']
        # Finally, NN will be KNN among the all_points_of_interest, and we can use these for the purposes of sampling...
        # TODO: look into FAISS for getting KNN among all_points_of_interest

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

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


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

