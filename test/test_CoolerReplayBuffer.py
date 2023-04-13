import pytest
from sumo.replay_buffer import TheCoolerReplayBuffer
import numpy as np
from sumo import sumo_utils

obs_dim = 2
bin_size = 10
batch_size = 32
fineness = 3
num_actions = 2
state_max = [300, 600]
state_min = [0,0]
tb = True

@pytest.fixture
def replay_buffer():
    rep_buffer = TheCoolerReplayBuffer(obs_dim, bin_size, batch_size, fineness,
                                       num_actions, state_max, state_min, tb)
    yield rep_buffer
    print("closing fixture")

def test_foo_first(replay_buffer):
    print(replay_buffer) # Init works
    obs = np.array([[0, 0], [101, 100], [201, 201], [900, 900]])
    actions = [0,1] # TODO: CHANGE THIS WHEN YOU UPDATE STUFF
    rews = np.arange(len(obs))
    next_obs = np.copy(obs)
    done = np.array([False] * len(obs))
    idx = [0, 1, 6, 0] # Last one test for TB

    # Test storing
    for action in actions:
        for i in range(len(obs)):
            to_store = [obs[i], action, rews[i], next_obs[i], done[i], idx[i]]
            replay_buffer.store(*to_store)

    assert len(replay_buffer) == 8, "4 states times two actions"

    for i in range(1000): # repeats
        for r in range(2): # actions
            obs = np.random.randint(0, 1000, size=2) # Loads of trash observations
            rews = np.random.randn()
            next_obs = np.copy(obs)
            done = False

            idx = replay_buffer.get_bin_idx(s=obs, single_dim=True)
            to_store = [obs, r, rews, next_obs, done, idx]
            replay_buffer.store(*to_store)