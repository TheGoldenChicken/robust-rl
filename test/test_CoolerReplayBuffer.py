import pytest
from sumo.replay_buffer import TheCoolerReplayBuffer
import numpy as np

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
    actions = [1,2] # TODO: CHANGE THIS WHEN YOU UPDATE STUFF
    rews = np.arange(len(obs))
    next_obs = np.copy(obs)
    done = np.array([False] * len(obs))
    idx = [0, 1, 6, 0] # Last one test for TB
    to_store = [obs, actions, rews, next_obs, done, idx]
    replay_buffer.store(*to_store)



