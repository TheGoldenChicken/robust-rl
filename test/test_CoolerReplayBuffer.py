import pytest
from sumo.replay_buffer import TheCoolerReplayBuffer
import numpy as np
from sumo import sumo_utils
# TODO: PERHAPS TEST MULTIPLE INSTANCES OF A REPLAY BUFFER
obs_dim = 2
bin_size = 100
batch_size = 32
fineness = 3
num_actions = 2
state_max = [600, 600]
state_min = [0,0]
tb = True


def random_replay_buffer():
    obs_dim = np.arange(10)
    bin_size = np.arange(10, 1000, 10)
    batch_size = np.arange(32, 32*6, 32)
    fineness = np.arange(3, ) # TODO: CONSIDER WHETHER OR NOT HAVING A LARGE FINENESS VALUE COMPARED TO MAX/MIN WILL BE SCREWY - YA KNOW DECIMALS BEING ROUNDED DOWN TO THE SAME VALUES??

@pytest.fixture
def replay_buffer():
    rep_buffer = TheCoolerReplayBuffer(obs_dim, bin_size, batch_size, fineness,
                                       num_actions, state_max, state_min, tb)

    yield rep_buffer
    print("closing fixture")

def test_advanced_storing():

    replay_buffer = TheCoolerReplayBuffer(obs_dim, bin_size, batch_size, fineness,
                                          num_actions, state_max, state_min, tb)
    trash_counter = 0
    # Test storing loads of obs
    obs_num = 5000
    number_actions = 2
    for i in range(obs_num): # repeats
        for r in range(number_actions): # actions
            obs = np.random.randint(0, 600, size=2) # Loads of trash observations
            rews = np.random.randn()
            next_obs = np.copy(obs)
            done = False

            idx = replay_buffer.get_bin_idx(s=obs, single_dim=True)
            to_store = [obs, r, rews, next_obs, done, idx]

            trash_counter += replay_buffer.store(*to_store)

    assert len(replay_buffer) + trash_counter == obs_num * number_actions, "Correct size of replay_buffer"

def test_basic_storing(replay_buffer):
    print(replay_buffer)  # Init works
    obs = np.array([[0, 0], [101, 100], [201, 201], [900, 900]])
    actions = [0, 1]  # TODO: CHANGE THIS WHEN YOU UPDATE STUFF
    rews = np.arange(len(obs))
    next_obs = np.copy(obs)
    done = np.array([False] * len(obs))
    idx = [0, 1, 6, 0]  # Last one test for TB

    # Test storing
    for action in actions:
        for i in range(len(obs)):
            to_store = [obs[i], action, rews[i], next_obs[i], done[i], idx[i]]
            replay_buffer.store(*to_store)

    assert len(replay_buffer) == 8, "4 states times two actions"

def test_retrieval(replay_buffer):
    obs_num = 10000
    number_actions = 2
    trash_counter = 0

    for i in range(obs_num): # repeats
        for r in range(number_actions): # actions
            obs = np.random.randint(0, 600, size=2) # Loads of trash observations
            rews = np.random.randn()
            next_obs = np.copy(obs)
            done = False

            idx = replay_buffer.get_bin_idx(s=obs, single_dim=True)
            to_store = [obs, r, rews, next_obs, done, idx]

            trash_counter += replay_buffer.store(*to_store)

    states_to_check = [[0,0],[100,100],[150,150],[201,1],[1,201],[401,401],[601,601],[399, 401]]
    correct_bins = [18, 0, 0, 1, 3, 8, 18, 7] # Yeah it is not inclusive on either min or max values... makes it simpler
    correct_bins_actions = [correct_bins, [i+9 if i != 18 else i for i in correct_bins]]
    print(correct_bins)
    for p, action_i in enumerate(correct_bins_actions):
        for i, r in zip(states_to_check, action_i):
            idx = replay_buffer.get_bin_idx(i, single_dim=True, test=True)
            if idx != 18:
                idx += 9*p
            assert idx == r
            #assert replay_buffer.get_bin_idx(i, single_dim=True, test=True)+9*p* == r

