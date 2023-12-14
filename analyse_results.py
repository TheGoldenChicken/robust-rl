import torch
import numpy as np
from cliff_car_env_minimize import CliffCar as CliffCarMinimize
from cliff_car_dqn_agent import DQNCliffCarAgent
from network import RadialNonLinearNetwork2d

model_path = ""

env = CliffCarMinimize(noise_var = 0.01,
                       radial_basis_dist = 2,
                       radial_basis_var = 3)
network = RadialNonLinearNetwork2d(env)
agent = DQNCliffCarAgent(env=env, replay_buffer=None, network = network)

agent.load_model(model_path)

all_sars = agent.test(test_games = 100, frame = None)

env_size = (env.BOUNDS[2] - env.BOUDNS[0], env.BOUNDS[3] - env.BOUNDS[1])
res = 50
X, Y = np.mgrid[env.BOUNDS[0]:env.BOUNDS[2]:env_size[0]*res, env.BOUNDS[1]:env.BOUNDS[3]:env_size[0]*res]
xy = np.vstack((X.flatten(), Y.flatten())).T
states = torch.FloatTensor(xy).to(agent.device)

q_vals = agent.get_q_vals(states)

q_vals = torch.reshape(q_vals, X.shape)





