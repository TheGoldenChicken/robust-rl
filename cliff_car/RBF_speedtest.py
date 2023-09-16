import torch
import numpy
import time
from cliff_car_env import CliffCar
from network import Network, RadialNetwork2d, RadialNonlinearNetwork2d
from tqdm import tqdm
#
# selected_action = self.dqn(
#                 torch.FloatTensor(state).to(self.device)
#             ).argmax(axis=1)[0]
#
# agent = RobustCliffCarAgent(env=env, replay_buffer=replay_buffer, network=RadialNetwork2d,
#                             grad_batch_size=grad_batch_size,
#                             delta=delta, epsilon_decay=epsilon_decay, max_epsilon=0.5, min_epsilon=0.05,
#                             gamma=0.99, robust_factor=factor, linear_only=linear)
#
#
# def torch_multi_test(num_iterations)
#

# To get random states from the environment:
env = CliffCar(noise_var=0.5, max_duration=None)

num_forward = 10000
device = 'cpu'
rbf_network = RadialNetwork2d(env)



for i in tqdm(range(num_forward)):
    position, _, _, _ = env.step(0) # Don't need the rest
    # forward = rbf_network(torch.FloatTensor(position).to(device))
    forward = rbf_network.fast_forward(position)