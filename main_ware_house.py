#%% Imports
import rl.manager
from ware_house import ware_house
# from agent.robust_distributional_agent import robust_distributional_agent
from agent.robust_distributional_agent_2 import robust_distributional_agent
from agent.replay_agent import replay_agent
import numpy as np
import matplotlib.pyplot as plt
import rl.policy
import pickle
from collections import defaultdict

#%% Training
env = ware_house.Env(playerOptions = None)

Q = []

for i in range(5):
    agent = robust_distributional_agent(env, tol = 0.01)
    manager = rl.manager.Manager(agent, render = True)
    print(f"iteration: {i}: " + str(manager.run(iterations = 10000)))
    print("total samples: " + str(agent.total_samples))

    Q += [list(agent.Q.values()),list(agent.Q.keys())]

with open('Q_values.pickle', 'wb') as handle:
    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Testing

class policy_optimal:

    def get_action(self, agent):
        if(agent.state <= 2):
            return [(8-agent.state,)]
        else:
            return [(0,)]

class policy_robust:

    def get_action(self, agent):
        if(agent.state <= 4):
            return [(7-agent.state,)]
        else:
            return [0]


# Load the Q values
with open('Q_values.pickle', 'rb') as handle:
    Q_unpacked = pickle.load(handle)

Q = defaultdict(lambda : 0)

for i in range(0, len(Q_unpacked), 2):
    for j in range(len(Q_unpacked[i])):
        Q[Q_unpacked[i+1][j]] += (1/(i//2+1))*(Q_unpacked[i][j]-Q[Q_unpacked[i+1][j]])
# for array in Q_unpacked:
#     for i, e in enumerate(array):
#         Q[e[0]] += (1/(i+1))*e[1]

env = ware_house.Env(playerOptions = None)

# Convert Q values to policy values
pi = defaultdict(float)
for state in env.get_states():
    Q_max = float('-inf')
    best_action_set = None
    for action_set in env.A(state):
        if Q_max < Q[(state, action_set)]:
            Q_max = Q[(state, action_set)]
            best_action_set = action_set
    # self.V[state] = Q_max
    pi[state] = best_action_set

print(pi)
# fig, ax = plt.subplots(1, 4, figsize = (20, 5))

# n = 10
# for i, b in enumerate([1, 1.5, 2, 2.5]):
#     for m in range(0, n):
#         env_non_uniform = ware_house.EnvNonUniform(playerOptions = None, n = n, b = 1, m = 0)
#         policy = rl.policy.EpsilonGreedy(env_non_uniform, epsilon = 0, decay = 1)
#         agent_converged = replay_agent(env_non_uniform, policy)

#         optimal = policy_optimal()
#         robust = policy_robust()

#         agent_optimal = replay_agent(env_non_uniform, optimal)
#         agent_robust = replay_agent(env_non_uniform, robust)

#         manager_converged = rl.manager.Manager(agent_converged, render = False)
#         manager_optimal = rl.manager.Manager(agent_optimal, render = False)
#         manager_robust = rl.manager.Manager(agent_robust, render = False)

#         print(f"Evaluating: b = {b}, m = {m}")
#         manager_converged.run(iterations = 2000)
#         manager_optimal.run(iterations = 2000)
#         manager_robust.run(iterations = 2000)

        
#         mean_return_converged = np.mean([np.mean(value) for value in agent_converged.results.values()])
#         mean_return_optimal = np.mean([np.mean(value) for value in agent_optimal.results.values()])
#         mean_return_robust = np.mean([np.mean(value) for value in agent_robust.results.values()])
        
#         ax[i].plot(m, mean_return_converged, 'o', color = 'blue')
#         ax[i].plot(m, mean_return_optimal, 'o', color = 'red')
#         ax[i].plot(m, mean_return_robust, 'o', color = 'green')

# plt.show()


# #%%

# from agent.td_zero import TDZero

# env = ware_house.Env(playerOptions = None)
# policy = rl.policy.EpsilonGreedy(env, epsilon = 0.05, decay = 1)
# agent = TDZero(env, policy, gamma = 0.95)
# manager = rl.manager.Manager(agent, render = True)

# manager.run(iterations = 100000)

# # # # Action value function
# # img = np.zeros((env.n+1, env.n+1))
# # for i in agent.Q.keys():
# #     img[i[0],i[1]] = agent.Q[i]

# # plt.imshow(img)
# # plt.colorbar()

# # # Save the image
# # plt.savefig("Q_function.png")

# # plt.show()


# # for i in range(0, 2000):
# #     initialize s_0 0 0
# #     simulate for T steps
# #     s, r_t = next(s,a)
# #     cum_r_t = sum of rewards sum(gamma^t * r_t)
# # 1/2000 sum(cum_r)