import rl.manager
from ware_house import ware_house
from agent.robust_distributional_agent import robust_distributional_agent
import numpy as np
import matplotlib.pyplot as plt
from agent.td_zero import TDZero
import rl.policy
import pickle

env = ware_house.Env(playerOptions = None)
agent = robust_distributional_agent(env)
# policy = rl.policy.EpsilonGreedy(env, epsilon = 0.05, decay = 1)
# agent = TDZero(env, policy, gamma = 0.95, lr = lambda : 0.95)
manager = rl.manager.Manager(agent, render = False)

print("iteration: " + str(manager.run(iterations = 10)))
print("total samples: " + str(agent.total_samples))

Q = [list(agent.Q.values()),list(agent.Q.keys())]

with open('Q_values.pickle', 'wb') as handle:
    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Action value function
img = np.zeros((env.n+1, env.n+1))
for i in agent.Q.keys():
    img[i[0],i[1]] = agent.Q[i]

plt.imshow(img)
plt.colorbar()

# Save the image
plt.savefig("Q_function.png")

plt.show()

