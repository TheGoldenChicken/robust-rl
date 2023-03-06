import rl.manager
import ware_house
from agent.robust_distributional_agent import robust_distributional_agent
import numpy as np
import matplotlib.pyplot as plt

env = ware_house.Env(playerOptions = None)
agent = robust_distributional_agent(env)

manager = rl.manager.Manager(env, agent)

print("iteration: " + manager.run(iterations = 300))

# # Action value function
img = np.zeros((env.n+1, env.n+1))
for i in agent.Q.keys():
    img[i[0],i[1]] = agent.Q[i]

plt.imshow(img)
plt.colorbar()

# Save the image
plt.savefig("Q_function.png")

plt.show()