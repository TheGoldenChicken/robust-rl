import WareHouseEnv as warehouse
import GridWorldEnv as gridworld
import Agent
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

env = warehouse.WareHouseEnv(n = 10, h = 1, k = 2, p = 3)


### Value iteration agent
agent = Agent.DiscreteDestributionalMLMCRobustAgent(env)
for i in tqdm(range(2000)):
    agent.next()

# Value function
# x = list(agent.v.keys())
# y = list(agent.v.values())
# plt.plot(x,y)
# plt.show()

# # Action value function
img = np.zeros((env.n+1, env.n+1))
for i in agent.q.keys():
    img[i[0],i[1]] = agent.q[i]

plt.imshow(img)
plt.colorbar()
plt.show()


# ### Manual agent
# agent = Agent.ManualAgent(env)
# running = True
# while(running):
#     running = agent.next()
    

    
    
    
    




