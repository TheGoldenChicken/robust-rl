import WareHouseEnv as warehouse
import GridWorldEnv as gridworld
import WareHousePlayer as wh_player
import rl.agent as agent
import matplotlib.pyplot as plt
import numpy as np

env = gridworld.GridWorldEnv()


### Value iteration agent
agent = agent.DiscreteActionValueIterationAgent(env)
for i in range(5):
    agent.next()


layout = np.array(env.layout)
v = np.zeros((layout.shape[0], layout.shape[1]))
for i in agent.v.keys():
    v[i] = agent.v[i]
    
fig, ax = plt.subplots()
im = ax.imshow(v)
    
for x in range(layout.shape[0]):
    for y in range(layout.shape[1]):
        print(x,y)
        text = ax.text(y, x, round(v[x, y],2), ha="center", va="center", color="w")

plt.imshow(v)
plt.show()