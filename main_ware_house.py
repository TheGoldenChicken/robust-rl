#%% Imports
import rl.manager
from ware_house import ware_house
from agent.robust_distributional_agent import robust_distributional_agent
import pickle
from collections import defaultdict

#%% Training
env = ware_house.Env(n = 10)

Q = []
    
for i in range(1):
    agent = robust_distributional_agent(env, tol = 0.05)
    manager = rl.manager.Manager(agent, render = True)
    print(f"iteration: {i}: " + str(manager.run(iterations = 20000)))
    print("total samples: " + str(agent.total_samples))

    Q += [list(agent.Q.values()),list(agent.Q.keys())]

with open('Q_values.pickle', 'wb') as handle:
    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Load the Q values
# Load the Q values
with open('Q_values.pickle', 'rb') as handle:
    Q_unpacked = pickle.load(handle)

Q = defaultdict(lambda : 0)

for i in range(0, len(Q_unpacked), 2):
    for j in range(len(Q_unpacked[i])):
        Q[Q_unpacked[i+1][j]] += (1/(i//2+1))*(Q_unpacked[i][j]-Q[Q_unpacked[i+1][j]])

env = ware_house.Env()

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