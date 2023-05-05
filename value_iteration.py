#%%
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class ValueIteration:

    def __init__(self):
        self.Q = defaultdict(float)
        self.V = defaultdict(float)

    def __call__(self, env, gamma = 0.99, tol = 1e-3):

        Delta = float('inf')
        
        while(Delta > tol):        
            Q_ = defaultdict(float)
            Delta = 0

            # State value iteration
            for state in env.get_states():
                for action_set in env.A(state):
                    Q_old = self.Q[state, action_set]
                    for (state_, reward), p in env.get_transistion_probabilities(state, action_set).items():
                        Q_max = max([self.Q[state_, action_set_] for action_set_ in env.A(state_)])
                        Q_[state,action_set] += p * (reward + gamma * Q_max)

                    Delta = max(Delta, abs(Q_old - Q_[(state,action_set)]))
            # Update the value functions
            self.Q = Q_
        
        # Convert Q values to policy values
        pi = defaultdict(float)
        for state in env.get_states():
            Q_max = float('-inf')
            best_action_set = None
            for action_set in env.A(state):
                if Q_max < self.Q[(state, action_set)]:
                    Q_max = self.Q[(state, action_set)]
                    best_action_set = action_set
            self.V[state] = Q_max
            pi[state] = best_action_set
        
        return pi

def optimal_std_policy(state):

    if state <= 2:
        return (8-state,)
    else:
        return (0,)

def optimal_robust_policy(state):

    if state <= 4:
        return (7-state,)
    else:
        return (0,)

def std_policy_distribution(state, action_set):
    if state <= 2 and action_set == (8-state,):
        return 1
    elif state > 2 and action_set == (0,):
        return 1
    else:
        return 0

def robust_policy_distribution(state, action_set):
    if state <= 4 and action_set == (7-state,):
        return 1
    elif state > 4 and action_set == (0,):
        return 1
    else:
        return 0

class policy_evaluation:

    def __init__(self):
        self.V = defaultdict(float)
        self.Q = defaultdict(float)

    def __call__(self, env, pi_dist, gamma = 0.9, iterations = 2000):
        
        for i in range(iterations):
            V_ = defaultdict(float)

            for s in env.get_states():
                for a in env.A(s):
                    pi = pi_dist(s, a)
                    trans_prob = env.get_transistion_probabilities(s, a)
                    tmp = 0
                    for s_, r in list(trans_prob.keys()):
                        tmp += trans_prob[(s_, r)] * (r + gamma * self.V[s_])
                    V_[s] += pi * tmp
            
            self.V = V_

        return self.V

import ware_house.ware_house as ware_house
# env = ware_house.Env()

# value_iter = ValueIteration()
# pi = value_iter(env)
# print("policy:", pi)

print("\n------------\n")

# env = ware_house.EnvNonUniform(b = 1, m = 0)
policy_eval = policy_evaluation()

# V = policy_eval(env, std_policy_distribution)
# print("value iteration:", list(V.items()))


def iterate(env, policy, runs = 2000):
    reward = [0]*runs
    s = 0
    for i in range(runs):
        action_set = policy(s)
        s, r = env.step(s, action_set)
    
        reward[i] += r
    return np.mean(reward)

average_std = [[],[],[],[]]
average_robust = [[],[],[],[]]
for i, b in tqdm(enumerate([1,1.5,2,2.5])):
    for m in tqdm(range(0,10)):
        env = ware_house.EnvNonUniform(b = b, m = m)
        V_std = policy_eval(env, std_policy_distribution)
        V_robust = policy_eval(env, robust_policy_distribution)

        average_std[i].append(np.mean(list(V_std.values())))
        average_robust[i].append(np.mean(list(V_robust.values())))
        # average_std.append(iterate(env, optimal_std_policy))
        # average_robust.append(iterate(env, optimal_robust_policy))



for robust, std in zip(average_robust, average_std):
    plt.plot(robust, '.', label = "Distributionally robust policy")
    plt.plot(std, '.', label = "Non Distributionally robust policy")
    plt.legend()
    plt.xlabel("m")
    plt.ylabel("Cost")
    plt.show()

