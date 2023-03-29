from collections import defaultdict

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

import ware_house.ware_house as ware_house
env = ware_house.Env(playerOptions = None)

pi = ValueIteration()
print(pi(env))