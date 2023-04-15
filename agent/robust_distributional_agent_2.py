
import rl.agent
from collections import defaultdict
import random
import numpy as np
import pygame
import matplotlib.pyplot as plt

class robust_distributional_agent(rl.agent.ShallowAgent):
    
    def __init__(self, env, gamma = 0.9, delta = 1, epsilon = 0.5, tol = 0.05):
        super().__init__(env)
        
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.tol = tol
        
        self.lr = lambda t : 1/(1+(1-self.gamma)*(t-1))
        self.t = 0
        
        self.total_samples = 0
        
        self.Q = defaultdict(lambda : 0)
    
    # Returns True if the environment is done (won or lost)
    def next(self) -> bool:
        self.t += 1
        alpha_t = self.lr(self.t)

        Q_ = defaultdict(lambda : 0)
        
        for state in self.env.get_states():
            actions = self.env.A(state)
            for action in actions:

                state_, reward = self.env.step(state, action)
                
                T_rob_e = reward + self.gamma*max([self.Q[(state_, b)] for b in self.env.A(state_)])
                
                Q_[state,action] = (1-alpha_t)*self.Q[state, action] + alpha_t*T_rob_e


        Q_diffs = []
        for key in self.Q.keys():
            Q_diffs.append(np.abs(self.Q[key]-Q_[key]))
        distance = np.max(Q_diffs)
        
        if distance < self.tol:
            print(">>> (CONVERGED) Diff Inf Norm:", distance)
            return True
        elif(self.t%100 == 0):
            print(">>> Diff Inf Norm:", distance)
        
        self.Q = Q_
        
        return False