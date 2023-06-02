import rl.agent
from collections import defaultdict
import numpy as np
import pygame
import time

class FirstVisitMonteCarlo(rl.agent.ShallowAgent):
    
    def __init__(self, env, policy, gamma = 0.95, lr = lambda : 0.01) -> None:
        super().__init__(env)
        
        self.policy = policy
        self.lr = lr
        self.gamma = gamma
        
        # self.returns = defaultdict(lambda : [])
        self.N = defaultdict(lambda : 0)
        
        
    def next(self):
        
        self.state = self.env.reset()
        trajectory = []
        rewards = []

        prev_reward = 0
        while not self.env.is_terminal(self.state):
            action = self.policy.get_action(self)
            
            trajectory.append((self.state, action, ))
            rewards.append(prev_reward)

            
            self.env.render(self)
            time.sleep(0.5)
            
            self.state, prev_reward = self.env.step(self.state, action)
            
            
            
            # If the state is terminal, append the last state and reward
            if self.env.is_terminal(self.state):
                trajectory.append((self.state, None, ))
                rewards.append(prev_reward)
        
        G = 0
        # Enumerate reversed trajectory
        for i, (state, action) in enumerate(reversed(trajectory)):
            index = len(trajectory) - i - 1
            G = self.gamma * G + rewards[index]
            if ((state, action) not in trajectory[:index]):
                self.N[(state, action)] += 1
                self.Q[(state, action)] += 1/self.N[(state, action)] * (G - self.Q[(state, action)])
                # self.returns[(state, action)].append(G)
                # self.Q[(state, action)] = np.mean(self.returns[(state, action)])