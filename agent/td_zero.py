import rl.agent
from collections import defaultdict
import numpy as np
import pygame
import time

class TDZero(rl.agent.ShallowAgent):
    
    def __init__(self, env, policy, gamma = 0.95, lr = lambda : 0.95) -> None:
        super().__init__(env)
        
        self.policy = policy
        self.lr = lr
        self.gamma = gamma
        
        self.returns = defaultdict(lambda : [])
        
        
    def next(self):

        self.env.render(self)
        # time.sleep(0.01)

        action = self.policy.get_action(self)

        state = self.state
        state_, reward = self.env.step(state, action)
        
        self.state = state_
        action_ = self.policy.get_action(self)
        
        self.Q[(state, action)] += self.lr()*(reward
                                                + self.gamma*self.Q[(state_, action_)]
                                                - self.Q[(state, action)]) 
        
        return False
        
        