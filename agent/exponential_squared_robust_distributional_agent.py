
import rl.agent
from collections import defaultdict
import random
import numpy as np


class robust_distributional_agent(rl.agent.DeepAgent):
    
    def __init__(self, env, gamma = 0.5, delta = 1, epsilon = 0.05, tol = 0.05):
        super().__init__(env)
        
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon # Epsilon greedy
        self.tol = tol
        
        self.lr = lambda t : 1/(1+(1-self.gamma)*(t-1))
        self.t = 0
    
    # Returns True if the environment is done (won or lost)
    def next(self) -> bool:
        
        # Epsilon greedy sample an action from the current state
        
        
        # Sample similar states and state values from the replay buffer
        states, state_values = self.replay_buffer.sample(state, samples = 100)
        
        # Sample new states from similar states where the 
        
        return False