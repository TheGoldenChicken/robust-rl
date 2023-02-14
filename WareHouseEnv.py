import numpy as np
from collections import defaultdict

class WareHouseEnv():
    
    def __init__(self, n = 10, h = 1, p = 2, k = 3):
        self.n = n
        self.h = h
        self.p = p
        self.k = k


    def step(self, state, action, r = None):
        
        # Random uniformly distributed demand between 0 and n
        if(r == None):
            demand = np.random.randint(0, self.n)
        else:
            demand = r
        
        # Holding cost
        cost = self.h*(state + action - demand)
        
        # Lost sales penalty
        cost += self.p*(demand - state - action)
        
        # Fixed ordering cost
        #cost += self.k*action
        if(action > 0): cost += self.k
        
        next_state = state + action - min(demand, state + action)
        
        reward = -cost
        
        return next_state, reward
    
    def get_transistion_probabilities(self, state, action):
        
        p = defaultdict(lambda : 0)
        
        num_states = len(self.get_states())
        for d in range(num_states):
            
            next_state, reward = self.step(state, action, d)
            
            p[(next_state, reward)] += 1/num_states
        
        return p
        
    # Function that returns the available actions
    def A(self, state):
        return np.arange(self.n - state + 1)
    
    # Function that return all possible states
    def get_states(self):
        return np.arange(self.n + 1)

    def reset(self):
        return 0

    def init_render(self):
        pass
        #pygame.init()
        #self.display = new_display()
		
    def render(self):
        pass
        #if not rendering:
        #    self.init_render()
        #self.display.update()
        
    def get_accumulative_reward(self, agent):
        return sum(agent.obtained_rewards)