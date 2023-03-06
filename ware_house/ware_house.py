import numpy as np
from collections import defaultdict
import rl.env
import rl.player
import rl.display

class Env(rl.env.DiscreteEnv):
    
    def __init__(self, playerOptions, n = 10, h = 1, p = 2, k = 3) -> None:
        super().__init__()
        
        self.n = n
        self.h = h
        self.p = p
        self.k = k
        
        # List of players
        self.players = list()
        self.state_size = 0
    
    # Return the next state and reward
    def step(self, state : tuple, actions : list[tuple]) -> tuple[list[tuple],list[int]]:
        # Random uniformly distributed demand between 0 and n
        demand = np.random.randint(0, self.n)
        
        # Holding cost
        cost = np.abs(self.h*(state + actions[0] - demand))
        
        # Lost sales penalty
        cost += np.abs(self.p*(demand - state - actions[0]))
        
        # Fixed ordering cost
        # cost += self.k*actions[0]
        if(actions[0] > 0): cost += self.k
        
        next_state = state + actions[0] - min(demand, state + actions[0])
            
        reward = -cost
        
        return next_state, reward
    
    # Function that returns the available actions
    def A(self, state):
        return np.arange(self.n - state + 1)
    
    def goal_test(self, state) -> bool:
        raise NotImplementedError("ContinuousEnv.goal_test() is not implemented")
        
        # Return True if the state is a goal state
        return False
    
    # Reset state by returning initial state
    def reset(self) -> None:
        return 0
        
    def init_render(self):
        pass
    
    def render(self, state):
        
        print("Current state:", state)
        print("Max inventory:", self.n, "\n")
    
    def get_transistion_probabilities(self, state, action):
        
        p = defaultdict(lambda : 0)
        
        num_states = len(self.get_states())
        for d in range(num_states):
            
            next_state, reward = self.step(state, action, d)
            
            p[(next_state, reward)] += 1/num_states
        
        return p
    
    def get_states(self) -> list:
        return np.arange(self.n + 1)
    
