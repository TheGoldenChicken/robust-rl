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
        cost = max(0,(self.h*(state + actions[0] - demand)))
        
        # Lost sales penalty
        cost += max(0,(self.p*(demand - state - actions[0])))
        
        # Fixed ordering cost
        # cost += self.k*actions[0]
        if(actions[0] > 0): cost += self.k
        
        next_state = state + actions[0] - min(demand, state + actions[0])
            
        reward = cost
        
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
        
    def init_render(self, grid_scale = 50):
        self.grid_scale = grid_scale
        
        self.width = (self.n+1)*self.grid_scale + self.grid_scale
        self.height = (self.n+1)*self.grid_scale

        self.display = rl.display.displayHandler(self.width, self.height)
        
    
    def render(self, agent):
        gs = self.grid_scale
        min_q = 1e10
        max_q = 0
        
        color_r = [(i, 0, 0) for i in np.linspace(255, 0, int(self.n/2 + 1))]
        color_g = [(0, i, 0) for i in np.linspace(0, 255, int(self.n/2 + 1))]
        color = color_r[:-1] + color_g
        for a in range(self.n+1):
            for s in range(self.n+1):
                if(a <= self.n-s):
                    q = agent.Q[(s,a)]
                    
                    if q < min_q: min_q = q
                    elif q > max_q: max_q = q
        
        for a in range(self.n+1):
            for s in range(self.n+1):
                if(a <= self.n-s):
                    q = agent.Q[(s,a)]
                    
                    # if q < min_q: min_q = q
                    # elif q > max_q: max_q = q
                    color_index = int((q-min_q)/((max_q - min_q +1e-05) / len(color)))  
                    color_index = min(color_index, self.n)
                    self.display.draw_square((a*gs + gs/2, s*gs + gs/2), (gs, gs), color[color_index])
                    self.display.draw_text(str(round(q, 2)), (a*gs + gs/2, s*gs + gs/2), (255,255,255), align="center")
                else:
                    self.display.draw_square((a*gs + gs/2, s*gs + gs/2), (gs, gs), (0,0,0))
        
        q_range = np.linspace(min_q, max_q, self.n+1)
        
        for s in range(self.n+1):
            self.display.draw_square(((self.n+1)*gs + gs/2, s*gs + gs/2), (gs, gs), color[s], width = 10)
            self.display.draw_text(str(round(q_range[s], 2)), ((self.n+1)*gs + gs/2, s*gs + gs/2), (255,255,255), align="center")

        self.display.update()
        
        
        return self.display.eventHandler()
    
    def is_terminal(self, state) -> bool:
        return False
    
    def get_transistion_probabilities(self, state, action):
        
        p = defaultdict(lambda : 0)
        
        num_states = len(self.get_states())
        for d in range(num_states):
            
            next_state, reward = self.step(state, action, d)
            
            p[(next_state, reward)] += 1/num_states
        
        return p
    
    def get_states(self) -> list:
        return np.arange(self.n + 1)
    
