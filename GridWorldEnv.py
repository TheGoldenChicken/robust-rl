from collections import defaultdict

class GridWorldEnv:
    
    layout = [[' ',' ',' ',  1],
              [' ','#',' ', -1],
              ['S',' ',' ',' ']]
    
    def __init__(self, layout = None) -> None:
        if(layout != None):
            self.layout = layout

    def step(self, state, action, r = None):
        # walk in the direction of the action.
        # If the action is not possible, stay in the same state
        # Actions that are states outsite the layout and the "#" state.
        next_state = (state[0] + action[0], state[1] + action[1])
        if(next_state not in self.get_states()):
            next_state = state
        
        # If the state is a number then return that number as reward
        reward = 0
        if(type(self.layout[next_state[0]][next_state[1]]) == int):
            reward = self.layout[next_state[0]][next_state[1]]
        
        return next_state, reward
    
    def get_transistion_probabilities(self, state, action):
        
        p = defaultdict(lambda : 0)
        
        next_state, reward = self.step(state, action)
            
        p[(next_state, reward)] += 1
        
        return p
        
    # Function that returns the available actions
    def A(self, state):
        return [(0,1), (0,-1), (1,0), (-1,0)]
    
    # Function that return all possible states
    def get_states(self):
        states = []
        for i in range(len(self.layout)):
            for j in range(len(self.layout[i])):
                if self.layout[i][j] != '#':
                    states.append((i,j))
        return states

    def reset(self):
        for i in range(len(self.layout)):
            for j in range(len(self.layout[i])):
                if self.layout[i][j] == 'S':
                    return (i,j)
                
        # Print error message indicating that there is no start state
        print('Error: No start state found')
        return None
        

    def init_render(self):
        pass
        #pygame.init()
        #self.display = new_display()
		
    def render(self, agent):
        pass
        #if not rendering:
        #    self.init_render()
        #self.display.update()
        
    def get_accumulative_reward(self, agent):
        return 0