import pygame
import numpy as np

import rl.env, rl.player, rl.agent, rl.display
from scipy.stats import multivariate_normal

        
class Player(rl.player.Player):
    
    def __init__(self, env, speed = 0.2, noise_std = 0.1) -> None:
        super().__init__(env)
        
        self.speed = speed
        self.noise_var = noise_std**2
        
class Env(rl.env.ContinuousEnv):
    
    ### Layout ###
    # ' ' = Empty
    # 'S' = Start
    # 'E' = End
    # '#' = Cliff
    layout = [[' ',' ',' ',' ',' ',' ',' '],
              [' ',' ',' ',' ',' ',' ',' '],
              [' ','S',' ',' ',' ','E',' '],
              [' ',' ',' ','#','#','#','#']]
    players = []
    
    def __init__(self, layout = None, playerOptions = []):
        super().__init__()
        
        if(layout != None): self.layout = layout
        
        self.cliff_pos = []
        
        # Find the goal and start positions
        for i in range(len(self.layout)):
            for j in range(len(self.layout[i])):
                if(self.layout[i][j] == 'S'):
                    player = Player(env = self, *playerOptions)
                    self.players.append(player)
                if(self.layout[i][j] == 'E'): self.goal = (j, i)
                if(self.layout[i][i] == '#'): self.cliff_pos.append((j, i))
        
        self.state_size = len(self.players)*2
        
    def step(self, state, actions):
        
        state_ = ()
        reward = 0
        
        # sample from 2d normal distribution
        for i, (action, player) in enumerate(zip(actions, self.players)):
            noise = multivariate_normal.rvs([0,0], [[player.noise_var,0],
                                                    [0,player.noise_var]])
            
            new_pos = (state[i*2] + action[0]*player.speed + noise[0],
                       state[i*2+1] + action[1]*player.speed + noise[1])
            
            reward -= np.sqrt((new_pos[0]-self.goal[0])**2 + (new_pos[1]-self.goal[1])**2)
            
            for cliff in self.cliff_pos:
                if((new_pos[0] - cliff[0])**2 < 1**2 and (new_pos[1] - cliff[1])**2 < 1**2):
                    reward -= 1000
                    break
            
            state_ += new_pos
        
        return state_, reward
    
    def A(self, state):
        A = ()
        for _ in range(len(self.players)):
            A += ([(-1,0), (0,1), (1,0), (0,-1), (0,0)])
        return A
    
    def reset(self):
        state = []
        for player in self.players:
            for i in range(len(self.layout)):
                for j in range(len(self.layout[i])):
                    if(self.layout[i][j] == 'S'): state += [j,i]
        return state
        
    def init_render(self, grid_scale = 100):
        self.width = len(self.layout[0]) * grid_scale
        self.height = len(self.layout) * grid_scale
        
        self.grid_scale = grid_scale
        self.display = rl.display.displayHandler(self.width, self.height)
        
    def render(self, state):
        gs = self.grid_scale
        for i in range(len(self.layout)):
            for j in range(len(self.layout[i])):
                if(self.layout[i][j] == ' '): self.display.draw_square((j*gs + gs/2, i*gs + gs/2), (95,95), (0,0,0))
                elif(self.layout[i][j] == 'S'): self.display.draw_square((j*gs + gs/2, i*gs + gs/2), (95,95), (0,255,0))
                elif(self.layout[i][j] == 'E'): self.display.draw_square((j*gs + gs/2, i*gs + gs/2), (95,95), (255,0,0))
                elif(self.layout[i][j] == '#'): self.display.draw_square((j*gs + gs/2, i*gs + gs/2), (95,95), (0,0,255))
                
        for i in range(len(self.players)):
            self.display.draw_image('cliff_car\Car.png', (state[i*2]*gs + gs/2, state[i*2+1]*gs + gs/2), (100,100))

        
        self.display.update(backgroundColor = ( 50, 50, 50))
        
        return self.display.eventHandler()
        
class PlayMode(rl.agent.ShallowAgent):
    
    def __init__(self, env) -> None:
        super().__init__(env)
        self.state = self.env.reset()
        
    def next(self) -> bool:
        
        key_released = self.env.display.key_released

        if(key_released[pygame.K_UP%512] == True):
            state_, reward = self.env.step(self.state,[(0,-1)])
            self.state = state_
        elif(key_released[pygame.K_DOWN%512] == True):
            state_, reward = self.env.step(self.state,[(0,1)])
            self.state = state_
        elif(key_released[pygame.K_LEFT%512] == True):
            state_, reward = self.env.step(self.state,[(-1,0)])
            self.state = state_
        elif(key_released[pygame.K_RIGHT%512] == True):
            state_, reward = self.env.step(self.state,[(1,0)])
            self.state = state_
        elif(key_released[pygame.K_TAB%512] == True):
            state_, reward = self.env.step(self.state,[(0,0)])
            self.state = state_
        
        if(key_released[pygame.K_ESCAPE%512]):
            self.env.display.close()
            return True
        
        return False