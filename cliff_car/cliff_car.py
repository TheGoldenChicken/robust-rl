import pygame
import numpy as np
import rl.display
from scipy.stats import multivariate_normal

        

class Env():
    
    ### Layout ###
    # ' ' = Empty
    # 'S' = Start
    # 'E' = End
    # '#' = Cliff
    layout = [[' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
              [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
              [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
              [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
              [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
              [' ',' ','S',' ',' ',' ',' ',' ','E',' ',' '],
              ['#','#','#','#','#','#','#','#','#','#','#']]
    
    START_OFFSET = [0,0]

    def __init__(self, layout = None, speed = 0.3, max_duration = 100,
                 noise_mean = [0,0], noise_cov = [[0.01,0],[0,0.01]], render = False) -> None:
        
        self.speed = speed
        self.noise_mean = noise_mean
        self.noise_cov = noise_cov

        self.frame = 0
        self.max_duration = max_duration
        
        if(layout != None): self.layout = layout
        
        self.cliff_pos = []
        
        # Find the goal and start positions
        for i in range(len(self.layout)):
            for j in range(len(self.layout[i])):
                if(self.layout[i][j] == 'S'): self.state = [j + self.START_OFFSET[0], i + self.START_OFFSET[1]]
                if(self.layout[i][j] == 'E'): self.goal = [j, i]
                if(self.layout[i][i] == '#'): self.cliff_pos.append([j, i])

        if render: self.init_render()
        
    def step(self, action):
        self.frame += 1
        
        noise = multivariate_normal.rvs(self.noise_mean, self.noise_cov)
        
        self.state = (self.state[0] + action[0]*self.speed + noise[0],
                    self.state[1] + action[1]*self.speed + noise[1])
        
        reward = np.sqrt((self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2)
        
        done = False

        # If driving off cliff
        for cliff in self.cliff_pos:
            if((self.state[0] - cliff[0])**2 < 1**2 and (self.state[1] - cliff[1])**2 < 1**2):
                reward = 0
                done = True
        
        # If reached goals
        if self.frame >= self.max_duration:
            done = True
            reward = 0

        if self.render: self.render()
        
        return self.state, reward, done, 'derp'
    
    def A(self):
        return np.array([[-1,0], [0,1], [1,0], [0,-1], [0,0]])
    
    def reset(self):
        for i in range(len(self.layout)):
            for j in range(len(self.layout[i])):
                if(self.layout[i][j] == 'S'): self.state = [j + self.START_OFFSET[0],i + self.START_OFFSET[1]]
        self.frame = 0
        return self.state
        
    def init_render(self, grid_scale = 30):
        self.width = len(self.layout[0]) * grid_scale
        self.height = len(self.layout) * grid_scale
        
        self.grid_scale = grid_scale
        self.display = rl.display.displayHandler(self.width, self.height)
        
    def render(self):
        gs = self.grid_scale
        for i in range(len(self.layout)):
            for j in range(len(self.layout[i])):
                if(self.layout[i][j] == ' '): self.display.draw_square((j*gs + gs/2, i*gs + gs/2), (int(gs*0.95),int(gs*0.95)), (0,0,0))
                elif(self.layout[i][j] == 'S'): self.display.draw_square((j*gs + gs/2, i*gs + gs/2), (int(gs*0.95),int(gs*0.95)), (0,255,0))
                elif(self.layout[i][j] == 'E'): self.display.draw_square((j*gs + gs/2, i*gs + gs/2), (int(gs*0.95),int(gs*0.95)), (255,0,0))
                elif(self.layout[i][j] == '#'): self.display.draw_square((j*gs + gs/2, i*gs + gs/2), (int(gs*0.95),int(gs*0.95)), (0,0,255))
                
        self.display.draw_image('cliff_car\Car.png', (self.state[0]*gs + gs/2, self.state[1]*gs + gs/2), (gs,gs))

        self.display.update(backgroundColor = ( 50, 50, 50))
        
        return self.display.eventHandler()
        
class PlayMode():
    
    def __init__(self, env) -> None:
        self.env = env
        self.visited_states = []
        self.collected_rewards = []
        
    def next(self) -> bool:
        
        assert self.env.display != None, "Render not initialized. Required for PlayMode"

        done = False

        key_released = self.env.display.key_released

        if(key_released[pygame.K_UP%512] == True):
            state_, reward, done, _ = self.env.step([0,-1])
            self.visited_states.append(state_)
            self.collected_rewards.append(reward)
        elif(key_released[pygame.K_DOWN%512] == True):
            state_, reward, done, _ = self.env.step([0,1])
            self.visited_states.append(state_)
            self.collected_rewards.append(reward)
        elif(key_released[pygame.K_LEFT%512] == True):
            state_, reward, done, _ = self.env.step([-1,0])
            self.visited_states.append(state_)
            self.collected_rewards.append(reward)
        elif(key_released[pygame.K_RIGHT%512] == True):
            state_, reward, done, _ = self.env.step([1,0])
            self.visited_states.append(state_)
            self.collected_rewards.append(reward)
        elif(key_released[pygame.K_TAB%512] == True):
            state_, reward, done, _ = self.env.step(self.env.state,[(0,0)])
            self.visited_states.append(state_)
            self.collected_rewards.append(reward)   
    
        if(key_released[pygame.K_ESCAPE%512]):
            self.env.display.close()
            return True
        
        if(done): return self.env.reset()
        
        self.env.render()

        return False