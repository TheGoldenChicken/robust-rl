import numpy as np
import pygame
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from torch.distributions import MultivariateNormal
import torch

def drawImage(display, path, center, scale=None, angle=None):
    img = pygame.image.load(path)
    if (scale != None):
        img = pygame.transform.scale(img, scale)
    if (angle != None):
        img = pygame.transform.rotate(img, angle)

    display.blit(img, center)

class CliffCar:

    translate_action = {0 : np.array([0,0]),
                        1 : np.array([1,0]),
                        2 : np.array([0,1]),
                        3 : np.array([-1,0]),
                        4 : np.array([0,-1])}
    CLIFF_PENALTY = 2
    ACTION_DIM = len(translate_action)
    OBS_DIM = 2
    START_POSITION = np.array([1,15], dtype=np.float32) # has to be discrete for discrete agent to work
    GOAL_POSITION = np.array([15,1], dtype=np.float32) # has to be discrete for discrete agent to work
    BOUNDS = np.array([-10,-2,25,20]) # has to be discrete for discrete agent to work
    CLIFF_HEIGHT = 0
    SPEED = 0.5

    def __init__(self, noise_mean = 0, noise_var = 0, mode = "abrupt",
                 radial_basis_dist = 1, radial_basis_var = 2, torch_mode = True, **kwargs):

        # Car settings
        self.position = self.START_POSITION

        self.noise_mean = noise_mean
        self.noise_var = noise_var
        
        # World settings
        self.mode = mode # "abrupt" or "penalty" - How the cliff is handled
        self.max_goal_distance = self.get_max_goal_distance()
        
        # Simulation settings
        self.max_duration = 500
        self.frame = 0
        
        self.max_min = [[self.BOUNDS[2], self.BOUNDS[3]], [self.BOUNDS[0], self.BOUNDS[1]]]
        
        self.radial_basis_dist = radial_basis_dist
        self.radial_basis_var = radial_basis_var
        
        # Rendering
        self.rendering = False
        
        # For slightly faster random samples (saves around .5 seconds per 20000 samples) - It adds up?
        self.torch_mode = torch_mode
        if self.torch_mode and self.noise_var != 0:
            mean = torch.tensor([self.noise_mean,self.noise_mean],dtype = torch.float32)
            var = torch.tensor([[self.noise_var,0],[0,self.noise_var]],dtype = torch.float32)
            self.mult_normal = MultivariateNormal(mean, var)
        
    def clamp_position(self, position):
        position[0] = min(max(position[0], self.BOUNDS[0]), self.BOUNDS[2])
        position[1] = min(max(position[1], self.BOUNDS[1]), self.BOUNDS[3])
        
        return position
    
    def get_max_goal_distance(self):
        d1 = np.sum((self.GOAL_POSITION - self.BOUNDS[0:2])**2)
        d2 = np.sum((self.GOAL_POSITION - self.BOUNDS[2:4])**2)
        d3 = np.sum((self.GOAL_POSITION - self.BOUNDS[[0,3]])**2)
        d4 = np.sum((self.GOAL_POSITION - self.BOUNDS[[2,1]])**2)
        return max(d1,d2,d3,d4)
    
    def step(self, action):
        
        self.position = self.result(self.position, action)

        reward, done = self.reward_fn(self.position)
        
        if self.frame >= self.max_duration:
            done = True
            reward = 0

        if self.rendering:
            self.render()
            
        self.frame += 1

        return self.position, reward, done, 'dummy' # Final value is dummy for working with gym envs
    
    def result(self, position, action):
        noise = np.zeros(2)
        mean = np.array([self.noise_mean,self.noise_mean])
        if(self.noise_var != 0):
            if self.torch_mode:
                noise = self.mult_normal.sample((1,)).squeeze().numpy()
            else:
                mean = np.array([self.noise_mean,self.noise_mean])
                var = np.array([[self.noise_var,0],[0,self.noise_var]])
                noise = np.random.multivariate_normal(mean, var)

        action_dir = self.translate_action[action]

        # Calculate the new position and clamp it
        position = position + self.SPEED * action_dir + noise
        position = self.clamp_position(position)
        
        return position

    def reward_fn(self, position):

        # Calculate distance to cliff (as reward)
        reward = (position[0] - self.GOAL_POSITION[0])**2 + (position[1] - self.GOAL_POSITION[1])**2
        reward = reward/self.max_goal_distance # Normalize distance
        
        # If the car is below the cliff 
        if position[1] < self.CLIFF_HEIGHT:
            if self.mode == "abrupt":
                # End the game immediatly
                return 1, True
            elif self.mode == "penalty":
                # Give a penalty for being in the cliff. Don't end the game
                return reward + self.CLIFF_PENALTY, False
            else: raise ValueError("Invalid mode. Should either be 'abrupt' or 'penalty'")
        
        return reward, False
    
    def reset(self):
        self.position = self.START_POSITION
        self.frame = 0 # Reset the timer (pretty important)
        return self.position

    def init_render(self, scale = 25):
        self.rendering = True
        self.scale = scale
        self.clock = pygame.time.Clock()
        self.render_width = abs(self.BOUNDS[2] - self.BOUNDS[0]) * scale
        self.render_height = abs(self.BOUNDS[3] - self.BOUNDS[1]) * scale
        self.display = pygame.display.set_mode((self.render_width, self.render_height))
        
        self.show_grid = False # Show grid lines (scale): keycode g

    def render(self):

        def tfx(x):
            return (abs(self.BOUNDS[0]) + x) * self.scale

        def tfy(y):
            return (abs(self.BOUNDS[1]) + y) * self.scale

        # Transform the coordinates of the game logic to the coordinates of the display
        def tf2(coord):
            x = tfx(coord[0])
            y = tfy(coord[1])
            return (x, y)
        
        def flip_rect(rect):
            return (rect[0], self.render_height - rect[1] - rect[3], rect[2], rect[3])

        if not self.rendering:
            self.init_render()


        self.clock.tick(60) # frame_rate = 60
        pygame.display.flip() # Why this?
        self.display.fill((255,255,255)) # Fill with background color - Do before other stuff
        
        # The cliff
        
        cliff_rect = flip_rect((0, 0, self.render_width, tfy(self.CLIFF_HEIGHT)))
        tf_goal = tf2(self.GOAL_POSITION)
        tf_car = tf2(self.position)
        goal_rect = flip_rect((tf_goal[0] - 10, tf_goal[1] - 10, 20, 20))
        car_rect = flip_rect((tf_car[0] - 20, tf_car[1] - 20, 40, 40))
        
        pygame.draw.rect(self.display, (0, 0, 255), cliff_rect)
        
        if(self.show_grid):
            for x in range(0, self.render_width, self.scale):
                pygame.draw.line(self.display, (20,20,20), (x, 0), (x, self.render_height))
            for y in range(0, self.render_height, self.scale):
                pygame.draw.line(self.display, (20,20,20), (0, y), (self.render_width, y))
        
        pygame.draw.rect(self.display, (0,255,0), goal_rect)
        pygame.draw.rect(self.display, (255,0,0), car_rect)
        
class CliffCarDiscrete(CliffCar):

    def __init__(self, noise_mean = [0,0], noise_var = None, mode = "abrupt"):
        super().__init__(noise_mean, noise_var, mode)
    
    def result(self, position, action):
        position = super().result(position, action)
        
        # Round to nearest integer
        position = np.round(position,0)
        
        return position
    
    def get_states(self):
        states = []
        for x in range(self.BOUNDS[0], self.BOUNDS[2]+1):
            for y in range(self.BOUNDS[1], self.BOUNDS[3]+1):
                states.append(np.array([x,y]))
        return states
        
    def transition_prob(self, position, action, sim = 50):
        # Returns a dictionary of probabilities where the keys are (s', reward)
        
        p = defaultdict(float)
        
        for i in range(sim):
            next_position = self.result(position, action)
            reward = self.reward_fn(next_position)[0]
            p[tuple(next_position), reward] += 1/sim
        
        return p

    
car = CliffCar()

if __name__ == "__main__":
    env = CliffCarDiscrete(noise_var = np.array([[0.5,0],[0,0.5]]))
    env.init_render()

    action = 0
    done = False
    episode_reward = 0
    while True:

        while not done:
            
            action_taken = False
            
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        pygame.quit()
                        quit()

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        action = 3
                    elif event.key == pygame.K_d:
                        action = 1
                    elif event.key == pygame.K_w:
                        action = 2
                    elif event.key == pygame.K_s:
                        action = 4
                    elif event.key == pygame.K_g:
                        env.show_grid = not env.show_grid
                        
                    action_taken = True
            
            if action_taken:
                print(env.reward_fn(env.position)[0])
                next_state, reward, done, _ = env.step(action)

                episode_reward += reward
            env.render()
            
        done = False
        state = env.reset()
        print(episode_reward)

