from collections import defaultdict
import numpy as np
from cliff_car_env import CliffCarDiscrete
import pygame

class ValueIteration:

    def __init__(self):
        self.Q = defaultdict(float)
        self.V = defaultdict(float)
        
        self.rendering = False

    def __call__(self, env, gamma = 0.9, tol = 1e-3, render_frequency = 1):

        Delta = float('inf')
        states = env.get_states()
        
        min_V = 0
        max_V = 255
        min_Q = 0
        max_Q = 255
        
        pi = defaultdict(lambda : tuple([0,0]))
        
        counter = 0
        while(Delta > tol):
            Q_ = defaultdict(lambda : 0)
            Delta = 0
            
            # State value iteration
            for state in states:
                for action in env.ACTIONS:
                    Q_old = self.Q[tuple(state), tuple(action)]
                    for (state_, reward), p in env.transition_prob(state, action, sim = 250).items():
                        Q_max = max([self.Q[tuple(state_), tuple(action_)] for action_ in env.ACTIONS])                        
                        Q_[tuple(state), tuple(action)] += p * (reward + gamma * Q_max)
                        
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                                    pygame.quit()
                                    quit()

                            if event.type == pygame.KEYUP:
                                if event.key == pygame.K_1:
                                    self.mode = "state_values"
                                    self.draw_state_values(env, pi, min_V, max_V, min_Q, max_Q) 
                                elif event.key == pygame.K_2:
                                    self.mode = "action_values"
                                    self.draw_state_values(env, pi, min_V, max_V, min_Q, max_Q) 
                                elif event.key == pygame.K_3:
                                    self.mode = "policy"
                                    self.draw_state_values(env, pi, min_V, max_V, min_Q, max_Q) 

                    Delta = max(Delta, abs(Q_old - Q_[tuple(state), tuple(action)]))
                    
                    
            
            # Update the value functions
            print("Delta: " + str(Delta))
            self.Q = Q_
            
            
            
            if counter % render_frequency == 0:
                # Convert Q values to policy values
                
                min_V = float('inf')
                max_V = float('-inf')
                min_Q = float('inf')
                max_Q = float('-inf')
                
                pi = defaultdict(lambda : tuple([0,0]))
                for state in states:
                    Q_max = float('-inf')
                    for action in env.ACTIONS:
                        if Q_max < self.Q[tuple(state), tuple(action)]:
                            Q_max = self.Q[tuple(state), tuple(action)]
                            best_action = action
                            
                        Q_new = self.Q[tuple(state), tuple(action)]    
                        min_Q = min(min_Q, Q_new)
                        max_Q = max(max_Q, Q_new)
                        
                    self.V[tuple(state)] = Q_max
                    
                    pi[tuple(state)] = tuple(best_action)
                    
                    min_V = min(min_V, Q_max)
                    max_V = max(max_V, Q_max)
                    
                print("min_V: " + str(min_V), "max_V: " + str(max_V))
                
                self.draw_state_values(env, pi, min_V, max_V, min_Q, max_Q) 
            
            counter += 1
        
        # Convert Q values to policy values
        pi = defaultdict(float)
        for state in states:
            Q_max = float('-inf')
            best_action = None
            for action in env.ACTIONS:
                if Q_max < self.Q[tuple(state), tuple(action)]:
                    Q_max = self.Q[tuple(state), tuple(action)]
                    best_action = action
            self.V[tuple(state)] = Q_max
            pi[tuple(state)] = tuple(best_action)
        
        # Save pygame screen as image
        for m in "state_values", "action_values", "policy":
            self.mode = m
            self.draw_state_values(env, pi, min_V, max_V, min_Q, max_Q)
            d_val = str(round(Delta,3)).replace('.','_')
            pygame.image.save(self.display, rf"post_bachelor\cliff_car\value_iter_{self.mode}_noise-{env.noise_var[0][0]}_tol-{d_val}.png")
            #f"vi_policy:{self.mode}_delta:{str(round(Delta,3)).replace('.',',')}.png"1
        
        return pi
    
    def init_render(self, scale = 25):
        self.rendering = True
        self.scale = scale
        self.clock = pygame.time.Clock()
        self.render_width = abs(env.BOUNDS[2] - env.BOUNDS[0]) * scale
        self.render_height = abs(env.BOUNDS[3] - env.BOUNDS[1]) * scale
        self.display = pygame.display.set_mode((self.render_width, self.render_height))
        self.mode = "action_values" # "state_values", "action_values", or "policy"
        
    
    def draw_state_values(self, env, pi, min_V, max_V, min_Q, max_Q):
        
        def value_to_color(value, minimum, maximum):
            value = (value - minimum)/(maximum - minimum)
            
            color = (int((value < 0.5)*(0.5-value)*255*2), int((value > 0.5)*(value-0.5)*255*2), 0)
            
            return color
        
        def draw_action_values(offset, state):
            
            offset = (offset[0], self.render_height - offset[1] - self.scale)
            scale = self.scale
            diag = scale*0.33
            
            color = value_to_color(self.Q[state,tuple([0,1])], min_Q, max_Q)
            pygame.draw.polygon(self.display, color, [(offset[0], offset[1]),
                                                        (offset[0] + scale, offset[1]),
                                                        (offset[0] + scale - diag, offset[1] + diag),
                                                        (offset[0] + diag, offset[1] + diag)])
            color = value_to_color(self.Q[state,tuple([1,0])], min_Q, max_Q)
            pygame.draw.polygon(self.display, color, [(offset[0] + scale, offset[1]),
                                                        (offset[0] + scale, offset[1] + scale),
                                                        (offset[0] + scale - diag, offset[1] + scale - diag),
                                                        (offset[0] + scale - diag, offset[1] + diag)])
            color = value_to_color(self.Q[state,tuple([0,-1])], min_Q, max_Q)
            pygame.draw.polygon(self.display, color, [(offset[0] + scale, offset[1] + scale),
                                                        (offset[0], offset[1] + scale),
                                                        (offset[0] + diag, offset[1] + scale - diag),
                                                        (offset[0] + scale - diag, offset[1] + scale - diag)])
            color = value_to_color(self.Q[state,tuple([-1,0])], min_Q, max_Q)
            pygame.draw.polygon(self.display, color, [(offset[0], offset[1] + scale),
                                                        (offset[0], offset[1]),
                                                        (offset[0] + diag, offset[1] + diag),
                                                        (offset[0] + diag, offset[1] + scale - diag)])
            color = value_to_color(self.Q[state,tuple([0,0])], min_Q, max_Q)
            pygame.draw.rect(self.display, color, (offset[0] + diag, offset[1] + diag, scale - 2*diag, scale - 2*diag))
            
        def show_policy(offset, state):
            
            offset = (offset[0], self.render_height - offset[1] - self.scale)
            scale = self.scale
            diag = scale*0.33
            
            if(pi[state] == tuple([0,1])):
                color = value_to_color(self.Q[state,tuple([0,1])], min_Q, max_Q)
                pygame.draw.polygon(self.display, color, [(offset[0], offset[1]),
                                                        (offset[0] + scale, offset[1]),
                                                        (offset[0] + scale - diag, offset[1] + diag),
                                                        (offset[0] + diag, offset[1] + diag)])
            elif(pi[state] == tuple([1,0])):
                color = value_to_color(self.Q[state,tuple([1,0])], min_Q, max_Q)
                pygame.draw.polygon(self.display, color, [(offset[0] + scale, offset[1]),
                                                        (offset[0] + scale, offset[1] + scale),
                                                        (offset[0] + scale - diag, offset[1] + scale - diag),
                                                        (offset[0] + scale - diag, offset[1] + diag)])
            elif(pi[state] == tuple([0,-1])):
                color = value_to_color(self.Q[state,tuple([0,-1])], min_Q, max_Q)
                pygame.draw.polygon(self.display, color, [(offset[0] + scale, offset[1] + scale),
                                                        (offset[0], offset[1] + scale),
                                                        (offset[0] + diag, offset[1] + scale - diag),
                                                        (offset[0] + scale - diag, offset[1] + scale - diag)])
            elif(pi[state] == tuple([-1,0])):
                color = value_to_color(self.Q[state,tuple([-1,0])], min_Q, max_Q)
                pygame.draw.polygon(self.display, color, [(offset[0], offset[1] + scale),
                                                        (offset[0], offset[1]),
                                                        (offset[0] + diag, offset[1] + diag),
                                                        (offset[0] + diag, offset[1] + scale - diag)])
            else:
                color = value_to_color(self.Q[state,tuple([0,0])], min_Q, max_Q)
                pygame.draw.rect(self.display, color, (offset[0] + diag, offset[1] + diag, scale - 2*diag, scale - 2*diag))
            
            
        
        
                        
        if not self.rendering:
            self.init_render()
        
        self.clock.tick(60) # frame_rate = 60
        pygame.display.flip() # Why this?
        self.display.fill((0,0,0)) # Fill with background color - Do before other stuff
        
        def tfx(x):
            return (abs(env.BOUNDS[0]) + x) * self.scale

        def tfy(y):
            return (abs(env.BOUNDS[1]) + y) * self.scale

        # Transform the coordinates of the game logic to the coordinates of the display
        def tf2(coord):
            x = tfx(coord[0])
            y = tfy(coord[1])
            return (x, y)
        
        def flip_rect(rect):
            return (rect[0], self.render_height - rect[1] - rect[3], rect[2], rect[3])
        
        green_max = 255
        green_min = 0
        
        for x in range(*env.BOUNDS[[0,2]]):
            for y in range(*env.BOUNDS[[1,3]]):
                
                
                if self.mode == "state_values":
                    # Linear map from [min_V, max_V] to [0,255]
                    color = value_to_color(self.V[tuple([x,y])], min_V, max_V)
                    pygame.draw.rect(self.display, color, flip_rect((tf2((x,y))[0], tf2((x,y))[1], 25, 25)))
                elif self.mode == "action_values":
                    draw_action_values(tf2((x,y)), tuple([x,y]))
                elif self.mode == "policy":
                    show_policy(tf2((x,y)), tuple([x,y]))
        
        # print("green_max: " + str(green_max), "green_min: " + str(green_min))
                
        # Drawing grid lines
        for x in range(0, self.render_width, self.scale):
            pygame.draw.line(self.display, (50,50,50), (x, 0), (x, self.render_height))
        for y in range(0, self.render_height, self.scale):
            pygame.draw.line(self.display, (50,50,50), (0, y), (self.render_width, y))
            
        # Draw the cliff as a blue line
        pygame.draw.line(self.display, (100, 100, 255), (0, self.render_height - tfy(env.CLIFF_HEIGHT)), (self.render_width, self.render_height - tfy(env.CLIFF_HEIGHT)), width = 2)
        
        # Draw the goal as a circle outline
        pygame.draw.circle(self.display, (100, 100, 255), (tfx(env.GOAL_POSITION[0]), self.render_height - tfy(env.GOAL_POSITION[1])), radius = 10, width = 2)


for i in [0,0.1,0.5,1,1.5]:    
    env = CliffCarDiscrete(noise_var = np.array([[i,0],[0,i]]), mode = "penalty")

    value_iteration = ValueIteration()
    value_iteration.init_render()


    pi = value_iteration(env, tol = 0.9)

# Save pygame screen as an image
