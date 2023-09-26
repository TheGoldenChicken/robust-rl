from collections import defaultdict
import rl.player
import rl.env
import rl.agent
import rl.display
import rl.policy
import numpy as np
import pygame
import time

class Player(rl.player.Player):
    
    def __init__(self,env, type):
        """
        Player class for the GridWorld environment. There are two types of players:
        1. The Player (P) which there can only be one of
        2. The Enemy (E) which there can be multiple of
        """
        super().__init__(env)
        
        self.type = type
    
    def __hash__(self):
        return hash(self.position)

class Env(rl.env.DiscreteEnv):
    
    layout0 = [[[' ',' ',' ','>','G'],
                [' ','#',' ','>','G'],
                ['P',' ',' ',' ','#']],
               [[  0,  0,  0,  0, 1],
                [  0,  0,  0,  0, -1],
                [  0,  0,  0,  0, 0]]]
    
    layout1 = [[[' ',' ','E','>','G'],
                [' ','#',' ','>','G'],
                ['P',' ',' ',' ','#']],
               [[  0,  0,  0,  0, 1],
                [  0,  0,  0,  0, -1],
                [  0,  0,  0,  0, 0]]]
    
    def __init__(self, layout = None) -> None:
        self.action_set = [(0,1), (0,-1), (1,0), (-1,0), (0,0)]
        
        if(layout == None): self.layout = self.layout1
        else: self.layout = layout
        
        self.reset()
        
        self.state_size = len(self.players)*3

    def step(self, state, actions):
        action = actions[0]
        
        reward = 0
        for i, (index, x, y) in enumerate(state):
            player = self.players[index]
            if player.type == 'Player':
                if not self.is_valid(index, x, y, state, action):
                    # If at goal state, do nothing
                    if self.layout[0][y][x] == 'G':
                        return state, 0
                    raise ValueError("Invalid action")
                else:
                    new_position = (x + action[0], y + action[1])
                    
                    state = list(state)
                    state[i] = (index, new_position[0], new_position[1])
                    state = tuple(state)
                    reward += self.layout[1][new_position[1]][new_position[0]]
                break
        
        for i, (index, x, y) in enumerate(state):
            enemy = self.players[index]
            if enemy.type == 'Enemy':
                valid_actions = [action for action in self.action_set if self.is_valid(index, x, y, state, action)]
                
                if len(valid_actions) > 0:
                    action = valid_actions[np.random.randint(0, len(valid_actions))]
                    
                    new_position = (x + action[0], y + action[1])
                    
                    state = list(state)
                    state[i] = (index, new_position[0], new_position[1])
                    state = tuple(state)
                    
                    for index_, x_, y_ in state:
                        if self.players[index_].type == 'Player':
                            if (x_, y_) == (new_position[0], new_position[1]):
                                reward -= 1
                                break
        
        return state, reward
    
    def is_valid(self, index, x, y, state, action):
        player = self.players[index]
        
        if self.layout[0][y][x] == 'G':
            return False
        
        new_position = (x + action[0], y + action[1])
        
        # Check if new position is out of bounds
        if new_position[0] < 0 or new_position[0] >= len(self.layout[0][0]) or new_position[1] < 0 or new_position[1] >= len(self.layout[0]):
            return False
        
        # Check if new position is a wall
        if self.layout[0][new_position[1]][new_position[0]] == '#':
            return False
        elif player.type == 'Enemy':
            # Check if new position is the goal
            if self.layout[0][new_position[1]][new_position[0]] == 'G':
                return False
        elif player.type == 'Player': 
            # Check if the direction is restricted
            match action:
                case (0,0):
                    return True
                case (0,1):
                    if self.layout[0][y][x] in ['v','<','>']:
                        return False
                case (0,-1):
                    if self.layout[0][y][x] in ['^','<','>']:
                        return False
                case (1,0):
                    if self.layout[0][y][x] in ['^','v','<']:
                        return False
                case (-1,0):
                    if self.layout[0][y][x] in ['^','>','^']:
                        return False
        
        # Check if new position is occupied by an enemy
        for index, x , y in state:
            player_ = self.players[index]
            if player_.type == 'Enemy':
                if (x,y) == new_position:
                    return False
        
        return True
    
    def is_terminal(self, state):
        for index, x, y in state:
            player = self.players[index]
            if(player.type == 'Player'):
                if self.layout[0][y][x] == 'G':
                    return True
                for index_, x_, y_ in state:
                    enemy = self.players[index_]
                    if(enemy.type == 'Enemy'):
                        if (x,y) == (x_,y_):
                            return True
        
    # Function that returns the available actions
    def A(self, state):
        actions = []
        for index, x, y in state:
            player = self.players[index]
            if(player.type == 'Player'):
                for action in self.action_set:
                    if self.is_valid(index, x, y, state, action):
                        actions.append(action)
        
        if actions == []:
            actions.append(None)
        
        return actions
    
    # Reset state by returning initial state
    def reset(self) -> None:
        self.players = []
        state = []
        for y in range(len(self.layout[0])):
            for x in range(len(self.layout[0][y])):
                if(self.layout[0][y][x] == 'P'):
                    state.append((len(state),x,y))
                    self.players.append(Player(self, "Player"))
                elif(self.layout[0][y][x] == 'E'):
                    state.append((len(state),x,y))
                    self.players.append(Player(self, "Enemy"))
                    
        return tuple(state)
        
    def init_render(self, aps = 5, grid_scale = 100):
        self.aps = aps
        self.grid_scale = grid_scale
        
        self.width = len(self.layout[0][0])*self.grid_scale
        self.height = len(self.layout[0])*self.grid_scale

        self.display = rl.display.displayHandler(self.width, self.height)
        
    
    def render(self, agent):
        width = 5
        
        def square(x, y, element):
            gs = self.grid_scale
            if element == 'G': fill_color = (200,100,0)
            else: fill_color = (0,0,0) 
            
            self.display.draw_square((x*gs + gs/2, y*gs + gs/2), (gs, gs), fill_color, width=width)
            
            state_ = []
            for index, x_, y_ in agent.state:
                player = self.players[index]
                if player.type == "Enemy":
                    state_.append((index, x_, y_))
                elif player.type == "Player":
                    state_.append((index, x, y))
            state_ = tuple(state_)
            
            def Q_to_color(Q):
                min_ = np.min([np.min(self.layout[1]), -1])
                max_ = np.max(self.layout[1])
                
                if Q <= 0:
                    return (255 + int(255*(Q-min_)/min_), 0, 0)
                if Q > 0:
                    return (0, 255 + int(255*(Q-(max_+0.001))/(max_+0.001)), 0)
            
            # state_ = agent.state
            def up():
                color = Q_to_color(agent.Q[(state_, ((0,-1),))])
                self.display.draw_polygon([(int(x*gs), int(y*gs)),(int(x*gs+gs), int(y*gs)),(int(x*gs+gs/2), int(y*gs+gs/2))], width=width, color = color)
                self.display.draw_text(str(round(agent.Q[(state_, ((0,-1),))],2)),
                                        (x*gs + gs/2, y*gs + width),
                                        (255,255,255),
                                        align = "center-top",
                                        font_size = self.grid_scale//6)
            def right():
                color = Q_to_color(agent.Q[(state_, ((1,0),))])
                self.display.draw_polygon([(int(x*gs+gs), int(y*gs+gs)),(int(x*gs+gs), int(y*gs)), (int(x*gs+gs/2), int(y*gs+gs/2))], width=width, color = color)
                self.display.draw_text(str(round(agent.Q[(state_, ((1,0),))],2)),
                                        (x*gs + gs - width, y*gs + gs/2),
                                        (255,255,255),
                                        align = "center-right",
                                        font_size = self.grid_scale//6,
                                        angle = -90)  
            def left():
                color = Q_to_color(agent.Q[(state_, ((-1,0),))])
                self.display.draw_polygon([(int(x*gs),  int(y*gs)),(int(x*gs), int(y*gs+gs)),(int(x*gs+gs/2), int(y*gs+gs/2))], width=width, color = color)
                self.display.draw_text(str(round(agent.Q[(state_, ((-1,0),))],2)),
                                        (x*gs + width, y*gs + gs/2),
                                        (255,255,255),
                                        align = "center-left",
                                        font_size = self.grid_scale//6,
                                        angle = 90)           
            def down():
                color = Q_to_color(agent.Q[(state_, ((0,1),))])
                self.display.draw_polygon([(int(x*gs+gs), int(y*gs+gs)),(int(x*gs), int(y*gs+gs)),(int(x*gs+gs/2), int(y*gs+gs/2))], width=width, color = color)
                self.display.draw_text(str(round(agent.Q[(state_, ((0,1),))],2)),
                                        (x*gs + gs/2, y*gs + gs - width),
                                        (255,255,255),
                                        align = "center-bottom",
                                        font_size = self.grid_scale//6)
            
            if element != 'G':
                match element:
                    case '^':
                        up()
                    case 'v':
                        down()
                    case '<':
                        left()
                    case '>':
                        right()
                    case _:
                        for action in self.action_set:
                            new_position = (x + action[0], y + action[1])
                            if new_position[0] >= 0 \
                                and new_position[0] < len(self.layout[0][0]) \
                                and new_position[1] >= 0 \
                                and new_position[1] < len(self.layout[0]) \
                                and self.layout[0][new_position[1]][new_position[0]] != '#':
                                match action:
                                    case (0,1):
                                        down()
                                    case (0,-1):
                                        up()
                                    case (1,0):
                                        right()
                                    case (-1,0):
                                        left()
                                    case _: pass
            
            reward = self.layout[1][y][x]
            if reward > 0: fill_color = (0,150,0)
            elif reward < 0: fill_color = (150,0,0)
            else: fill_color = (0,0,0)
            
            self.display.draw_square((x*gs + gs/2, y*gs + gs/2), (gs/2, gs/2), fill_color, width=3)
            if(reward != 0):
                self.display.draw_text(str(reward), (x*gs + gs/2, y*gs + gs/2), (255,255,255), align = "center", font_size = int(self.grid_scale/4))

        for y, row in enumerate(self.layout[0]):
            for x, element in enumerate(row):
                if element != '#':
                    square(x, y, element)
                else:
                    self.display.draw_square((x*self.grid_scale + self.grid_scale/2, y*self.grid_scale + self.grid_scale/2), (self.grid_scale, self.grid_scale), (0,0,0))
        
        for index, x, y in agent.state:
            player = self.players[index]
            if(player.type == 'Enemy'):
                self.display.draw_sphere((x*self.grid_scale + self.grid_scale//2, y*self.grid_scale + self.grid_scale//2), self.grid_scale//3, (100,0,0), width=3)
                self.display.draw_text('E', (x*self.grid_scale + self.grid_scale//2, y*self.grid_scale + self.grid_scale//2),
                                        (255,255,255), align = "center", font_size = self.grid_scale//4)
            elif(player.type == 'Player'):
                        self.display.draw_sphere((x*self.grid_scale + self.grid_scale//2, y*self.grid_scale + self.grid_scale//2), self.grid_scale//3, (0,100,0), width=3)
                        self.display.draw_text('P', (x*self.grid_scale + self.grid_scale//2, y*self.grid_scale + self.grid_scale//2),
                                               (255,255,255), align = "center", font_size = self.grid_scale//4)
            
        
        self.display.update()
        
        # Pygame equivalent to n actions per second (aps)
        # pygame.time.wait(1000//self.aps)
        
        return self.display.eventHandler()
    
class PlayModePolicy(rl.policy.Policy):
    
    def __init__(self, env):
        super().__init__(env)
        
    def get_action(self, agent) -> bool:
        
        while(True):
            
            self.env.render(agent)
            
            key_released = self.env.display.key_released
            
            actions = self.env.A(agent.state)
            print(agent.state, actions)

            if(key_released[pygame.K_UP%512] == True): 
                if (0,-1) in actions:
                    return [(0,-1)]
            elif(key_released[pygame.K_DOWN%512] == True): 
                if (0,1) in actions:
                    return [(0,1)]
            elif(key_released[pygame.K_LEFT%512] == True):
                if (-1,0) in actions:
                    return [(-1,0)]
            elif(key_released[pygame.K_RIGHT%512] == True):
                if (1,0) in actions:
                    return [(1,0)]
            elif(key_released[pygame.K_TAB%512] == True):
                return [(0,0)]