import pygame
import numpy as np
import scipy.stats

import display

width = 800
height = 600
backgroundColor = (255,255,255)


class Player:

    def __init__(self):
        self.position = [250, 250]
        self.angle = 0 # Angle - Radians, pls

        self.velocity = np.array([0,0], dtype=np.float64)
        self.angle_speed = 0 # Deg/second right now - cuz im stupid

        self.accel = 1
        self.angle_accel = 0.4
        self.mass = 1 # Mass
        self.k = 0.1 #Friction coeffcient

        # Wind - Move to env
        self.wind_direction_change = [0, 1] # Normal distribution change
        self.wind_force_change = [0, 1] # Normal distribution change


    def update_position(self, time_discret=1, radians = True):
        """

        :param int time_discret: int time-discretisation factor, divides all position updates by that factor
        :param radians: unused, TODO: implement plox
        :return:
        """
        if not radians:
            print("Please use radians")
            pass

        self.position += self.velocity / time_discret
        self.angle += self.angle_speed / time_discret

        # self.position += [np.cos(self.angle*np.pi/180) * self.speed,
        #                   np.sin(self.angle*np.pi/180) * self.speed]
        #
        # self.angle += self.angle_speed

    def update_velocity(self, actions: list, wind, wind_direction, time_discret=1):
        """
        Update angle speed and 'regular' speed
        :param int time_discret: time-discretisation factor, divides all accelerations by that factor
        :return:
        """

        x_accel = ((-self.k * self.velocity[0]
                  + actions[0] * self.accel * np.cos(self.angle*np.pi/180)
                  + wind * np.cos(wind_direction*np.pi/180)) / self.mass) / time_discret

        y_accel = ((-self.k * self.velocity[1]
                   + actions[0] * self.accel * np.sin(self.angle * np.pi / 180)
                   + wind * np.sin(wind_direction*np.pi/180)) / self.mass) / time_discret

        angle_accel = ((-self.k * self.angle_speed + actions[1] * self.angle_accel) / self.mass) / time_discret

        self.velocity += [x_accel / time_discret, y_accel / time_discret]
        self.angle_speed += angle_accel / time_discret

        # f_forward = -self.k * self.speed**2\
        #             + actions[0] * self.accel
        # self.accel = f_forward/self.mass
        #
        # f_angle = -self.k * self.angle_speed\
        #           + actions[1] * self.angle_accel
        # self.angle_accel = f_angle/self.mass
        #




class CliffCar:

    def __init__(self):
        pygame.init()
        self.rendering = False
        self.clock = pygame.time.Clock()

        self.discret_factor = 2 # Time discretitaion factor
        self.turn_reward, self.accel_reward = 0.01, 0.001

        self.width = 660
        self.height = 660
        self.players = [Player()]

        self.grid_scale = 80 # A 'block' every 20 units to more easily differentiate the field ## 10 too small for width=5
        self.init_render()

        self.state = [a for i in range(len(self.players)) for a in
                      [self.players[i].position[0], self.players[i].position[1], self.players[i].angle,
                      self.players[i].velocity[0], self.players[i].velocity[1], self.players[i].angle_speed]]

        #self.state = [[self.players[i].position[0], self.players[i].position[1], self.players[i].angle,
        #              self.players[i].velocity[0], self.players[i].velocity[1], self.players[i].angle_speed]
        #for i in range(len(self.players))]
        pass

    def init_render(self):
        self.display = pygame.display.set_mode((self.width, self.height))
        self.rendering = True

        pass

    def get_wind(self, mode='whack'):

        if mode=='whack':
            wind = scipy.stats.norm.rvs(loc=0, scale=2)
            wind_direction = scipy.stats.norm.rvs(loc=0, scale=360)
        return wind, wind_direction

    def render(self):

        # if not self.rendering:
        #     self.init_render()

        self.clock.tick(60)
        pygame.display.flip() # Why this?
        self.display.fill(backgroundColor) # Fill with background color - Do before other stuff

        # Make grid of *GRAY* Hollow Boxes for easier navigation
        for x in range(-self.grid_scale, self.width, self.grid_scale):
            for y in range(-self.grid_scale, self.height, self.grid_scale):

                pygame.draw.rect(self.display, (125,125,125),
                                 (x + self.grid_scale / 2, y + self.grid_scale / 2, self.grid_scale, self.grid_scale),
                                 width=5)
        # Draw the players
        for player in self.players:
            display.drawImage(self.display,path="Car.png" ,center=(player.position[0], player.position[1]),
                              scale=(self.grid_scale/2,self.grid_scale/2),angle=-player.angle)
            display.drawImage(self.display,path="WindArrow.png" ,center=(player.position[0], player.position[1]),
                              scale=(self.grid_scale/1.5,self.grid_scale/1.5),angle=-player.angle+45)

            #pygame.draw.rect(self.display, (255,0,0),
            #                 (player.position[0], player.position[1], 50, 50))

        # for x in range(0, int(width / grid_scale)):
        #     for y in [int(height / grid_scale) - 2, int(height / grid_scale) - 1]:
        #         render.drawImage("WaterTile.png", (x * grid_scale, y * grid_scale), (grid_scale, grid_scale))
        #pygame.Rect((x + self.grid_scale / 2, y + self.grid_scale / 2), self.grid_scale, self.grid_scale),

        pass

    def set_state(self, new_state):
        """
        Replace own state with new state
        """

        # self.state = [a for i in range(len(self.players)) for a in
        #               [self.players[i].position[0], self.players[i].position[1], self.players[i].angle,
        #               self.players[i].velocity[0], self.players[i].velocity[1], self.players[i].angle_speed]]
        

        ps_ratio = len(self.players)/len(new_state)
        for i in range(len(self.players)):
            self.players[i].position[0] = new_state[0 + i * ps_ratio]
            self.players[i].position[1] = new_state[1 + i * ps_ratio]
            self.players[i].angle = new_state[2 + i * ps_ratio]
            self.players[i].velocity[0] = new_state[3 + i * ps_ratio]
            self.players[i].velocity[1] = new_state[4 + i * ps_ratio]
            self.players[i].angle_speed = new_state[5 + i * ps_ratio]

        self.state = new_state
    
    def reset(self):
        # TODO: make this function not cancer
        for player in self.players:
            player.position[0] = 250
            player.position[1] = 250
            player.velocity[0] = 0
            player.velocity[1] = 0
            player.angle_speed = 0

        self.state = [a for i in range(len(self.players)) for a in
                      [self.players[i].position[0], self.players[i].position[1], self.players[i].angle,
                      self.players[i].velocity[0], self.players[i].velocity[1], self.players[i].angle_speed]]

        return self.state

    def step(self, actions: list, state=None):
        """
        :param actions: list of 2 ints [move_forward, shift_right] for each player
        :return:
        """
        # TODO: Make actions actually avaliable to multiple players

        wind, wind_direction = self.get_wind('whack')
        self.players[0].update_velocity(actions, wind=wind, wind_direction=wind_direction, time_discret=self.discret_factor)
        self.players[0].update_position(time_discret=self.discret_factor)

        reward = abs(actions[0]) * self.accel_reward + abs(actions[1]) * self.turn_reward
        reward += 100 * ( 350 < self.players[0].position[0] < 450 and 450 < self.players[0].position[1] < 450) # Goal cond.

        self.state = [a for i in range(len(self.players)) for a in
                      [self.players[i].position[0], self.players[i].position[1], self.players[i].angle,
                      self.players[i].velocity[0], self.players[i].velocity[1], self.players[i].angle_speed]]


        if self.rendering:
            self.render()

        return self.state, reward


        # Must make all cars move at the correct time



cliffcar = CliffCar()
cliffcar.rendering = True
time = 0
actions = [0, 0]

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                actions[1] = -1
            elif event.key == pygame.K_d:
                actions[1] = 1
            if event.key == pygame.K_w:
                actions[0] = 1
            elif event.key == pygame.K_s:
                actions[0] = -1
            if event.key == pygame.K_r:
                cliffcar.reset()


        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                actions[1] = 0
            elif event.key == pygame.K_d:
                actions[1] = 0
            if event.key == pygame.K_w:
                actions[0] = 0
            elif event.key == pygame.K_s:
                actions[0] = 0
    time += 1
    cliffcar.step(actions=actions)
    #print(cliffcar.players[0].velocity)
    print(cliffcar.players[0].angle_speed)