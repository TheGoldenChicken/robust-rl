
import numpy as np
import pygame

backgroundColor = (255,255,255)
red = (255, 0, 0)
blue = (0,0,255)
green = (0,255,0)
grey = (125,125,125)


def drawImage(display, path, center, scale=None, angle=None):
    img = pygame.image.load(path)
    if (scale != None):
        img = pygame.transform.scale(img, scale)
    if (angle != None):
        img = pygame.transform.rotate(img, angle)

    display.blit(img, center)


class SumoPPEnv:

    def __init__(self):
        self.line_length = 500
        self.start_position = self.line_length/5
        self.sumo_position = self.start_position
        self.hill_position = self.line_length/2

        self.sumo_speed = 5
        self.noise_mean = 0
        self.noise_var = self.sumo_speed / 2
        self.reward_function = lambda pos: -(pos - self.hill_position)**2

        self.width = 1000
        self.height = 1000

        self.block_size = 50
        self.moving = 0

        # TODO: Replace this stupid shit with stupid shit using the clock instead
        self.render_frame_interval = 20
        self.frame = 0
        self.path = 'sumo_1.png'

    def step(self, action):
        self.sumo_position += self.sumo_speed * action\
                              + np.random.normal(loc=self.noise_mean,scale=self.noise_var, size=1)[0]

        reward = self.reward_function(self.sumo_position)

        # if action != 0 and self.moving == 0:
        #     self.moving = 1
        # self.moving = self.moving * -1
        # if action == 0:
        #     self.moving = 0
        #self.moving = self.moving * -1 if action != 0 else 0

        if action == 0 and (self.path == 'sumo_2.png' or self.path == 'sumo_3.png'):
            if self.noise_var == 0: self.path = 'sumo_1.png'
            #else: self.path = 'sumo_win].png'

        if action != 0 and self.path == 'sumo_1.png':
            self.path = 'sumo_2.png'


        return self.sumo_position, reward

    def reset(self):
        self.sumo_position = self.start_position

    def init_render(self):
        self.display = pygame.display.set_mode((self.width, self.height))
        self.rendering = True
        self.clock = pygame.time.Clock()

    def render(self):
        self.frame += 1
        if self.frame % self.render_frame_interval == 0:
            if self.path == 'sumo_1.png':
                pass

            elif self.path == 'sumo_2.png':
                self.path = 'sumo_3.png'

            elif self.path == 'sumo_3.png':
                self.path = 'sumo_2.png'



        self.clock.tick(60)
        pygame.display.flip() # Why this?
        self.display.fill(backgroundColor) # Fill with background color - Do before other stuff

        for x in range(-int(self.block_size), self.width, self.block_size):
            pygame.draw.rect(self.display, red, (x, int(self.height/2), self.block_size-10, self.block_size-10))

        # if self.moving == 0:
        #     path = 'sumo_1.png'
        # elif self.moving == 1:
        #     path = 'sumo_2.png'
        # elif self.moving == -1:
        #     path = 'sumo_3.png'

        #pygame.draw.rect(self.display, blue, (self.sumo_position, int(self.height/2), self.block_size-10, self.block_size-10))
        drawImage(self.display, path=self.path, center=(self.sumo_position-23, int(self.height/2) - 84*1.5), scale=(150, 150))

env = SumoPPEnv()
env.init_render()
action = 0
while True:

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                action = -1
            elif event.key == pygame.K_d:
                action = 1
            if event.key == pygame.K_r:
                env.reset()

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                action = 0
            elif event.key == pygame.K_d:
                action = 0
    env.step(action)
    env.render()

