import numpy as np
import pygame
import matplotlib.pyplot as plt

# TODO: Get reward from testing_reward to this script
# TODO: FIX CLIFF HEIGHT AND ALL THAT SHIT
# TODO: MAKE CLIFF BE COLORED OR SOEMTHING

backgroundColor = (255,255,255)
red = (255, 0, 0)
blue = (0,0,255)
green = (0,255,0)
grey = (125,125,125)
black = (0,0,0)

# Find a better way of doing this
action_translator = {
    0: np.array([0, 0]),
    1: np.array([1, 0]),
    2: np.array([-1, 0]),
    3: np.array([0, 1]),
    4: np.array([0, -1])
}

def drawImage(display, path, center, scale=None, angle=None):
    img = pygame.image.load(path)
    if (scale != None):
        img = pygame.transform.scale(img, scale)
    if (angle != None):
        img = pygame.transform.rotate(img, angle)

    display.blit(img, center)

class CliffCar:

    def __init__(self, width=1500, height=1500, noise=True, start_pos_noise=False):

        self.width = width
        self.height = height
        self.noise = noise
        self.start_pos_noise = start_pos_noise

        # TODO: CHECK IF USIGN A LAMBDA FUNCTION CREATES SOME WEIRD POINTER FUCKERY
        self.car_position_func = lambda placing: np.array([self.width/placing, self.height/placing])\
                                                 + np.random.randn(2) * self.start_pos_noise

        self.car_position = self.car_position_func(4.5)

        self.cliff_height = self.height/5
        self.goal = [self.width*(4.5/6), self.height/(4.5)] # Where the car will fall down the cliff
        self.max_duration = 600 # Env terminates after this
        self.current_action = 0  # 0, 1, 2, 3, 4 NOOP, left, right, up, down

        self.max_x, self.max_y = self.width, self.height
        self.min_x, self.min_y = 0, 0

        # Perhaps replace above with below
        self.max = np.array([self.max_x, self.max_y])
        self.min = np.array([self.min_x, self.min_y])

        self.speed = 20
        self.noise_mean = 0
        # self.noise_var = 0.0000001
        self.noise_var = self.speed / 2
        self.reward_function = lambda x, y: sum([-x ** 2 + 2 * self.goal[0] * x, -y ** 2 + 2 * self.goal[1] * y])
        max_reward = self.reward_function(self.goal[0], self.goal[1])
        min_reward = self.reward_function(0, self.max_y)
        self.reward_normalizer = lambda reward: (reward-min_reward)/(max_reward-min_reward)

        # Technically this is the one we should use, but above should be faster
        # self.reward_normalizer = lambda val: (val - self.reward_func(0, 0)) / (reward_func(1000, 1000) - reward_func(0, 0))

        # self.noise_cov = noise_cov # Instead of cov we just use two variances

        self.frame = 0
        self.max_duration = 300

        self.block_size = [200, 200]
        self.moving = 0

        self.render_frame_interval = 10
        self.frame = 0
        self.path = 'Car.png'
        self.sprite_frame = 0
        self.rendering = False
        self.frame_rate = 60
        pygame.init()
        self.font = pygame.font.SysFont('Sans-serif', 50)

        # For interfacing with agent
        self.action_dim = 5
        self.obs_dim = 2
        # TODO: CHECK HOW MAX-MIN SHOULD WORK
        self.max_min = [[self.max_x, self.max_y], [self.min_x, self.min_y]]

    def step(self, action):
        self.frame += 1

        if type(action) == np.ndarray:
            action = action.item()

        # TODO: Remove action translator with seomthing a bit mroe sane
        self.car_position += self.speed * action_translator[action] + \
                             np.random.randn(2) * self.noise_var

        reward = self.reward_normalizer(self.reward_function(*self.car_position))

        if any(self.car_position >= self.max) or any(self.car_position <= self.min) or self.car_position[1] < self.cliff_height: # If to the right of cliff position -> Fall
            done = True
            reward = 0 # Would like to set this lower, but don't know if it works like that

        elif self.frame >= self.max_duration:
            done = True
            reward = 0
        else:
            done = False

        if self.rendering:
            self.render()

        return self.car_position, reward, done, 'derp' # Final value is dummy for working with gym envs


    def reset(self):
        self.car_position = np.array([self.width/4.5, self.height/4.5]) + np.random.randn(2) * self.start_pos_noise
        self.frame = 0 # Reset the timer (pretty important)
        return self.car_position

    def init_render(self):
        self.display = pygame.display.set_mode((self.width, self.height))
        self.rendering = True
        self.clock = pygame.time.Clock()

    def render(self):

        if not self.rendering:
            self.init_render()

        # TODO: FIND SOMETHING TO REPLACE THIS - Possibly something that changes how the car faces
        # self.sprite_frame += 1
        #
        # if self.sprite_frame % self.render_frame_interval == 0:
        #     if self.path == 'sumo/images/sumo_2.png' or self.path == 'sumo/images/sumo_1.png':
        #         self.path = 'sumo/images/sumo_3.png'
        #
        #     elif self.path == 'sumo/images/sumo_3.png':
        #         self.path = 'sumo/images/sumo_2.png'[self.width/(4.5/6), self.height/(4.5/6)]
        #
        #     if self.current_action == 0:
        #         self.path = 'sumo/images/sumo_1.png'

        self.clock.tick(self.frame_rate)
        pygame.display.flip() # Why this?
        self.display.fill(backgroundColor) # Fill with background color - Do before other stuff

        for x in range(-int(self.block_size[0]), self.width, self.block_size[0]):
            for y in range(-int(self.block_size[1]), self.height, self.block_size[1]):
                pygame.draw.rect(self.display, black, (x, y, self.block_size[0]-10, self.block_size[1]-10))

        # The cliff
        pygame.draw.rect(self.display, (0, 0, 255), (0, 0, self.width, self.cliff_height))
        # The Goal
        pygame.draw.rect(self.display, (0,255,0), (self.goal[0], self.goal[1], 50, 50))
        # Reference square, for absolute car position
        drawImage(self.display, path=self.path, center=(self.car_position[0]-10, self.car_position[1]-10), scale=(65, 65))
        pygame.draw.rect(self.display, blue, (*self.car_position, 50, 50))

        reward = self.font.render('Reward ' + str(self.reward_normalizer(self.reward_function(*self.car_position))), True, (125, 125, 125))
        self.display.blit(reward, (50,1050))
        pos = self.font.render('Position ' + str(self.car_position), True, (125, 125, 125))
        self.display.blit(pos, (50,1100))


if __name__ == "__main__":
    env = CliffCar()
    env.init_render()

    # Plot reward function
    # xs = np.linspace(0, env.max_x, env.max_x*2)
    # ys = env.reward_function(xs)
    # plt.plot(xs, env.reward_normalizer(ys))
    # plt.show()
    action = 0
    done = False
    episode_reward = 0
    while True:

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        action = 2
                    elif event.key == pygame.K_d:
                        action = 1
                    elif event.key == pygame.K_w:
                        action = 4
                    elif event.key == pygame.K_s:
                        action = 3

                    if event.key == pygame.K_r:
                        env.reset()

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        action = 0
                    elif event.key == pygame.K_d:
                        action = 0
                    elif event.key == pygame.K_w:
                        action = 0
                    elif event.key == pygame.K_s:
                        action = 0
            print(action, env.goal[0], env.goal[1])
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            env.render()
            print(done)
        done = False
        state = env.reset()
        print(episode_reward)

