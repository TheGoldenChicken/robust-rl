import numpy as np
import pygame
import matplotlib.pyplot as plt

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

    def __init__(self, line_length):
        self.line_length = line_length
        self.start_position = self.line_length/5 # TODO: RANDOMIZE THIS
        self.sumo_position = self.start_position
        self.hill_position = self.line_length/2 # TODO: RANDOMIZE THIS
        self.cliff_position = self.hill_position + 20 # Where the sumo will fall down the cliff
        self.max_duration = 300 # Env terminates after this

        self.sumo_speed = 5
        self.noise_mean = 0 # Not meant to be changed, or introduces bias
        self.noise_var = self.sumo_speed / 2 # Higher var <-> More difficulty
        self.reward_function = lambda pos: 1/(pos - self.hill_position)**2 # 1 over squared istances from the hill
        self.reward_function = lambda pos: -pos**2 + 2*self.hill_position*pos
        self.reward_normalizer = lambda reward: reward/self.reward_function(self.hill_position)

        self.max_x = line_length # Terminates beyond this - Hill may not move beyond 10 of this position
        self.min_x = 0 # Terminates beyond this


        # Used for rendering purposes
        self.width = 1000
        self.height = 1000
        self.block_size = 50
        self.moving = 0

        # TODO: Replace this stupid shit with stupid shit using the clock instead
        self.render_frame_interval = 10
        self.frame = 0
        self.path = 'sumo_1.png'

        self.current_action = 0 # 0, 1, 2, NOOP, left, right
        self.sprite_frame = 0

        self.rendering = False
        self.frame_rate = 60

    def step(self, action):
        self.frame += 1

        self.current_action = action
        # Terrible fucking solution to translate agent actions to this incredibly simple math
        if action == 2:
            actual_action = -1
        else:
            actual_action = action

        self.sumo_position += self.sumo_speed * actual_action\
                              + np.random.normal(loc=self.noise_mean,scale=self.noise_var, size=1)[0]

        reward = self.reward_normalizer(self.reward_function(self.sumo_position))

        if self.sumo_position >= self.cliff_position or self.sumo_position <= self.min_x: # If to the right of cliff position -> Fall
            done = True
            reward = 0 # Would like to set this lower, but don't know if it works like that

        elif self.frame >= self.max_duration:
            done = True
            reward = 50
        else:
            done = False

        if self.rendering:
            self.render()

        return np.array([self.sumo_position]), reward, done, 'derp' # Final value is dummy for working with gym envs

    def reset(self):
        self.sumo_position = self.start_position
        self.frame = 0 # Reset the timer (pretty important)
        return np.array([self.sumo_position])

    def init_render(self):
        self.display = pygame.display.set_mode((self.width, self.height))
        self.rendering = True
        self.clock = pygame.time.Clock()

    def render(self):

        if not self.rendering:
            self.init_render()

        self.sprite_frame += 1

        if self.sprite_frame % self.render_frame_interval == 0:
            if self.path == 'sumo_2.png' or self.path == 'sumo_1.png':
                self.path = 'sumo_3.png'

            elif self.path == 'sumo_3.png':
                self.path = 'sumo_2.png'

            if self.current_action == 0:
                self.path = 'sumo_1.png'

        self.clock.tick(self.frame_rate)
        pygame.display.flip() # Why this?
        self.display.fill(backgroundColor) # Fill with background color - Do before other stuff

        for x in range(-int(self.block_size), self.width, self.block_size):
            pygame.draw.rect(self.display, red, (x, int(self.height/2), self.block_size-10, self.block_size-10))

        pygame.draw.rect(self.display, (255,255,255), (self.cliff_position, int(self.height/2), 1000, 1000))
        pygame.draw.rect(self.display, (0,0,0), (self.cliff_position, int(self.height/2)+50, 1000, 1000))


        #pygame.draw.rect(self.display, blue, (self.sumo_position, int(self.height/2), self.block_size-10, self.block_size-10))
        drawImage(self.display, path=self.path, center=(self.sumo_position-100, int(self.height/2) - 84*1.5), scale=(150, 150))


if __name__ == "__main__":
    line_length = 1000
    env = SumoPPEnv(line_length=line_length)
    env.init_render()

    # Plot reward function
    xs = np.linspace(0, 500, line_length * 2)
    ys = env.reward_function(xs)
    #plt.plot(xs, ys)
    plt.plot(xs, env.reward_normalizer(ys))
    plt.show()
    print(xs[500],ys[500], env.hill_position)
    print(np.where(env.reward_normalizer(ys) == max(env.reward_normalizer(ys))))
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
                    if event.key == pygame.K_r:
                        env.reset()

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        action = 0
                    elif event.key == pygame.K_d:
                        action = 0
            next_state, reward, done, _ = env.step(action)
            print(done)
            episode_reward += reward
            env.render()

        done = False
        state = env.reset()
        print(episode_reward)
