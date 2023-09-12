from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import random
from IPython.display import clear_output
from tqdm import tqdm


class CliffCarAgent:
    def __init__(self, env, replay_buffer, epsilon_decay, network, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None):
        self.env = env
        self.max, self.min = np.array(self.env.max_min[0]), np.array(self.env.max_min[1])

        # Learning and exploration parameters
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.gamma = gamma

        self.replay_buffer = replay_buffer
        self.training_ready = False

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # print(self.device) # Just to know which one we're on

        self.dqn = network(env).to(self.device)
        if model_path is not None:
            self.load_model(model_path)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001) # Adam default lr=0.001

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # Should work for tensors and numpy...
        self.state_normalizer = lambda state: (state - self.min)/(self.max - self.min)

    def get_samples(self) -> tuple[dict, ]:
        """
        Should be updated for each individual agent type
        returns: tuple[samples,current_samples] current_samples only if robust agent, samples is list in this case
        """
        raise(NotImplementedError)
        return samples

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Needs to be implemented for each agent"""
        raise(NotImplementedError)
        return loss

    def select_action(self, state: int) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = random.randint(0, self.env.ACTION_DIM-1) # Why is this not inclusive, exclusive??? Stupid
        else:
            # select_state = self.state_normalizer(state)
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax(axis=1)[0]
            selected_action = int(selected_action)

        if not self.is_test:
            self.transition = [state, selected_action]
            
        if(selected_action >= 5):
            print(selected_action)
        return selected_action

    def step(self, action: int) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        # Store transitions
        if not self.is_test and len(next_state) > 0:
            # Store current transition
            self.transition += [reward, next_state, done]
            self.replay_buffer.store(*self.transition)

        return next_state, reward, done

    def update_model(self): # -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.get_samples() # Get_samples needs to be set for each subclass

        loss = self._compute_dqn_loss(*samples)

        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping here?
        self.optimizer.step()

        return loss.item()

    def train(self, train_frames: int, test_interval = 200, test_games = 100, do_test_plots = True, test_name_prefix = ""):
        """Train the agent."""
        self.is_test = False
        self.dqn.train()

        self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        # War up until we are ready to train.
        while(not self.training_ready):
            action = self.select_action(self.env.position)
            _, _, done = self.step(action)

            # if episode ends
            if done:
                self.env.reset()
            
            self.training_ready = self.replay_buffer.training_ready()
        
        # Start training for the given amount of trames
        self.env.reset()
        for frame_idx in tqdm(range(1, train_frames + 1)):
            action = self.select_action(self.env.position)
            _, reward, done = self.step(action)

            score += reward

            # if episode ends
            if done:
                self.env.reset()
                scores.append(score)
                score = 0
            

            loss = self.update_model()
            losses.append(loss)
            update_cnt += 1

            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                ) * self.epsilon_decay
            )
            epsilons.append(self.epsilon)

            if frame_idx % test_interval == 0:
                print(">>> Testing: " + test_name_prefix + "-frame-" + str(frame_idx) + "-epsilon-" + str(self.epsilon))
                self.test(test_games=test_games,
                          test_name_prefix = test_name_prefix + "-frame-" + str(frame_idx),
                          do_test_plots=do_test_plots)
                pass
            # # plotting
            # if frame_idx % plotting_interval == 0:
            #     self._plot(frame_idx, scores, losses, epsilons)

        print("Training complete")
        return scores, losses, epsilons

    def save_model(self, model_path):
        try:
            torch.save(self.dqn.state_dict(), model_path)
        except:
            print("ERROR! Could not save model!")

    def test(self, test_games=100, test_name_prefix: str = "", do_plots=False, render_games: int=0, render_speed: int=60):
        """
        Test the agent
        :param test_games: number of test games to get score from
        :param render_games: number of rendered games to inspect qualitatively
        :param render_speed: frame_rate of the rendered games
        :return:
        """
        self.is_test = True # Prevent from taking random actions
        all_sar = [] # All state_action_reward
        self.dqn.eval() # Set network to evaluation mode disabling dropout

        for i in tqdm(range(test_games)):
            self.env.reset()
            done = False

            # NOTE that we also get the last state, action, reward when the environment terminates here...

            sar = np.zeros([self.env.max_duration + 1, 3,2]) # state, action, reward

            frame = 0
            # Changed here from training, since we play games till the end, not for a certain number of steps (frames)
            while not done:
                action = self.select_action(self.env.position)
                _, reward, done = self.step(action)

                sar[frame] = np.array([self.env.position, np.array([action,0]), np.array([reward,0])])

                frame += 1

            all_sar.append(sar)

        # If statement necessary, otherwise Pygame opens and stays loitering around
        if render_games > 0:
            self.env.init_render()
            self.env.frame_rate = render_speed

        for i in range(render_games):
            self.env.reset()
            done = False
            while not done:
                action = self.select_action(self.env.position)
                _, reward, done = self.step(action)
                self.env.render()

        if do_plots:
            self.do_test_plots(all_sar, plot_name_prefix=test_name_prefix)

        self.is_test = False

        return np.array(all_sar)
    
    def do_test_plots(self, all_sar, plot_grid_spacing = 0.5, plot_name_prefix = ""):
        x_range = torch.arange(self.env.BOUNDS[0], self.env.BOUNDS[2], plot_grid_spacing)
        y_range = torch.arange(self.env.BOUNDS[1], self.env.BOUNDS[3], plot_grid_spacing)
        x, y = torch.meshgrid(x_range, y_range)
        state_grid = torch.column_stack((x.ravel(), y.ravel()))

        # Get q-values for the state grid
        q_vals = self.get_q_vals(state_grid)
        v_vals = np.max(q_vals, axis=1).reshape(x.shape)
        # Plot the q-values as image using the brq cmap
        plt.imshow(v_vals, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], origin='lower', cmap='brg')
        plt.colorbar()
        plt.savefig(f'v_vals_{plot_name_prefix}.png')
        plt.clf()

        # Plot the visited states as a heatmap
        states = np.array([sar[:,0] for sar in all_sar]).reshape(-1,2)
        states = states[~np.all(states == 0, axis=1)] # exclude rows with value [0,0]
        heatmap, _, _ = np.histogram2d(states[:,0], states[:,1], bins=[len(x_range), len(y_range)])
        plt.imshow(heatmap.T, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], origin='lower', cmap='brg')
        plt.colorbar()
        plt.savefig(f'heatmap_{plot_name_prefix}.png')
        plt.clf()
        
    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path, map_location=self.device))


    def get_q_vals(self, states):
        return self.dqn(states).detach().cpu().numpy()


    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
            plot_q_vals: bool = False
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)

        if plot_q_vals:
            states = torch.FloatTensor(np.linspace(0, 1, 1000)).reshape(-1, 1).to(self.device)
            q_vals = self.get_q_vals(states)
            plt.subplot(234)
            plt.title('action_0')
            plt.plot(q_vals[:, 0])
            plt.subplot(235)
            plt.title('action_1')
            plt.plot(q_vals[:, 1])
            plt.subplot(236)
            plt.title('action_2')
            plt.plot(q_vals[:, 2])

        plt.show()

    def _plot_q_vals(self):
        states = torch.FloatTensor(np.linspace(0, 1, 1000)).reshape(-1, 1).to(self.device)
        data = self.get_q_vals(states)
        action0 = data[:, 0]
        action1 = data[:, 1]
        action2 = data[:, 2]

        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('Action 0')
        plt.plot(action0)
        plt.subplot(132)
        plt.title('Action 1')
        plt.plot(action1)
        plt.subplot(133)
        plt.title('Action 2')
        plt.plot(action2)
        plt.show()
