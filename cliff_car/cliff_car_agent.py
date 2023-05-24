from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import random
from IPython.display import clear_output
from sumo.network import Network
from tqdm import tqdm


class CliffCarAgent:
    def __init__(self, env, replay_buffer, epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None):
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
        print(self.device) # Just to know which one we're on

        self.dqn = Network(env.obs_dim, env.action_dim).to(self.device)
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


    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = random.randint(0, self.env.action_dim-1) # Why is this not inclusive, exclusive??? Stupid
        else:
            select_state = self.state_normalizer(state)
            selected_action = self.dqn(
                torch.FloatTensor(select_state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        # Store transitions
        if not self.is_test:
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

    def train(self, num_frames: int, plotting_interval: int = 200, q_val_plotting_interval=200):
        """Train the agent."""
        self.is_test = False
        self.dqn.train()

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0


        for frame_idx in tqdm(range(1, num_frames + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # Update whether we're ready to train
            if not self.training_ready:
                self.training_ready = self.replay_buffer.training_ready()

            else:
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

            # plotting
                if frame_idx % plotting_interval == 0:
                    self._plot(frame_idx, scores, losses, epsilons)

        print("Training complete")
        return scores, losses, epsilons

    def save_model(self, model_path):
        try:
            torch.save(self.dqn.state_dict(), model_path)
        except:
            print("ERROR! Could not save model!")

    def test(self, test_games=100, render_games: int=0, render_speed: int=60):
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

        for i in range(test_games):
            state = self.env.reset()
            done = False

            # NOTE that we also get the last state, action, reward when the environment terminates here...

            sar = [(np.nan, np.nan, np.nan)] * self.env.max_duration

            i = 0
            # Changed here from training, since we play games till the end, not for a certain number of steps (frames)
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                # Using item here because they are numpy arrays... stupid
                sar[i] = (state, action.item(), reward)

                i += 1

                state = next_state

            all_sar.append(sar)

        # If statement necessary, otherwise Pygame opens and stays loitering around
        if render_games > 0:
            self.env.init_render()
            self.env.frame_rate = render_speed

        for i in range(render_games):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                self.env.render()

                state = next_state

        return np.array(all_sar)

    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))


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
