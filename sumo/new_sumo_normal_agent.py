# TODO: WE HAVE A PROBLEM!
# TODO: ACTIONS!!! CAN COME AS NUMPY ARRAYS??? WTFFFF
# TODO: MAKE THE WHOLE THING WORK IN CASE ACTIONS COME AS NUMPY ARRAYS AND NOT INTEGERS, THAT WOULD BE STUPIDDDD
# TODO GRIDKEYS: Some way of adding variable fineness to the different grid keys
# TODO: IMPORTANT: Find out if all types (tuples, np arrays, such)... are correct!
# Must be done in create_grid_keys
# Must change mult_dim og single_dim intepreter til at passe med forskellige fineness
# TODO: Det der skal lægges til INdex af action skal ændres til at være sum(actions) hvis actions er en liste
# TODO: CONSIDER ADDING NUM_NIEGHBOURS (CHECK HOW MANY GRIDS) TO AGENT

# from sumo import sumo_pp
from sumo import sumo_pp
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sumo_utils import normalize_tensor
from replay_buffer import TheCoolerReplayBuffer
import random
import distributionalQLearning2 as distributionalQLearning
from IPython.display import clear_output


# SubSymbolic AI? Knowing the effect of action and just calculating the noise instead of the transition probabilities


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class SumoNormalAgent:
    def __init__(self, fineness, env, state_dim, action_dim, batch_size, replay_buffer_size, max_min, epsilon_decay,
                 max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None, ripe_when=20):
        self.fineness = fineness
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max, self.min = np.array(max_min[0]), np.array(max_min[1])  # TODO: fix how max_min is represented, see grid_keys [[max_1, max_2], [min_1,min_2]]

        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.gamma = gamma

        # Not necessary anymore
        #self.grid_keys, self.grid_list = create_grid_keys(fineness)
        self.replay_buffer = TheCoolerReplayBuffer(state_dim, replay_buffer_size, batch_size=batch_size, fineness=fineness,
                                                   num_actions=action_dim, state_max=self.max, state_min=self.min, ripe_when=ripe_when)

        self.batch_size = batch_size # Nearest neighbours to update based on
        self.number_neighbours = 2

        self.device = torch.device(
            "cpu" if torch.cuda.is_available() else "cpu"
        )

        self.dqn = Network(state_dim, action_dim).to(self.device)

        if model_path is not None:
            self.dqn.load_state_dict(model_path)

        # self.dqn_target = Network(state_dim, action_dim).to(self.device) # Perhaps not needed
        # self.dqn_target.load_state_dict(self.dqn.state_dict())
        # self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # To hold which grid it's currently at
        self.current_grid = 0

        # Should work for tensors and numpy...
        self.state_normalizer = lambda state: (state - self.min)/(self.max - self.min)

        # Debugging
        self.debug_losses = []

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = random.randint(0, self.action_dim-1) # Why is this not inclusive, exclusive??? Stupid
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
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
            self.replay_buffer.store(*self.transition, idx=None) # Set idx to none so it can find it itself

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.replay_buffer.sample_from_scratch(K=self.batch_size, nn=self.number_neighbours)

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200, save_model=None):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        current_episode_score = 0
        episode_scores = []

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            current_episode_score += reward

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                done = False # Perhaps this should be done in the step function?
                scores.append(score)
                score = 0
                current_episode_score = 0
                episode_scores.append(current_episode_score)

            # TODO: FIND A BETTER FUCKING WAY OF DOING THIS!
            there_is_data = False
            if sum(self.replay_buffer.ripe_bins) >= 10: # How many ripe bins we require before starting to update
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
                there_is_data = True

            # plotting
            if frame_idx % plotting_interval == 0 and there_is_data:
                self._plot(frame_idx, scores, losses, epsilons)
                #self._special_plot(episode_scores, epsilons, losses, frame_idx)
                print(frame_idx, loss, self.epsilon, )

        print("Training complete")

        if save_model is not None:
            try:
                torch.save(self.dqn.state_dict(), save_model)
            except:
                print("ERROR! Could not save model!")

        # self.env.close()

    def test(self, test_games=100, render_games: int=0, render_speed: int=60):
        self.is_test = True # Prevent from taking random actions
        state = self.env.reset()
        scores = []
        done = False

        for i in range(test_games):
            score = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            scores.append(score)

        self.env.init_render()
        state = self.env.reset()
        self.env.frame_rate = 60
        for i in range(render_games):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                self.env.render()

                state = next_state

        return scores

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        # TODO: Do this for each action, we don't wanna
        # TODO: MAKE ROBUST ESTIMATOR WORK WITH TENSORS...

        device = self.device  # for shortening the following lines
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        # done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        done = np.expand_dims(samples['done'], -1) # Don't think we need this as a tensor... So discount reshaping here

        state = self.state_normalizer(state)
        next_state = self.state_normalizer(next_state)

        # TODO: CHECK IF V(S) ESTIMATE IS BASED ON CURRENT OR NEXT STATE
        Q_vals = self.dqn(torch.FloatTensor(state).to(device)) # Should all have the same action
        current_q_value = Q_vals.gather(1, action)
        Q_vals = Q_vals.max(dim=1, keepdim=True)[0].detach().cpu().numpy()

        if Q_vals[1] == Q_vals[0]:
            i = 5

        X_p=samples['obs']
        y_p = samples['next_obs']
        y_v = Q_vals
        #  TODO: THE ROBUST ESTIMATOR DOES NOT FAIL THE SAME ITERATION AS IT GETS NAN, ONLY THE ONE AFTER THAT!!!
        robust_estimator = distributionalQLearning.robust_estimator(X_p=state, y_p=next_state,
                                                                    X_v=next_state, y_v=Q_vals,
                                                                    delta=0.5)

        # robust_estimator = distributionalQLearning.robust_estimator(X_p=samples['obs'], y_p=samples['next_obs'],
        #                                                             X_v=samples['next_obs'], y_v=Q_vals,
        #                                                             delta=0.5)

        # print(f'X_p: {np.mean(state)} \n y_p: {np.mean(next_state)} \n y_v: {np.mean(y_v)}')
        # print("Robust estimator", robust_estimator)
        mask = 1 - done  # Remove effect from those that are done
        robust_estimator = reward + self.gamma * robust_estimator * mask

        # curr_q_value = Q_vals.gather(1, action) - Pretty sure we don't need this

        # calculate dqn loss
        loss = F.smooth_l1_loss(current_q_value, robust_estimator)
        # print(torch.mean(current_q_value), torch.mean(robust_estimator), loss.item())

        return loss

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
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
        plt.show()

    # def _special_plot(self, episode_scores, epsilons, losses, frame_idx):
    #     """
    #     Episode-wise plotting function that should be deleted when Karl is done using
    #     """
    #     """Plot the training progresses."""
    #     clear_output(True)
    #     plt.figure(figsize=(20, 5))
    #     plt.subplot(131)
    #     plt.title('frame %s. score: %s' % (len(episode_scores), episode_scores))
    #     plt.plot(np.arange(len(episode_scores)), episode_scores)
    #     plt.subplot(132)
    #     plt.title('loss')
    #     plt.plot(losses)
    #     plt.subplot(133)
    #     plt.title('epsilons')
    #     plt.plot(epsilons)
    #     plt.show()


if __name__ == "__main__":

    # environment
    line_length = 500
    env = sumo_pp.SumoPPEnv(line_length=line_length)

    seed = 777


    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


    np.random.seed(seed)
    seed_torch(seed)

    num_frames = 12000

    # parameters
    fineness = 100
    state_dim = 1
    action_dim = 3
    batch_size = 40
    replay_buffer_size = 500
    max_min = [[env.cliff_position],[0]]
    epsilon_decay = 1/1000
    ripe_when = 20

    # TODO: PERHAPS JUST PASS A PREMADE REPLAY BUFFER TO THE SUMO AGENT TO AVOID SO MANY PARAMETERS?
    agent = SumoNormalAgent(fineness=fineness, env=env, state_dim=state_dim, action_dim=action_dim, batch_size=batch_size,
                            replay_buffer_size=replay_buffer_size, max_min=max_min, epsilon_decay=epsilon_decay, ripe_when=ripe_when)

    agent.train(num_frames)
    scores = agent.test(test_games=1000, render_games=10)
    i = 5
    print(scores, np.mean(scores))
# if all([self.replay_buffer.spec_len(i) >= self.batch_size for i in range(self.replay_buffer.bins)]):