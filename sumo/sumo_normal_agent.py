# TODO: WE HAVE A PROBLEM!
# TODO: ACTIONS!!! CAN COME AS NUMPY ARRAYS??? WTFFFF
# TODO: MAKE THE WHOLE THING WORK IN CASE ACTIONS COME AS NUMPY ARRAYS AND NOT INTEGERS, THAT WOULD BE STUPIDDDD
# TODO GRIDKEYS: Some way of adding variable fineness to the different grid keys
# TODO: IMPORTANT: Find out if all types (tuples, np arrays, such)... are correct!
    # Must be done in create_grid_keys
    # Must change mult_dim og single_dim intepreter til at passe med forskellige fineness
# TODO: Det der skal lægges til INdex af action skal ændres til at være sum(actions) hvis actions er en liste


# from sumo import sumo_pp
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sumo_utils import create_grid_keys, stencil, single_dim_interpreter, multi_dim_interpreter, neighbours
from replay_buffer import TheCoolerReplayBuffer
import random


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
    # TODO: SWAP .INDEX FROM GRID_LIST TO GRID_KEYS (THEY ARE SORTED!!)
    def __init__(self, fineness, env, state_dim, action_dim, replay_buffer_size, max_min, epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99):
        self.fineness = fineness
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max, self.min = max_min[0], max_min[1] # TODO: fix how max_min is represented, see grid_keys

        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma

        self.grid_keys, self.grid_list = create_grid_keys(fineness)
        self.replay_buffer = TheCoolerReplayBuffer(state_dim, replay_buffer_size,batch_size=32, fineness=fineness,
                                                   num_actions=action_dim, state_max=self.max, state_min=self.min)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dqn = Network(state_dim, action_dim).to(self.device)
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

    def get_closest_grids(self, current_pos: tuple) -> tuple:
        # TODO Implement actual euclidian, not this cursed squared distance... doesn't scale well, as you probably know
        # Returns [x, y] location of the nearest grid...

        squared_dists = np.sum((current_pos - self.grid_keys) ** 2, axis=1)
        nearest_grid = self.grid_keys[np.argmin(squared_dists)] # Works since we have sorted grid_keys but not squared_dists
        return nearest_grid

    def get_neighbour_grids(self, current_grid: tuple, neighbour_grids: int) -> list:
        # Gets neighbouring grid [x, y] locations from current grid [x, y] location
        # Index, not coordinates!

        # Get the index of the current grid_position
        # TODO CHECK THIS IS TRUE, maybe grid_list?
        grid_index = np.where(self.grid_keys == current_grid) # Naw brother, this is gonna fail big time

        # Cursed Single->Multi->Single again merry-go-round
        multi_index = single_dim_interpreter(grid_index, fineness=self.fineness, dim=self.state_dim)
        neighbours_multi_index = neighbours(P=multi_index, neighbours=neighbour_grids,
                                            maxx=self.fineness - 1, minn=0)
        neighbours_single_index = [multi_dim_interpreter(i, fineness=self.fineness, dim=None)
                                   for i in neighbours_multi_index]

        return neighbours_single_index

    # TODO: FIND OUT IF ACTION SHOULD BE NUMPY ARRAY OR INTEGER
    def get_KNN(self, pos: tuple, action, K: int=0, neighbour_grids: int=0) -> dict:
        """
        Get K-nearest neighbours for a given pos based on state
        :param pos: Current pos (state) to look based on
        :param action: Only look up places in the replay buffer that have an action corresponding to current action
        :param K: K neighbours to find
        :param neighbour_grids: Number of neighbouring grids next to current to look at
        :return:
        """

        # (x,y) -> Get the cloest grid to current location
        closest_grid = self.get_closest_grids(pos)
        # [idxs] -> Get single-dim index of neighbouring grids
        idxs_to_look = self.get_neighbour_grids(closest_grid, neighbour_grids=neighbour_grids) # Flattened idxs of grids to search

        # {obs, rew, act...} -> Get all obs from replay_buffer partitions
        all_points = self.replay_buffer[self.replay_buffer.create_partition_idxs(idxs_to_look, action)]

        # [idxs] -> Get indices of KNN in previous dict
        all_state_values = all_points['obs']
        squared_dists = np.sum(np.sqrt((pos - all_state_values) ** 2, axis=1))
        indices_to_get = np.argpartition(squared_dists, kth=K)  # Probs inefficient K-NN algorithm who cares

        # {obs[idxs], rew[idxs], act[idxs]...} Final samples
        all_points = {key: value[indices_to_get] for key,value in all_points.items()}

        return all_points
        # TODO: look into FAISS for getting KNN among all_points_of_interest

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = random.randint(0, self.action_dim)
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
        # TODO: MAKE STEP FUNCTION TAKE CURRENT_STATE AND SELECTED_ACTION? TO NOT HAVE TO LOOK INTO SELF.TRANSITION?
        # TODO: HAVE COOLER_REPLAY_BUFFER.STORE TAKE ACTION INTO ACCOUNT - It already does u piece of dope
        if not self.is_test:
            # Set Idx of closest grid_location
            self.current_grid = int(np.where(self.grid_list == self.get_closest_grids(self.transition[0])))
            self.transition += [reward, next_state, done, self.current_grid]
            self.replay_buffer.store(*self.transition)

        return next_state, reward, done
    
    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""

        state, action = self.transition[0], self.transition[1]
        samples = self.get_KNN(state, action, K=100, neighbour_grids=2)

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready - Just check if current grid has like 10 points
            # TODO: Ask Calle how few points are necessary
            if self.replay_buffer.spec_len(self.current_grid) >= 10:
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

                # if hard update is needed
                # TODO: DELETE AFTER SHOWING; WE DON'T DO HARD UPDATES
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)

        self.env.close()

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        #self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        # TODO: Do this for each action, we don't wanna
        # TODO: CONSIDER WHETHER WE CAN ACTUALLY TRAIN ONLINE VS OFFLINE...
        # TODO: CONSIDER WHETHER THERE SHOULD BE SOME RANDOMNESS IN HOW WE SAMPLE...
        # ONLINE, AND STATE (FOR CURR_Q_VALUE) NEEDS TO BE THE STATE CURRENTLY IN
        # oFFLINE AND QVALS ARE USED BOTH FOR ROBUST ESTIMATE AND CURRENT ESTIMATE
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        states = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device) # Just the first one is actually important -  they're all the same
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        q_vals = self.dqn(states).gather(1, action)
        curr_q_value = self.dqn(states).gather(1, action) # Gather just grabs relevant Q-value for action along each value


        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

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

