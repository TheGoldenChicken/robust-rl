from sumo_agent import SumoAgent
import torch
import torch.nn.functional as F
import distributionalQLearning2 as distributionalQLearning
import numpy as np
from replay_buffer import TheCoolerReplayBuffer
from sumo_pp import SumoPPEnv
from typing import Dict, List

class RobustSumoAgent(SumoAgent):
    def __init__(self, env, replay_buffer, epsilon_decay, grad_batch_size=30, delta=0.5, max_epsilon=1.0,
                 min_epsilon=0.1, gamma=0.99, model_path=None, robust_factor=1):
        super().__init__(env, replay_buffer, epsilon_decay, max_epsilon, min_epsilon, gamma, model_path)

        self.robust_factor = robust_factor # Factor multiplied on the robust estimator
        self.grad_batch_size = grad_batch_size # How many samples to compute each loss based off of
        self.delta = delta # For the robust estimator

        # Adding the below as class specific values is kinda stupid, however not having them here would require changing the train function
        self.betas = []
        self.state_value_pairs = []
        self.robust_estimators = []
        self.quadratic_approximations = []
    def get_samples(self) -> tuple[dict, dict]:
        """
        Should be updated for each individual agent type
        """
        samples, current_samples = self.replay_buffer.sample_from_scratch(K=self.replay_buffer.batch_size,
                                                                          nn=self.replay_buffer.num_neighbours,
                                                                          num_times=self.grad_batch_size)

        return samples, current_samples
    def _compute_dqn_loss(self, samples: List[Dict[str, np.ndarray]],
                          current_samples: List[Dict[str, np.ndarray]] = 0) -> torch.Tensor: # Default value only so Pycharm doesn't shit itself
        """
        Blablabla compute the loss, this is a good docstring, screw u
        :param samples:
        :param curent_samples:
        :return:
        """
        # TODO: MAKE ROBUST ESTIMATOR WORK WITH TENSORS...
        device = self.device  # for shortening the following lines

        rewards = torch.FloatTensor(current_samples['rews']).reshape(-1,1).to(device)
        mask = torch.FloatTensor(1 - current_samples['done']).reshape(-1,1).to(device)
        current_sample_obs = torch.FloatTensor(self.state_normalizer(current_samples['obs'])).to(device)
        current_sample_actions = torch.LongTensor(current_samples['acts']).reshape(-1,1).to(device)
        current_q_values = self.dqn(current_sample_obs).gather(1, current_sample_actions)

        robust_estimators = []
        beta_maxss = []
        for i, sample in enumerate(samples):
            state = sample["obs"]
            next_state = sample["next_obs"]

            # Normalize state values (very important, trust me!)
            state = self.state_normalizer(state)
            next_state = self.state_normalizer(next_state)

            Q_vals = self.dqn(torch.FloatTensor(next_state).to(device))  # Should all have the same action
            Q_vals = Q_vals.max(dim=1, keepdim=True)[0].detach().cpu().numpy()

            # No reason for calculating robust estimator in the real way if it gets masked anyway
            if mask[i] == 0:
                robust_estimator = 0
            else:
                robust_estimator, beta_max = distributionalQLearning.robust_estimator(X_p=state, y_p=next_state,
                                                                                      X_v=next_state, y_v=Q_vals,
                                                                                      delta=self.delta)
                robust_estimator = robust_estimator


            robust_estimators.append(robust_estimator)

        # Debug text - MUST BE CHANGED BEFORE CAN USE
        # print(f'X_p: {np.mean(state)} \n y_p: {np.mean(next_state)} \n y_v: {np.mean(y_v)}')
        # print("Robust estimator", robust_estimator)

        robust_estimators = torch.FloatTensor(robust_estimators).to(device).reshape(-1, 1) * self.robust_factor # -1 because that works better? #unsqueezing because im stupid and lazy
        robust_estimators = rewards - self.gamma * robust_estimators * mask

        # mask = 1 - done  # Remove effect from those that are done

        # calculate dqn loss
        loss = F.smooth_l1_loss(current_q_values, robust_estimators, reduction='mean') # reduction same as default, no worries

        return loss

if __name__ == "__main__":

    # environment
    # line_length = 500
    env = SumoPPEnv()

    seed = 777
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    seed_torch(seed)


    obs_dim = env.obs_dim
    action_dim = env.action_dim
    batch_size = 40
    fineness = 100
    ripe_when = None
    state_max, state_min = np.array([env.max_min[0]]), np.array([env.max_min[1]])
    ready_when = 10
    num_neighbours = 2
    bin_size = 1000
    replay_buffer = TheCoolerReplayBuffer(obs_dim=obs_dim, bin_size=bin_size, batch_size=batch_size, fineness=fineness,
                                          num_actions=action_dim, state_max=state_max, state_min=state_min,
                                          ripe_when=ripe_when, ready_when=ready_when, num_neighbours=num_neighbours,
                                          tb=True)

    num_frames = 2000

    # parameters
    fineness = 100
    state_dim = 1
    action_dim = 3
    batch_size = 40
    grad_batch_size = 10
    replay_buffer_size = 500
    max_min = [[env.cliff_position],[0]]
    epsilon_decay = 1/1500
    ripe_when = None # Just batch size
    delta = 0.5 # Should basically be same as DQN with such a small value

    # TODO: PERHAPS JUST PASS A PREMADE REPLAY BUFFER TO THE SUMO AGENT TO AVOID SO MANY PARAMETERS?
    agent = RobustSumoAgent(env=env, replay_buffer=replay_buffer, grad_batch_size=grad_batch_size, delta=delta,
                            epsilon_decay=epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None)
    agent.device = 'cuda'

    agent.train(num_frames)
    scores = agent.test(test_games=100, render_games=30)
    i = 5
    print(scores, np.mean(scores))
