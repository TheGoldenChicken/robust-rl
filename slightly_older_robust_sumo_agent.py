from sumo_agent import SumoAgent
import torch
import torch.nn.functional as F
import distributionalQLearning2 as distributionalQLearning
import numpy as np
from replay_buffer import TheCoolerReplayBuffer
from sumo_pp import SumoPPEnv
from typing import Dict, List

class RobustSumoAgent(SumoAgent):
    def __init__(self, env, replay_buffer, epsilon_decay, max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None):
        super().__init__(env, replay_buffer, epsilon_decay, max_epsilon, min_epsilon, gamma, model_path)

    def get_samples(self) -> dict:
        """
        Should be updated for each individual agent type
        """
        samples, _ = self.replay_buffer.sample_from_scratch(K=self.replay_buffer.batch_size,
                                                      nn=self.replay_buffer.num_neighbours, num_samples=5)

        return samples
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

        state = samples["obs"]
        next_state = samples["next_obs"]
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = np.expand_dims(samples['done'], -1)  # Don't think we need this as a tensor... So discount reshaping here

        state = self.state_normalizer(state)
        next_state = self.state_normalizer(next_state)

        # TODO: CHECK IF V(S) ESTIMATE IS BASED ON CURRENT OR NEXT STATE
        Q_vals = self.dqn(torch.FloatTensor(state).to(device))  # Should all have the same action
        current_q_value = Q_vals.gather(1, action)
        Q_vals = Q_vals.max(dim=1, keepdim=True)[0].detach().cpu().numpy()

        robust_estimator = distributionalQLearning.robust_estimator(X_p=state, y_p=next_state,
                                                                    X_v=next_state, y_v=Q_vals,
                                                                    delta=0.5)

        # Debug text
        # print(f'X_p: {np.mean(state)} \n y_p: {np.mean(next_state)} \n y_v: {np.mean(y_v)}')
        # print("Robust estimator", robust_estimator)

        mask = 1 - done  # Remove effect from those that are done
        robust_estimator = reward + self.gamma * robust_estimator * mask * -1

        # calculate dqn loss
        loss = F.smooth_l1_loss(current_q_value, robust_estimator, reduction='mean') # reduction same as default, no worries

        return loss

if __name__ == "__main__":

    # environment
    line_length = 500
    env = SumoPPEnv(line_length=line_length)

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



    num_frames = 10000

    # parameters
    fineness = 100
    state_dim = 1
    action_dim = 3
    batch_size = 40
    replay_buffer_size = 500
    max_min = [[env.cliff_position],[0]]
    epsilon_decay = 1/2000
    ripe_when = 20

    # TODO: PERHAPS JUST PASS A PREMADE REPLAY BUFFER TO THE SUMO AGENT TO AVOID SO MANY PARAMETERS?
    agent = RobustSumoAgent(env=env, replay_buffer=replay_buffer, epsilon_decay=epsilon_decay, max_epsilon=1.0,
                            min_epsilon=0.1, gamma=0.99, model_path=None)

    agent.train(num_frames)
    scores = agent.test(test_games=100, render_games=30)
    i = 5
    print(scores, np.mean(scores))
