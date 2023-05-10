from sumo_agent import SumoAgent
import torch
import torch.nn.functional as F
import distributionalQLearning2 as distributionalQLearning
import numpy as np
from replay_buffer import ReplayBuffer
from sumo_pp import SumoPPEnv
from typing import Dict
from network import Network

class DQNSumoAgent(SumoAgent):
    def __init__(self, env, replay_buffer, epsilon_decay, target_update,
                 max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None):
        super().__init__(env, replay_buffer, epsilon_decay, max_epsilon, min_epsilon, gamma, model_path)

        self.target_update = target_update
        self.dqn_target = Network(env.obs_dim, env.action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.state_normalizer = lambda state: (state-0)/(1200-0)

    def get_samples(self) -> tuple[dict, ]:
        """
        Pretty self-explanatory for this one tbh
        """
        return (self.replay_buffer.sample_batch(),)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        state = self.state_normalizer(state)
        next_state = self.state_normalizer(next_state)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

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

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            current_episode_score += reward

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

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()


            # plotting
                if frame_idx % plotting_interval == 0:
                    self._plot(frame_idx, scores, losses, epsilons)
                    #self._special_plot(episode_scores, epsilons, losses, frame_idx)
                    print(frame_idx, loss, self.epsilon, )

        print("Training complete")

        if save_model is not None:
            try:
                torch.save(self.dqn.state_dict(), save_model)
            except:
                print("ERROR! Could not save model!")

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

if __name__ == "__main__":

    # environment
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
    batch_size =  32
    size = 50000

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=size, batch_size=batch_size, ready_when=batch_size)

    # parameters
    epsilon_decay = 1/2000
    target_update = 100

    agent = DQNSumoAgent(env=env, replay_buffer=replay_buffer, epsilon_decay=epsilon_decay, target_update=target_update,
                         max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None)

    num_frames = 2000
    agent.train(num_frames)
    scores = agent.test(test_games=1000, render_games=10)
    print(scores, np.mean(scores))
