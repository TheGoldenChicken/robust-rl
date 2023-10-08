from cliff_car_agent import CliffCarAgent
import torch
import torch.nn.functional as F
import distributionalQLearning as distributionalQLearning
import numpy as np
from replay_buffer import ReplayBuffer
from cliff_car_env import CliffCar
from typing import Dict
import time
import wandb
from tqdm import tqdm

class DQNCliffCarAgent(CliffCarAgent):
    def __init__(self, env, replay_buffer, network, epsilon_decay=0.0001, target_update=300,
                 learning_rate=0.0001, weight_decay=0.0001,
                 max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None, **kwargs):
        super().__init__(env, replay_buffer, network, epsilon_decay, max_epsilon, min_epsilon,
                         gamma, learning_rate, weight_decay, model_path)
        self.target_update = target_update
        self.dqn_target = network(env).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.state_normalizer = lambda state: (state - self.min)/(self.max - self.min)

        # Yeah, this isn't even the stupidest solution...
        self.tens_min,self.tens_max = torch.FloatTensor(self.min).to(self.device), torch.FloatTensor(self.max).to(self.device)
        self.tensor_normalizer = lambda state: (state - self.tens_min)/(self.tens_max - self.tens_min)


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

        state = self.tensor_normalizer(state)
        next_state = self.tensor_normalizer(next_state)
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)
        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def train(self, train_frames: int, test_interval = 200,
              test_games = 100, plot_path = None,
              wandb_active = False, silence_tqdm = False, **kwargs):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        if wandb_active: wandb.log({'score': score})

        current_episode_score = 0

        for frame in tqdm(range(1, train_frames + 1), disable=silence_tqdm):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            current_episode_score += reward

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                if wandb_active: wandb.log({'score': score})
                scores.append(score)
                score = 0

            # Update whether we're ready to train
            if not self.training_ready:
                self.training_ready = self.replay_buffer.training_ready()

            else:
                
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                if wandb_active: wandb.log({'loss': loss}) # Loss
                if wandb_active: wandb.log({'epsilon': self.epsilon})

                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

                if frame % test_interval == 0:
                    print(f">>> Testing at frame: {frame}. Time: {time.ctime()}")
                    self.test(test_games=test_games,
                            frame = frame,
                            plot_path = plot_path)

        print("Training complete")
        return scores, losses, epsilons

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())



if __name__ == "__main__":

    # environment
    env = CliffCar()

    seed = 777
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    seed_torch(seed)

    obs_dim = env.obs_dim
    batch_size = 64
    size = 50000

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=size, batch_size=batch_size, ready_when=batch_size)

    # parameters
    epsilon_decay = 1/4000
    target_update = 500

    agent = DQNCliffCarAgent(env=env, replay_buffer=replay_buffer, epsilon_decay=epsilon_decay, target_update=target_update,
                         max_epsilon=1.0, min_epsilon=0.1, gamma=0.99, model_path=None)

    num_frames = 8000
    agent.train(num_frames)
    # agent._plot_q_vals()

    agent.test(test_games=10, render_games=10)

    scores = agent.test(test_games=10, render_games=10)
    print(scores, np.mean(scores))
