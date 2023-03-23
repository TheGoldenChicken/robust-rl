import rl.agent
from collections import defaultdict

class replay_agent(rl.agent.ShallowAgent):

    def __init__(self, env, policy) -> None:
        super().__init__(env)

        self.policy = policy
        self.results = defaultdict(lambda : [])

        self.state = env.reset()

    def next(self):
        action = self.policy.get_action(self)

        state_, reward = self.env.step(self.state,action)

        self.results[state_].append(reward)

        self.state = state_
        return self.env.is_terminal(self.state)