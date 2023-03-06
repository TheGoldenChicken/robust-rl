import rl.agent
from collections import defaultdict


class FirstVisitMonteCarlo(rl.agent.ShallowAgent):
    
    def __init__(self, env, policy, gamma = 0.95, lr = lambda N : 1/N) -> None:
        super().__init__(env)
        
        self.policy = policy
        self.lr = lr
        self.gamma = gamma
        
        self.N = defaultdict(lambda : 0)
        
        
    def next(self):
        
        self.state = self.env.reset()
        trajectory = []
        
        G = 0
        while not self.env.is_terminal(self.state):
            action = self.policy.get_action(self)
            action = tuple(action)
            
            if (self.state, action) not in trajectory:
                trajectory.append((self.state, action))
                self.N[(self.state, action)] += 1
                
            state_, reward = self.env.step(self.state, action)
            
            G = self.gamma*G + reward
            self.state = state_
        
        for state, action in trajectory:
            self.Q[(state, action)] += self.lr(G - self.Q[(state, action)])