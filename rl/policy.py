from abc import abstractmethod
import random

class Policy:
    
    def __init__(self, env) -> None:
        self.env = env
    
    @abstractmethod
    def get_action(self, agent):
        pass
    
class Random(Policy):
    
    def __init__(self, env) -> None:
        self.env = env
    
    def get_action(self, agent):
        actions = self.env.A(agent.state)
        return actions[random.randint(0, len(actions) - 1)]
    
class EpsilonGreedy(Policy):
    
    def __init__(self, env, epsilon = 0.1, decay = 1) -> None:
        """
        Decayed epsilon greedy policy
        At decay = 1: Classic epsilon greedy policy
        At epsilon = 0: Greedy policy
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.decay = decay
    
    def get_action(self, agent):
        actions = self.env.A(agent.state)
        self.epsilon *= self.decay
        if(random.random() < self.epsilon):
            return actions[random.randint(0, len(actions) - 1)]
        else:
            return max(actions, key = lambda action: agent.Q[(agent.state, action)])
        
        
        
        
    