class ContinuousEnv:
    
    def __init__(self) -> None:
        # List of players
        self.players = list()
    
    # Return the next state and reward
    def step(self, state : tuple, actions : list(tuple)) -> list(tuple) | list(int):
        
        # Return next state and reward
        return [()], [0]
    
    # Return a list of all possible actions
    def A(self, state) -> list:
        
        # Return a list of all possible actions
        return list()
    
    # Reset all players
    def reset(self) -> None:
        for player in self.players:
            player.reset()

    def init_render(self):
        pass
    
    def render(self):
        pass
    
class DiscreteEnv(ContinuousEnv):
    
    def __init__(self) -> None:
        super.__init__()
    
    def get_transistion_probabilities(self, state, action) -> dict:
        
        # Return a dictionary with the next state and reward as key and the probability as value
        return dict()
    
    def get_states(self) -> list:
        
        # Return a list of all possible states
        return list()