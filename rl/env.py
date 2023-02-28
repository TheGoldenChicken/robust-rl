import rl.player

class ContinuousEnv:
    
    def __init__(self, playerOptions : list) -> None:
        # List of players
        self.players = list()
        self.state_size = 0
    
    # Return the next state and reward
    def step_(self, state : tuple, actions : list[tuple]) -> tuple[list[tuple],list[int]]:
        raise NotImplementedError("ContinuousEnv.step() is not implemented")
    
        # Return next state and reward
        return [()], [0]
    
    # Return a list of all possible actions
    def A(self, state) -> list:
        raise NotImplementedError("ContinuousEnv.A() is not implemented")
    
        # Return a list of all possible actions
        return list()
    
    def goal_test(self, state) -> bool:
        raise NotImplementedError("ContinuousEnv.goal_test() is not implemented")
        
        # Return True if the state is a goal state
        return False
    
    # Reset all players
    def reset(self) -> None:
        for player in self.players:
            player.reset()

    def init_render(self):
        raise NotImplementedError("ContinuousEnv.init_render() is not implemented")
    
        self.width = 800
        self.height = 600
        self.display = rl.display.displayHandler(self.width, self.height)
    
    def render(self):
        raise NotImplementedError("ContinuousEnv.render() is not implemented")
    
        self.display.update()
        
        return self.display.eventHandler()
    
class DiscreteEnv(ContinuousEnv):
    
    def __init__(self, playerOptions : list) -> None:
        super.__init__(playerOptions)
    
    def get_transistion_probabilities(self, state, action) -> dict:
        raise NotImplementedError("DiscreteEnv.get_transistion_probabilities() is not implemented")
        
        # Return a dictionary with the next state and reward as key and the probability as value
        return dict()
    
    def get_states(self) -> list:
        raise NotImplementedError("DiscreteEnv.get_states() is not implemented")
        
        # Return a list of all possible states
        return list()