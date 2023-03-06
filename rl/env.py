import rl.player
from abc import abstractmethod

class ContinuousEnv:
    
    def __init__(self) -> None:
        """
        Continuous environment. All environments must inherit from this class
        """
        # List of players
        self.players = list()
        self.state_size = int
    
    # Return the next state and reward
    @abstractmethod
    def step(self, state : tuple, actions : list[tuple]) -> tuple[tuple,int]:
        """
        Step method. Returns the next state and reward
        """
        # Return next state and reward
        return list, int
    
    # Return a list of all possible actions
    @abstractmethod
    def A(self, state) -> list:
        """
        Action method. Returns a list of all possible actions
        """
        raise NotImplementedError("ContinuousEnv.A() is not implemented")
    
        # Return a list of all possible actions
        # Each action is a tuple. The inner list represents the action for each player
        return list(list(tuple))
    
    @abstractmethod
    def is_terminal(self, state) -> bool:
        """
        Terminal state method. Returns True if the state is a terminal state (won or lost)
        """
        raise NotImplementedError("ContinuousEnv.is_terminal() is not implemented")
        
        # Return True if the state is a goal state
        return False
    
    @abstractmethod
    # Reset state by returning initial state
    def reset(self) -> None:
        """
        Reset method. Returns the initial state
        The method is called automatically when an agent is initialized
        """
        raise NotImplementedError("ContinuousEnv.reset() is not implemented")
        
        state = Tuple()
        return state
        
    def init_render(self, aps = 5):
        """
        Initialize the render method. Set up the display
        """
        raise NotImplementedError("ContinuousEnv.init_render() is not implemented")
    
        self.width = 800
        self.height = 600
        self.display = rl.display.displayHandler(self.width, self.height)
    
    def render(self, state):
        """
        Render method. Draw the current state
        Returns True if the display is closed
        """
        raise NotImplementedError("ContinuousEnv.render() is not implemented")
    
        self.display.update()
        
        return self.display.eventHandler()
    
class DiscreteEnv(ContinuousEnv):
    
    def __init__(self) -> None:
        """
        Discrete environment
        """
        super().__init__()
    
    # Get the probability transistion destribution for a given state and action
    # Only viable for discrete environments with a small state space
    def Ptd(self, state, action) -> dict:
        """
        Probability transistion destribution method. Returns a dictionary with the next state and reward as key and the probability as value
        """
        raise NotImplementedError("DiscreteEnv.get_transistion_probabilities() is not implemented")
        
        # Return a dictionary with the next state and reward as key and the probability as value
        return dict()
    
    def get_states(self) -> list:
        """
        Get all possible states. Not viable/implemented for large state spaces
        """
        raise NotImplementedError("DiscreteEnv.get_states() is not implemented")
        
        # Return a list of all possible states
        return list()