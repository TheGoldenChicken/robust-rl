
class Player:
    
    def __init__(self, env) -> None:
        self.env = env
        
    # Update the variables of the player
    def update(self):
        raise NotImplementedError("Player.update() is not implemented")
    
    def reset(self):
        raise NotImplementedError("Player.reset() is not implemented")