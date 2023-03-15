from tqdm import tqdm

class Manager:
    
    def __init__(self, agent, render = False, renderOptions = []) -> None:
        self.agent = agent
        self.renderOptions = renderOptions
        self.render = render
        if(self.render): self.agent.env.init_render(*self.renderOptions)
    
    # Return iteration number
    def run(self, iterations = None):
        
        def iteration():
            if(self.agent.next()): return True
            
            if(self.render): self.agent.env.render(self.agent)
            
            if self.agent.env.is_terminal(self.agent.state):
                self.agent.state = self.agent.env.reset()
                return False
        
        if(iterations == None):
            i = 0
            while True:
                i += 1
                if(iteration()): return i
        else:
            for i in tqdm(range(iterations)):
                if(iteration()): return i
            return i
        