from tqdm import tqdm

class Manager:
    
    def __init__(self, env, agent, render = False, renderOptions = []) -> None:
        self.env = env
        self.agent = agent
        self.renderOptions = renderOptions
        self.render = render
        if(self.render): self.env.init_render(*self.renderOptions)
    
    # Return iteration number
    def run(self, iterations = None):
        
        def iteration():
            done = self.agent.next()
            
            if(self.render): self.env.render(self.agent)
            
            if(done): return True
            else: return False
        
        if(iterations == None):
            i = 0
            while True:
                i += 1
                if(iteration()): return i
        else:
            for i in tqdm(range(iterations)):
                if(iteration()): return i
        