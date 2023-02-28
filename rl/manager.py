class Manager:
    
    def __init__(self, env, agent, render = False, renderOptions = []) -> None:
        self.env = env
        self.agent = agent
        
        self.renderOptions = renderOptions
        self.render = render
        if(self.render): self.env.init_render(*self.renderOptions)
        
    def run(self, iterations = None):
        
        def iteration():
            done = self.agent.next()
            
            if(self.render): self.env.render(*self.renderOptions)
            
            if(done): return True
            else: return False
        
        if(iterations == None):
            while True:
                if(iteration()): return
        else:
            for _ in range(iterations):
                if(iteration): return
        