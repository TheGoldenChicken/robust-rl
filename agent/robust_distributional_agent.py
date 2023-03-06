
import rl.agent
from collections import defaultdict
import random
import numpy as np


class robust_distributional_agent(rl.agent.ShallowAgent):
    
    def __init__(self, env, gamma = 0.9, delta = 1, epsilon = 0.5, tol = 0.05):
        super().__init__(env)
        
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.tol = tol
        
        self.lr = lambda t : 1/(1+(1-self.gamma)*(t-1))
        self.t = 0
    
    # Returns True if the environment is done (won or lost)
    def next(self) -> bool:
        
        def locate_maxima(f, x0, tol = 5e-2, max_iter = 100):
            x = x0
            for i in range(max_iter):
                f_prime = lambda x : (f(x+tol)-f(x-tol))/(2*tol)
                x += f_prime(x)
                if(x <= tol): x = 2*tol
                if(abs(f_prime(x)) < tol): break
                if(np.isnan(np.array(x))):
                    return tol*5
            return x
        
        def fDelta_r(N, reward):
            
            r_max = max(-reward/0.05)
            
            part_1 = lambda alpha : -alpha*np.log(1/(2**(N+1))) + r_max + \
                np.log(sum([np.exp(-reward[i]/(alpha)-r_max) for i in range(int(2**(N+1)))]) + 1e-5) - alpha*self.delta
            part_2 = lambda alpha : -alpha*np.log(1/(2**N)) + r_max + \
                np.log(sum([np.exp(-reward[2*i]/(alpha)-r_max) for i in range(int(2**(N)))]) + 1e-5) - alpha*self.delta
            part_3 = lambda alpha : -alpha*np.log(1/(2**N)) + r_max + \
                np.log(sum([np.exp(-reward[2*i-1]/(alpha)-r_max) for i in range(int(2**(N)))]) + 1e-5) - alpha*self.delta

            Delta_r = part_1(locate_maxima(lambda x : part_1(x), x0 = 1))
            Delta_r -= 1/2 * part_2(locate_maxima(lambda x : part_2(x), x0 = 1))
            Delta_r -= 1/2 * part_3(locate_maxima(lambda x : part_3(x), x0 = 1))
            
            return Delta_r
        
        def fDelta_q(N, state_):
            
            v_max = max([np.exp(-max([self.Q[(state_[i], b)] / 0.05 for b in self.env.A(state_[i])])) \
                                                           for i in range(int(2**(N+1)))])
            
            part_1 = lambda beta : -beta*np.log(1/(2**(N+1))) + v_max + \
                                                       np.log(sum([np.exp(-max([self.Q[(state_[i], b)] / (beta) for b in self.env.A(state_[i])])-v_max) + 1e-5 \
                                                           for i in range(int(2**(N+1)))])) - beta*self.delta
            part_2 = lambda beta : -beta*np.log(1/(2**N)) + v_max + \
                                                       np.log(sum([np.exp(-max([self.Q[(state_[2*i], b)] / (beta) for b in self.env.A(state_[2*i])])-v_max) + 1e-5 \
                                                           for i in range(int(2**(N)))])) - beta*self.delta
            part_3 = lambda beta : -beta*np.log(1/(2**N)) + v_max + \
                                                       np.log(sum([np.exp(-max([self.Q[(state_[2*i-1], b)] / (beta) for b in self.env.A(state_[2*i-1])])-v_max) + 1e-5 \
                                                           for i in range(int(2**(N)))])) - beta*self.delta
            
            Delta_q = part_1(locate_maxima(lambda x : part_1(x), x0 = 1))
            Delta_q -= 1/2 * part_2(locate_maxima(lambda x : part_2(x), x0 = 1))
            Delta_q -= 1/2 * part_3(locate_maxima(lambda x : part_3(x), x0 = 1))
        
            return Delta_q
        
        self.t += 1
        alpha_t = self.lr(self.t)
        Q_ = defaultdict(lambda : 0)
        
        for state in self.env.get_states():
            actions = self.env.A(state)
            for action in actions:
                p = lambda n : self.epsilon*(1-self.epsilon)**(n)
                cp = lambda n : 1/p(n)
                N = cp(random.random())
                samples = np.array([self.env.step(state, [action]) for _ in range(int(2**(N+1)))])

                Delta_r = fDelta_r(N, samples[:,1])
                Delta_q = fDelta_q(N, samples[:,0])
                
                R_rob = samples[0][1] + Delta_r/p(N)
                T_rob = max([self.Q[(samples[0][0], b)] for b in self.env.A(samples[0][0])]) + Delta_q/p(N)
                T_rob_e = R_rob + self.gamma*T_rob
                
                Q_[(state,action)] = (1-alpha_t)*self.Q[(state, action)] + alpha_t*T_rob_e
        
        converged = True
        # Check convergence
        for sa, q in self.Q.items():
            if abs(q - Q_[sa]) > self.tol:
                converged = False
                break
        
        self.Q = Q_
        
        
        if(converged): return True
        return False
    