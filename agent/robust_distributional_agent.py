
import rl.agent
from collections import defaultdict
import random
import numpy as np


class robust_distributional_agent(rl.agent.ShallowAgent):
    
    def __init__(self, env, gamma = 0.5, delta = 1, epsilon = 0.5, tol = 0.05):
        super().__init__(env)
        
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.tol = tol
        
        self.lr = lambda t : 1/(1+(1-self.gamma)*(t-1))
        self.t = 0
    
    # Returns True if the environment is done (won or lost)
    def next(self, state) -> bool:
        
        def locate_maxima(f, x0, tol = 1e-2, max_iter = 100):
            x = x0
            for i in range(max_iter):
                f_prime = lambda x : (f(x+tol)-f(x-tol))/(2*tol)
                x += (0.1/np.sqrt(i+1))*f_prime(x)
                if(x <= tol): x = 2*tol
                if(abs(f_prime(x)) < tol): break
            return x
        
        def fDelta_r(N, reward):
            
            r_max = max(reward)
            
            part_1 = lambda N, r, alpha : -alpha*np.log((1/(2**(N+1)))*np.exp(r_max)* \
                sum([np.exp(-r[i]/(alpha)-r_max) for i in range(int(2**(N+1)))])) - alpha*self.delta
            part_2 = lambda N, r, alpha : -alpha*np.log((1/(2**(N)))*np.exp(r_max)* \
                sum([np.exp(-r[2*i]/(alpha)-r_max) for i in range(int(2**(N)))])) - alpha*self.delta
            part_3 = lambda N, r, alpha : -alpha*np.log((1/(2**(N)))*np.exp(r_max)* \
                sum([np.exp(-r[2*i-1]/(alpha)-r_max) for i in range(int(2**(N)))])) - alpha*self.delta

            Delta_r = part_1(N,reward, locate_maxima(lambda x : part_1(N, reward, x), x0 = 1))
            Delta_r -= 1/2 * part_2(N,reward, locate_maxima(lambda x : part_2(N, reward, x), x0 = 1))
            Delta_r -= 1/2 * part_3(N,reward, locate_maxima(lambda x : part_3(N, reward, x), x0 = 1))
            
            return Delta_r
        
        def fDelta_q(N, state_):
            
            v_max = max([np.exp(-max([self.Q[(state_[i], b)] for b in self.env.A(state_[i])])) \
                                                           for i in range(int(2**(N+1)))])
            
            part_1 = lambda N, s_, beta : -beta*np.log((1/(2**(N+1))) * \
                                                       np.exp(v_max)*sum([np.exp(-max([self.Q[(s_[i], b)] / (beta) for b in self.env.A(s_[i])])-v_max) \
                                                           for i in range(int(2**(N+1)))])) - beta*self.delta
            part_2 = lambda N, s_, beta : -beta*np.log((1/(2**(N))) *
                                                       np.exp(v_max)*sum([np.exp(-max([self.Q[(s_[2*i], b)] / (beta) for b in self.env.A(s_[2*i])])-v_max) \
                                                           for i in range(int(2**(N)))])) - beta*self.delta
            part_3 = lambda N, s_, beta : -beta*np.log((1/(2**(N))) *
                                                       np.exp(v_max)*sum([np.exp(-max([self.Q[(s_[2*i-1], b)] / (beta) for b in self.env.A(s_[2*i-1])])-v_max) \
                                                           for i in range(int(2**(N)))])) - beta*self.delta
            
            Delta_q = part_1(N,state_, locate_maxima(lambda x : part_1(N, state_, x), x0 = 1))
            Delta_q -= 1/2 * part_2(N,state_, locate_maxima(lambda x : part_2(N, state_, x), x0 = 1))
            Delta_q -= 1/2 * part_3(N,state_, locate_maxima(lambda x : part_3(N, state_, x), x0 = 1))
            
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
    