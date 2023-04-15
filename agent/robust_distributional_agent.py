
import rl.agent
from collections import defaultdict
import random
import numpy as np
import pygame
import matplotlib.pyplot as plt

class robust_distributional_agent(rl.agent.ShallowAgent):
    
    def __init__(self, env, gamma = 0.9, delta = 1, epsilon = 0.5, tol = 0.05):
        super().__init__(env)
        
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.tol = tol
        
        self.lr = lambda t : 1/(1+(1-self.gamma)*(t-1))
        self.t = 0
        
        self.total_samples = 0
        
        self.Q = defaultdict(lambda : 0)
    
    # Returns True if the environment is done (won or lost)
    def next(self) -> bool:
        
        def locate_maxima(f, x0, tol = 1e-3, max_iter = 100):
            
            def get_not_nan(n):
                while(np.isnan(np.array(f(n)))):
                    n += tol
                if(n < 0): print(n)
                return n
            
            x = x0
            for i in range(max_iter):
                f_prime = lambda x : (f(x+tol)-f(x-tol))/(2*tol)
                grad = f_prime(x)
                if(abs(f_prime(x)) < tol):
                    break
                x += grad
                if(x <= tol):
                    if(grad < 0):
                        return get_not_nan(tol)
                    else: x = 2*tol
                if(np.isnan(np.array(x))):
                    return get_not_nan(tol)
            return get_not_nan(x)
            
        def fDelta_r(N, reward):
            # reward = -reward
            # r_max = max(-reward/0.05)
            
            def part_1(alpha):
                r = reward[:2**(N+1)+1]
                return -alpha*(np.log(1/(2**(N+1))) \
                               + np.logaddexp.reduce(-r/alpha)) - alpha*self.delta

            def part_2(alpha):
                r = reward[2:(2**N)*2+1:2]
                if len(r) == 0: r = np.array([0])
                return -alpha*(np.log(1/(2**N)) \
                               + np.logaddexp.reduce(-r/alpha)) - alpha*self.delta

            def part_3(alpha):
                r = np.roll(reward,1)[:(2**N)*2+1:2]
                return -alpha*(np.log(1/(2**N)) \
                               + np.logaddexp.reduce(-r/alpha)) - alpha*self.delta

            Delta_r = part_1(locate_maxima(lambda x : part_1(x), x0 = 1))
            Delta_r -= 1/2 * part_2(locate_maxima(lambda x : part_2(x), x0 = 1))
            Delta_r -= 1/2 * part_3(locate_maxima(lambda x : part_3(x), x0 = 1))
            
            # Delta_r = part_1(0.001)
            # Delta_r -= 1/2 * part_2(0.001)
            # Delta_r -= 1/2 * part_3(0.001)

            return Delta_r
        
        def fDelta_q(N, state_):
            
            def part_1(beta):
                s_ = tuple(state_[:2**(N+1)+1])
                # Array of max Q values for each state in s_
                Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])

                return -beta*np.log(1/(2**(N+1))) - beta*np.log(2**(N+1)) - 1 - beta*np.logaddexp.reduce(-Q_max) - beta*self.delta

            def part_2(beta):
                s_ = tuple(state_[1:(2**N)+1:2])
                if len(s_) == 0: s_ = np.array([0])
                Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])
                if len(s_) >= 10:
                    print("here now")
                return -beta*np.log(1/(2**N)) - beta*np.log(2**N) - 1 - beta*np.logaddexp.reduce(-Q_max) - beta*self.delta

            def part_3(beta):
                s_ = tuple(state_[:(2**N)+1:2])
                Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])

                return -beta*np.log(1/(2**N)) - beta*np.log(2**N) - 1 - beta*np.logaddexp.reduce(-Q_max) - beta*self.delta

            # Delta_q = part_1(locate_maxima(lambda x : part_1(x), x0 = 1))
            # Delta_q -= 1/2 * part_2(locate_maxima(lambda x : part_2(x), x0 = 1))
            # Delta_q -= 1/2 * part_3(locate_maxima(lambda x : part_3(x), x0 = 1))

            Delta_q = part_1(0.001)
            Delta_q -= 1/2 * part_2(0.001)
            Delta_q -= 1/2 * part_3(0.001)

            return Delta_q
        
        self.t += 1
        alpha_t = self.lr(self.t)
        Q_ = defaultdict(lambda : 0)
        
        for state in self.env.get_states():
            actions = self.env.A(state)
            for action in actions:
                p = lambda n : self.epsilon*(1-self.epsilon)**n

                N = None
                while True:
                    l = 0.5
                    k = 1

                    z = np.round(np.random.exponential(scale = 1/l, size = 1)[0])
                    u = np.random.uniform(low = 0, high = k*z, size = 1)

                    if u <= p(z):
                        # accept
                        N = int(z)
                        break
                # max_p = 100
                # # prop_adjust = sum([p(i) for i in np.arange(0,max_p)])
                # N = np.random.choice(np.arange(0,max_p), 1, replace = True,
                #                      p = [p(i) for i in np.arange(0,max_p)])[0]
                # print()
                if(N >= 15): print("\n>>> N:", N, "| p(N) =", p(N), "| Action:", action, "| State:", state)
                
                samples = np.array([self.env.step(state, action) for _ in range(2**(N+1))])

                self.total_samples += 2**(N+1)
                
                Delta_r = fDelta_r(N, samples[:,1])
                Delta_q = fDelta_q(N, samples[:,0])
                Delta_r = 0
                Delta_q = 0
                
                R_rob = samples[0][1] + Delta_r/p(N)
                T_rob = max([self.Q[(samples[0][0], b)] for b in self.env.A(samples[0][0])]) + Delta_q/p(N)
                T_rob_e = R_rob + self.gamma*T_rob
                
                Q_[state,action] = (1-alpha_t)*self.Q[state, action] + alpha_t*T_rob_e
        
        Q_diffs = []
        for key in self.Q.keys():
            Q_diffs.append(np.abs(self.Q[key]-Q_[key]))
        distance = np.max(Q_diffs)
        
        if distance < self.tol:
            print(">>> (CONVERGED) Diff Inf Norm:", distance)
            return True
        elif(self.t%100 == 0):
            print(">>> Diff Inf Norm:", distance)
        
        self.Q = Q_
        
        return False
    