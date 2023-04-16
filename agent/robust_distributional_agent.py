
# import rl.agent
# from collections import defaultdict
# import random
# import numpy as np
# import pygame
# import matplotlib.pyplot as plt

# class robust_distributional_agent(rl.agent.ShallowAgent):
    
#     def __init__(self, env, gamma = 0.9, delta = 1, epsilon = 0.5, tol = 0.05):
#         super().__init__(env)
        
#         self.gamma = gamma
#         self.delta = delta
#         self.epsilon = epsilon
#         self.tol = tol
        
#         self.lr = lambda t : 1/(1+(1-self.gamma)*(t-1))
#         self.t = 0
        
#         self.total_samples = 0
        
#         self.Q = defaultdict(lambda : 0)
    
#     # Returns True if the environment is done (won or lost)
#     def next(self) -> bool:
        
#         def locate_maxima(f, x0, tol = 1e-3, max_iter = 100):
            
#             def get_not_nan(n):
#                 while(np.isnan(np.array(f(n)))):
#                     n += tol
#                 if(n < 0): print(n)
#                 return n
            
#             x = x0
#             for i in range(max_iter):
#                 f_prime = lambda x : (f(x+tol)-f(x-tol))/(2*tol)
#                 grad = f_prime(x)
#                 if(abs(f_prime(x)) < tol):
#                     break
#                 x += grad
#                 if(x <= tol):
#                     if(grad < 0):
#                         return get_not_nan(tol)
#                     else: x = 2*tol
#                 if(np.isnan(np.array(x))):
#                     return get_not_nan(tol)
#             return get_not_nan(x)
            
#         def fDelta_r(N, reward):
#             # reward = -reward
#             # r_max = max(-reward/0.05)
            
#             def part_1(alpha):
#                 r = reward[:2**(N+1)+1]
#                 return -alpha*(np.log(1/(2**(N+1))) \
#                                + np.logaddexp.reduce(-r/alpha)) - alpha*self.delta

#             def part_2(alpha):
#                 r = reward[2:(2**N)*2+1:2]
#                 if len(r) == 0: r = np.array([0])
#                 return -alpha*(np.log(1/(2**N)) \
#                                + np.logaddexp.reduce(-r/alpha)) - alpha*self.delta

#             def part_3(alpha):
#                 r = np.roll(reward,1)[:(2**N)*2+1:2]
#                 return -alpha*(np.log(1/(2**N)) \
#                                + np.logaddexp.reduce(-r/alpha)) - alpha*self.delta

#             Delta_r = part_1(locate_maxima(lambda x : part_1(x), x0 = 1))
#             Delta_r -= 1/2 * part_2(locate_maxima(lambda x : part_2(x), x0 = 1))
#             Delta_r -= 1/2 * part_3(locate_maxima(lambda x : part_3(x), x0 = 1))
            
#             # Delta_r = part_1(0.001)
#             # Delta_r -= 1/2 * part_2(0.001)
#             # Delta_r -= 1/2 * part_3(0.001)

#             return Delta_r
        
#         def fDelta_q(N, state_):
            
#             def part_1(beta):
#                 s_ = tuple(state_[:2**(N+1)+1])
#                 # Array of max Q values for each state in s_
#                 Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])

#                 return -beta*np.log(1/(2**(N+1))) - beta*np.log(2**(N+1)) - 1 - beta*np.logaddexp.reduce(-Q_max) - beta*self.delta

#             def part_2(beta):
#                 s_ = tuple(state_[1:(2**N)+1:2])
#                 if len(s_) == 0: s_ = np.array([0])
#                 Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])
#                 if len(s_) >= 10:
#                     print("here now")
#                 return -beta*np.log(1/(2**N)) - beta*np.log(2**N) - 1 - beta*np.logaddexp.reduce(-Q_max) - beta*self.delta

#             def part_3(beta):
#                 s_ = tuple(state_[:(2**N)+1:2])
#                 Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])

#                 return -beta*np.log(1/(2**N)) - beta*np.log(2**N) - 1 - beta*np.logaddexp.reduce(-Q_max) - beta*self.delta

#             # Delta_q = part_1(locate_maxima(lambda x : part_1(x), x0 = 1))
#             # Delta_q -= 1/2 * part_2(locate_maxima(lambda x : part_2(x), x0 = 1))
#             # Delta_q -= 1/2 * part_3(locate_maxima(lambda x : part_3(x), x0 = 1))

#             Delta_q = part_1(0.001)
#             Delta_q -= 1/2 * part_2(0.001)
#             Delta_q -= 1/2 * part_3(0.001)

#             return Delta_q
        
#         self.t += 1
#         alpha_t = self.lr(self.t)
#         Q_ = defaultdict(lambda : 0)
        
#         for state in self.env.get_states():
#             actions = self.env.A(state)
#             for action in actions:
#                 p = lambda n : self.epsilon*(1-self.epsilon)**n

#                 N = None
#                 while True:
#                     l = 0.5
#                     k = 1

#                     z = np.round(np.random.exponential(scale = 1/l, size = 1)[0])
#                     u = np.random.uniform(low = 0, high = k*z, size = 1)

#                     if u <= p(z):
#                         # accept
#                         N = int(z)
#                         break
#                 # max_p = 100
#                 # # prop_adjust = sum([p(i) for i in np.arange(0,max_p)])
#                 # N = np.random.choice(np.arange(0,max_p), 1, replace = True,
#                 #                      p = [p(i) for i in np.arange(0,max_p)])[0]
#                 # print()
#                 if(N >= 15): print("\n>>> N:", N, "| p(N) =", p(N), "| Action:", action, "| State:", state)
                
#                 samples = np.array([self.env.step(state, action) for _ in range(2**(N+1))])

#                 self.total_samples += 2**(N+1)
                
#                 Delta_r = fDelta_r(N, samples[:,1])
#                 Delta_q = fDelta_q(N, samples[:,0])
#                 Delta_r = 0
#                 Delta_q = 0
                
#                 R_rob = samples[0][1] + Delta_r/p(N)
#                 T_rob = max([self.Q[(samples[0][0], b)] for b in self.env.A(samples[0][0])]) + Delta_q/p(N)
#                 T_rob_e = R_rob + self.gamma*T_rob
                
#                 Q_[state,action] = (1-alpha_t)*self.Q[state, action] + alpha_t*T_rob_e
        
#         Q_diffs = []
#         for key in self.Q.keys():
#             Q_diffs.append(np.abs(self.Q[key]-Q_[key]))
#         distance = np.max(Q_diffs)
        
#         if distance < self.tol:
#             print(">>> (CONVERGED) Diff Inf Norm:", distance)
#             return True
#         elif(self.t%100 == 0):
#             print(">>> Diff Inf Norm:", distance)
        
#         self.Q = Q_
        
#         return False

import rl.agent
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import random
import pygame
from scipy.stats import geom

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

    def f_stable(self, x, c, K):
        """
        The function to be minimized.
        This version is more numerically stable.
        """

        c_star = np.max(c)

        tmp1 = -x*(c_star - np.log(2**K))
        tmp2 = -x*np.logaddexp.reduce(-c/x - c_star)
        tmp3 = -x*self.delta

        return tmp1 + tmp2 + tmp3


    def f_prime(self, x, c, K):
        """
        The derivative of the function to be minimized.
        Use f_prime_approx instead due to numerical instability.
        """

        c_star = np.max(c)

        tmp1 = -c_star + np.log(2**K)
        tmp2 = -np.logaddexp.reduce(-c/x - c_star)
        tmp3 = -np.sum(c*np.exp(-c/x - c_star)) / (x * np.sum(np.exp(-c/x - c_star)))
        tmp4 = -self.delta

        return tmp1 + tmp2 + tmp3 + tmp4
    
    def f_prime_approx(self, x, c, K, tol):
        """
        An approximation of the derivative of the function to be minimized.
        This should be used instead of f_prime due to numerical instability
        """

        return (self.f_stable(x+tol, c, K) - self.f_stable(x-tol, c, K)) / (2*tol)

    def maximize(self, c, K, tol = 1e-3):
        """
        Maximize f_stable with respect to x by using the derivative f_prime.
        Also note that f_prime is either monotonic decreasing or only has one maximum.
        Also note that x is always positive.
        """
        
        x_min = tol*2
        x_max = 10
        
        # If f_prime(tol) < 0 the function is monotonic decreasing.
        if self.f_prime_approx(x_min, c, K, tol) < 0:
            return x_min

        while True:
            # If f_prime(10) > 0, adjust the maximum
            if self.f_prime_approx(x_max, c, K, tol) > 0:
                x_max *= 2
            else: 
                break
        
        # Find the maximum using devide and conquer
        while True:
            x_mid = (x_min + x_max) / 2
            if x_max - x_min < tol:
                return x_mid
            f_prime_x_mid = self.f_prime_approx(x_mid, c, K, tol)
            if f_prime_x_mid > 0 + tol:
                x_min = x_mid
            elif f_prime_x_mid < 0 - tol:
                x_max = x_mid
            else:
                return x_mid

    def fDelta_q(self, N, state_):
        """
        The Delta_q function.
        """
        # Section 1
        c = state_[:2**(N+1)+1]
        c = np.array([max([self.Q[c_i, b] for b in self.env.A(c_i)]) for c_i in c])
        K = N+1

        s1 = self.f_stable(self.maximize(c, K), c, K)

        # Section 2
        c = state_[1:(2**N)+1:2]
        c = np.array([max([self.Q[c_i, b] for b in self.env.A(c_i)]) for c_i in c])
        K = N

        s2 = self.f_stable(self.maximize(c, K), c, K)

        # Section 3
        c = state_[:(2**N)+1:2]
        c = np.array([max([self.Q[c_i, b] for b in self.env.A(c_i)]) for c_i in c])
        K = N

        s3 = self.f_stable(self.maximize(c, K), c, K)

        return s1 - (1/2)*s2 - (1/2)*s3

    def fDelta_r(self, N, reward):
        """
        The Delta_r function.
        """
        # Section 1
        c = reward[:2**(N+1)+1]
        K = N+1

        s1 = self.f_stable(self.maximize(c, K), c, K)

        # Section 2
        c = reward[1:(2**N)+1:2]
        K = N

        s2 = self.f_stable(self.maximize(c, K), c, K)

        # Section 3
        c = reward[:(2**N)+1:2]
        K = N

        s3 = self.f_stable(self.maximize(c, K), c, K)

        return s1 - (1/2)*s2 - (1/2)*s3

    def next(self):
        self.t += 1
        alpha_t = self.lr(self.t)
        Q_ = defaultdict(lambda : 0)
        
        for state in self.env.get_states():
            actions = self.env.A(state)
            for action in actions:
                loc = -1
                N = geom.rvs(p = self.epsilon, loc = loc, size = 1)[0]

                samples = np.array([self.env.step(state, action) for _ in range(2**(N+1))])

                self.total_samples += 2**(N+1)
                
                Delta_q = self.fDelta_q(N, samples[:,0])
                Delta_r = self.fDelta_r(N, samples[:,1])
                
                p_N = geom.pmf(N, p = self.epsilon, loc = loc)

                R_rob = samples[0][1] + Delta_r/p_N
                T_rob = max([self.Q[(samples[0][0], b)] for b in self.env.A(samples[0][0])]) + Delta_q/p_N
                T_rob_e = R_rob + self.gamma*T_rob
                
                Q_[state,action] = (1-alpha_t)*self.Q[state, action] + alpha_t*T_rob_e
        
        Q_diffs = []
        for key in self.Q.keys():
            Q_diffs.append(np.abs(self.Q[key]-Q_[key]))
        distance = np.max(Q_diffs)
        
        if distance < self.tol:
            print(">>> (CONVERGED) Diff Inf Norm:", distance, "| Total Samples:", self.total_samples)
            return True
        elif(self.t%100 == 0):
            print(">>> Diff Inf Norm:", distance)
        
        self.Q = Q_
        
        return False