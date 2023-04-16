#%%
import rl.agent
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import random
import pygame

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

    def f(self, x, c, K, delta):
        """The function to be minimized."""

        return - x*np.log((1/2**K)*np.sum(np.exp(-c/x))) - x*delta

    def f_stable(self, x, c, K, delta):
        """
        The function to be minimized.
        This version is more numerically stable.
        """

        c_star = np.max(c)

        tmp1 = -x*(c_star - np.log(2**K))
        tmp2 = -x*np.logaddexp.reduce(-c/x - c_star)
        tmp3 = -x*delta

        return tmp1 + tmp2 + tmp3


    def f_prime(self, x, c, K, delta):
        """
        The derivative of the function to be minimized.
        """

        c_star = np.max(c)

        tmp1 = -c_star + np.log(2**K)
        tmp2 = -np.logaddexp.reduce(-c/x - c_star)
        tmp3 = -np.sum(c*np.exp(-c/x - c_star)) / (x * np.sum(np.exp(-c/x - c_star)))
        tmp4 = -delta

        return tmp1 + tmp2 + tmp3 + tmp4

    def f_prime_approx(self, x, c, K, delta, tol):
        """
        An approximation of the derivative of the function to be minimized.
        """

        return (self.f_stable(x+tol, c, K, delta) - self.f_stable(x-tol, c, K, delta)) / (2*tol)

    def maximize(self, c, K, delta, tol = 1e-3):
        """
        Maximize f_stable with respect to x by using the derivative f_prime.
        Also note that f_prime is either monotonic decreasing or only has one maximum.
        Also note that x is always positive.
        """
        
        x_min = tol
        x_max = 10
        
        # If f_prime(tol) < 0 the function is monotonic decreasing.
        if self.f_prime(x_min, c, K, delta) < 0:
            return x_min

        while True:
            # If f_prime(10) > 0, adjust the maximum
            if self.f_prime(x_max, c, K, delta) > 0:
                x_max *= 2
            else: 
                break
        
        # Find the maximum using devide and conquer
        while True:
            x_mid = (x_min + x_max) / 2
            f_prime_x_mid = self.f_prime(x_mid, c, K, delta)
            if f_prime_x_mid > 0 + tol:
                x_min = x_mid
            elif f_prime_x_mid < 0 - tol:
                x_max = x_mid
            else:
                return x_mid

    def fDelta_q(self, N, state_):
        # Section 1
        c = tuple(state_[:2**(N+1)+1])
        c = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])
        K = N+1

        s1 = self.maximize(c, K, delta)

        # Section 2
        c = tuple(state_[1:(2**N)+1:2])
        c = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])
        K = N

        s2 = self.maximize(c, K, delta)

        # Section 3
        c = tuple(state_[:(2**N)+1:2])
        c = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])
        K = N

        s3 = self.maximize(c, K, delta)

        return s1 - (1/2)*(s2 + s3)

    def fDelta_r(self, N, reward):
        # Section 1
        c = tuple(reward[:2**(N+1)+1])
        K = N+1

        s1 = self.maximize(c, K, delta)

        # Section 2
        c = tuple(reward[1:(2**N)+1:2])
        K = N

        s2 = self.maximize(c, K, delta)

        # Section 3
        c = tuple(reward[:(2**N)+1:2])
        K = N

        s3 = self.maximize(c, K, delta)

        return s1 - (1/2)*(s2 + s3)

    def next(self):
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
                
                Delta_r = self.fDelta_r(N, samples[:,1])
                Delta_q = self.fDelta_q(N, samples[:,0])
                
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


#%%
X = np.linspace(0.01, 1, 1000)
c = np.array([1, 1, 0.5, 2])
K = int(math.floor(np.log2(len(c))))
delta = 0.1
tol = 1e-3

maxima = maximize(c, K, delta, tol)

plt.plot(X, [f(x, c, K, delta) for x in X], label='f')
plt.plot(X, [f_stable(x, c, K, delta) for x in X], label='f_stable')
plt.scatter(maxima, f_stable(maxima, c, K, delta), label='max')
plt.show()

print("Monotonic decreasing?", is_monotonic_decreasing(c, K))

# plt.plot(X, [f_prime(x, c, K, delta) for x in X], label='f_prime')
# plt.plot(X, [f_prime_approx(x, c, K, delta, tol) for x in X], label='f_prime_approx')
# plt.legend()
# plt.show()
# %%
# Testing the speed of calculation

import time

elaped_time_list = []
for i in range(1000):
    start_time = time.time()

    [f_prime_approx(x, c, K, delta, tol) for x in X]

    end_time = time.time()
    elaped_time_list.append(end_time - start_time)

print("Average time (approximation): ", np.mean(elaped_time_list))
print("Absolute time (approximation): ", np.sum(elaped_time_list))

elaped_time_list = []
for i in range(1000):
    start_time = time.time()

    [f_prime(x, c, K, delta) for x in X]

    end_time = time.time()
    elaped_time_list.append(end_time - start_time)

print("Average time (exact): ", np.mean(elaped_time_list))
print("Absolute time (exact): ", np.sum(elaped_time_list))
