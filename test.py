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


#%%
from scipy.stats import multivariate_normal
import numpy as np


mu = np.array([0, 0])
Sigma = np.array([[3, 1], [1, 4]])

A = np.array([[1, 0], [0, 1]])
b = np.array([0, 0])
c = 0

f1 = lambda beta, s_ : multivariate_normal.pdf(s_, mean = mu, cov = Sigma) * \
                       np.exp((s_.T @ A @ s_ + b.T @ s_ + c) * (-1/beta))

from itertools import product
# Numerical multivariate integration over f1 with respect to s_ using Monte Carlo
def monte_carlo_integration(f, beta, mean, cov, N):
    f_ = lambda s_ : f(beta, s_)

    x = []
    for i in range(len(mean)): # For each dimension
        x.append(np.linspace(start = mean[i] - cov[i,i]*5,
                             stop = mean[i] + cov[i,i]*5,
                             num = N))
    x = np.array(list(product(*x)))

    y = np.array([f_(x[i]) for i in range(N**len(mean))])

    domain = np.prod([(cov[i,i]*10)/(N-1) for i in range(len(mean))])

    return np.sum(y*domain)

f_norm = lambda beta, s_ : multivariate_normal.pdf(s_, mean = mu, cov = Sigma)

# print("Test to see if the integral of f_norm is 1:")
# print("Monte Carlo integration:", monte_carlo_integration(f_norm, 1, mu, Sigma, N = 100))

# print("Integration estimate of f1:", monte_carlo_integration(f1, 1, mu, Sigma, N = 100))


def f2(beta):
    A_inv = np.linalg.inv(A)
    S = (beta/2)*A_inv
    m = (-1/2)*b@A_inv
    S_inv = np.linalg.inv(S)
    S_det = np.linalg.det(S)
    k = (np.exp(c/-beta)/np.exp(-(1/2)*m.T@S_inv@m))*np.sqrt(S_det*(2*np.pi)**len(m))

    return k*multivariate_normal.pdf(mu, mean = m, cov = S+Sigma)

print("Test to see if the integral of f2: ", f2(1))

from matplotlib import pyplot as plt

X = np.linspace(0.01, 100, 1000)
plt.plot(X, [-x*np.log(f2(x))-x*0.1 for x in X])