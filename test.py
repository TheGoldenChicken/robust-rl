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


mu = np.array([1, 2])
Sigma = np.array([[3, 1], [1, 4]])

A = np.array([[1, 0], [0, 1]])
b = np.array([1, -1])
c = 0

from itertools import product
# Numerical multivariate integration over f1 with respect to s_ using Monte Carlo
def monte_carlo_integration(f, mean, cov, N):

    x = []
    for i in range(len(mean)): # For each dimension
        x.append(np.linspace(start = mean[i] - cov[i,i]*10,
                             stop = mean[i] + cov[i,i]*10,
                             num = N))
    x = np.array(list(product(*x)))

    y = np.array([f(x[i]) for i in range(N**len(mean))])

    domain = np.prod([(cov[i,i]*20)/(N-1) for i in range(len(mean))])

    return np.sum(y*domain)

f_norm = lambda s_ : multivariate_normal.pdf(s_, mean = mu, cov = Sigma)

print("Test to see if the integral of f_norm is 1:")
print("Monte Carlo integration:", monte_carlo_integration(f_norm, mu, Sigma, N = 100))

delta = 0.1

def f1(beta):
    f = lambda s_ : multivariate_normal.pdf(s_, mean = mu, cov = Sigma)*np.exp((s_.T @ A @ s_ + b.T @ s_ + c) * (-1/beta))
    # tmp1 = multivariate_normal.pdf(s_, mean = mu, cov = Sigma)
    # tmp2 = np.exp((s_.T @ A @ s_ + b.T @ s_ + c) * (-1/beta))

    expectation = monte_carlo_integration(f, mu, Sigma, N = 100)
    return -beta*np.log(expectation)-beta*delta

def f2(beta):

    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_tilde = np.linalg.inv(Sigma_inv+(2/beta)*A)
    Sigma_tilde_inv = np.linalg.inv(Sigma_tilde)
    mu_tilde = (mu.T@Sigma_inv+(-b.T/beta))@Sigma_tilde
    k = np.sqrt(np.linalg.det(Sigma_tilde)/np.linalg.det(Sigma)) \
        * np.exp((-1/2)*mu.T@Sigma_inv@mu+(1/2)*mu_tilde.T@Sigma_tilde_inv@mu_tilde+(c/-beta))
    
    return -beta*np.log(k)-beta*delta

def f2_prime(beta):

    A_inv = np.linalg.inv(A)
    S = (beta/2)*A_inv
    m = (-1/2)*b@A_inv
    S_inv = np.linalg.inv(S)
    S_det = np.linalg.det(S)
    k = (np.exp(c/-beta)/np.exp(-(1/2)*m.T@S_inv@m))*np.sqrt(S_det*(2*np.pi)**len(m))

    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_hat = np.linalg.inv(Sigma_inv + S_inv)
    mu_hat = Sigma_hat@(Sigma_inv@mu+S_inv@m)
    N_m = multivariate_normal.pdf(m, mean = mu, cov = S+Sigma)
    tmp1 = -np.log(k*N_m)
    tmp2 = (-1/beta)*(np.trace(A@Sigma_hat)+mu_hat.T@A@mu_hat + b.T@mu_hat + c)
    tmp3 = -beta/N_m
    return tmp1 + tmp2 + tmp3 - delta

def f2_prime_approx(beta, tol):
    return (f2(beta+tol) - f2(beta-tol))/2*tol

print("f1(1):", f1(2))
print("f2(1):", f2(2))
print("f2_prime(1):", f2_prime(1))
print("f2_prime_approx(1):", f2_prime_approx(1, 1e-3))

X = np.linspace(0.01, 100, 1000)

# from matplotlib import pyplot as plt

# X = np.linspace(0.01, 100, 1000)
# plt.plot(X, [-x*np.log(f2(x))-x*0.1 for x in X])


#%%
import matplotlib.pyplot as plt

mu = np.array([1, 2])
Sigma = np.array([[3, 1], [1, 4]])

A = np.array([[1, 0], [0, 1]])
b = np.array([1, -1])
c = 3

beta = 1

f = lambda x : multivariate_normal.pdf(x, mean = mu, cov = Sigma)*np.exp((x.T @ A @ x + b.T @ x + c) * (-1/beta))


Sigma_inv = np.linalg.inv(Sigma)
Sigma_tilde = np.linalg.inv(Sigma_inv+(2/beta)*A)
Sigma_tilde_inv = np.linalg.inv(Sigma_tilde)
mu_tilde = (mu.T@Sigma_inv+(-b.T/beta))@Sigma_tilde
k = np.sqrt(np.linalg.det(Sigma_tilde)/np.linalg.det(Sigma)) \
    * np.exp((-1/2)*mu.T@Sigma_inv@mu+(1/2)*mu_tilde.T@Sigma_tilde_inv@mu_tilde+(c/-beta))

# return -beta*np.log(k*multivariate_normal.pdf(mu, mean = mu_tilde, cov = Sigma_tilde))-beta*delta


# 3D plot of f
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([f(np.array([X[i,j], Y[i,j]])) for i in range(len(x)) for j in range(len(y))]).reshape(len(x), len(y))

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('f')
plt.show()


#%%

# Kullback leibler test
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from itertools import product

mu1 = np.array([1, 2])
mu2 = np.array([2, 1])
cov1 = np.array([[3, 1], [1, 4]])
cov2 = np.array([[2, 0], [0, 2]])


def kl_divergence(mu, m, Sigma, S):
    det_Sigma = np.linalg.det(Sigma)
    det_S = np.linalg.det(S)
    S_inv = np.linalg.inv(S)

    term1 = np.log(det_S/det_Sigma)+(m-mu).T@S_inv@(m-mu)+np.trace(S_inv@Sigma)-len(mu)
    return (1/2)*term1

print("exact", kl_divergence(mu1, mu2, cov1, cov2))

# Interactivate plot with sliders for mu1, mu2, cov1, cov2
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def f(mu1_x, mu1_y, mu2_x, mu2_y, cov1_xx, cov1_xy, cov1_yx, cov1_yy, cov2_xx, cov2_xy, cov2_yx, cov2_yy):
    mu1 = np.array([mu1_x, mu1_y])
    mu2 = np.array([mu2_x, mu2_y])
    cov1 = np.array([[cov1_xx, cov1_xy], [cov1_yx, cov1_yy]])
    cov2 = np.array([[cov2_xx, cov2_xy], [cov2_yx, cov2_yy]])
    return kl_divergence(mu1, mu2, cov1, cov2)

interact(f, mu1_x = widgets.FloatSlider(min=-5, max=5, step=0.1, value=1),
            mu1_y = widgets.FloatSlider(min=-5, max=5, step=0.1, value=2),
            mu2_x = widgets.FloatSlider(min=-5, max=5, step=0.1, value=2),
            mu2_y = widgets.FloatSlider(min=-5, max=5, step=0.1, value=1),
            cov1_xx = widgets.FloatSlider(min=0.1, max=5, step=0.1, value=3),
            cov1_xy = widgets.FloatSlider(min=-5, max=5, step=0.1, value=1),
            cov1_yx = widgets.FloatSlider(min=-5, max=5, step=0.1, value=1),
            cov1_yy = widgets.FloatSlider(min=0.1, max=5, step=0.1, value=4),
            cov2_xx = widgets.FloatSlider(min=0.1, max=5, step=0.1, value=2),
            cov2_xy = widgets.FloatSlider(min=-5, max=5, step=0.1, value=0),
            cov2_yx = widgets.FloatSlider(min=-5, max=5, step=0.1, value=0),
            cov2_yy = widgets.FloatSlider(min=0.1, max=5, step=0.1, value=2))

#%%
# 2D plot of f
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([f(np.array([X[i,j], Y[i,j]])) for i in range(len(x)) for j in range(len(y))]).reshape(len(x), len(y))

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('f')
plt.show()

#%%

from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def signal(amp, freq):
    return amp * sin(2 * pi * freq * t)

axis_color = 'lightgoldenrodyellow'

fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

t = np.arange(0.0, 1.0, 0.001)
amp_0 = 5
freq_0 = 3

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax.plot(t, signal(amp_0, freq_0), linewidth=2, color='red')
ax.set_xlim([0, 1])
ax.set_ylim([-10, 10])

# Add two sliders for tweaking the parameters

# Define an axes area and draw a slider in it
amp_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
amp_slider = Slider(amp_slider_ax, 'Amp', 0.1, 10.0, valinit=amp_0)

# Draw another slider
freq_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
freq_slider = Slider(freq_slider_ax, 'Freq', 0.1, 30.0, valinit=freq_0)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    line.set_ydata(signal(amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()
amp_slider.on_changed(sliders_on_changed)
freq_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    freq_slider.reset()
    amp_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)

# Add a set of radio buttons for changing color
color_radios_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15], facecolor=axis_color)
color_radios = RadioButtons(color_radios_ax, ('red', 'blue', 'green'), active=0)
def color_radios_on_clicked(label):
    line.set_color(label)
    fig.canvas.draw_idle()
color_radios.on_clicked(color_radios_on_clicked)

plt.show()

#%%
