import numpy as np
import matplotlib.pyplot as plt
import math
def f(x, c, K, delta):

    return - x*np.log((1/2**K)*np.sum(np.exp(-c/x))) - x*delta

def f_stable(x, c, K, delta):

    c_star = np.max(c)

    tmp1 = -x*np.log(np.exp(c_star)/2**K)
    tmp2 = -x*np.logaddexp.reduce(-c/x - c_star)
    tmp3 = -x*delta

    return tmp1 + tmp2 + tmp3

def f_prime(x, c, K, delta):

    c_star = np.max(c)

    tmp1 = -np.log(np.exp(c_star)/2**K)
    tmp2 = -np.logaddexp.reduce(-c/x - c_star)
    tmp3 = -np.sum(c*np.exp(-c/x - c_star)) / (x * np.sum(np.exp(-c/x - c_star)))
    tmp4 = -delta

    return tmp1 + tmp2 + tmp3 + tmp4

def f_prime_approx(x, c, K, delta, tol):

    return (f_stable(x+tol, c, K, delta) - f_stable(x-tol, c, K, delta)) / (2*tol)

X = np.linspace(0.01, 10, 1000)
c = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
K = int(math.floor(np.log2(len(c))))
delta = 0.1
tol = 1e-3


plt.plot(X, [f(x, c, K, delta) for x in X], label='f')
plt.plot(X, [f_stable(x, c, K, delta) for x in X], label='f_stable')
# plt.show()

plt.plot(X, [f_prime(x, c, K, delta) for x in X], label='f_prime')
plt.plot(X, [f_prime_approx(x, c, K, delta, tol) for x in X], label='f_prime_approx')
plt.legend()
plt.show()