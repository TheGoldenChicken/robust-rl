import numpy as np
from scipy.stats import multivariate_normal

def quadratic_approximation(X,y,beta):
    ### Quadratic approximation using Bayesian linear regression ###
    # Hyperparameters: alpha = 0, beta = 1
    D = X.shape[1]

    # Compute the design matrix for the quadratic function for n-dimensions
    Phi = np.array([np.ones(X.shape[0])/(-beta), *[X[:,i]/(-beta) for i in range(D)], *[(X[:,i]*X[:,j])/(beta) for i in range(D) for j in range(i,D)]]).T

    # Compute the posteriror using linear algebra
    Sigma = np.linalg.inv(Phi.T@Phi)
    mu = Sigma@Phi.T@y

    # Extract c, b, A from the fitted mean values.
    c = mu[0]
    b = mu[1:D+1]
    A = np.zeros((D,D))
    for i in range(D):
        A[i,i] = mu[D+1+i*2]
        for j in range(i+1,D):
            A[i,j] = mu[D+1+i+j]
            A[j,i] = A[i,j]

    return A, b, c

def linear_approximation(X,y,beta):
    ### Linear approximation using Bayesian linear regression ###
    # Hyperparameters: alpha = 0, beta = 1
    D = X.shape[1]

    # Compute the design matrix for the linear function for n-dimensions
    Phi = np.array([np.ones(X.shape[0])/(-beta), *[X[:,i]/(-beta) for i in range(D)]]).T

    # Compute the posteriror using linear algebra
    Sigma = np.linalg.inv(Phi.T@Phi)
    mu = Sigma@Phi.T@y

    c = mu[0]
    b = mu[1:]

    return b, c

def expectation(A, b, c, mu, Sigma):
    A_inv = np.linalg.inv(A)
    S = (-1/2)*A_inv
    m = (-b/2)@A_inv
    S_inv = np.linalg.inv(S)
    S_det = np.linalg.det(S)
    k = (np.exp(c)*np.sqrt(S_det*(2*np.pi)**len(m))/np.exp(-(1/2)*m.T@S_inv@m))

    return k*multivariate_normal.pdf(mu, mean = m, cov = S+Sigma)

def maximize(f_prime, tol = 1e-3):
    """
    Maximize f_stable with respect to x by using the derivative f_prime.
    Also note that f_prime is either monotonic decreasing or only has one maximum.
    Also note that x is always positive.
    """
    
    x_min = tol*2
    x_max = 10
    
    # If f_prime(tol) < 0 the function is monotonic decreasing.
    if f_prime(x_min) < 0:
        return x_min

    while True:
        # If f_prime(10) > 0, adjust the maximum
        if f_prime(x_max) > 0:
            x_max *= 2
        else: 
            break
    
    # Find the maximum using devide and conquer
    while True:
        x_mid = (x_min + x_max) / 2
        if x_max - x_min < tol:
            return x_mid
        f_prime_x_mid = f_prime(x_mid)
        if f_prime_x_mid > 0 + tol:
            x_min = x_mid
        elif f_prime_x_mid < 0 - tol:
            x_max = x_mid
        else:
            return x_mid

def pre_sub_robust_estimator(X_p,y_p,X_v,y_v,beta, delta = 0.1):
    ### Quadratic approximation ###
    A, b, c = quadratic_approximation(X_v, y_v,beta)

    ### Check if A is positive definite ###
    # We know it is semi- since A is always symmetric
    w, _ = np.linalg.eig(A)
    if np.any(w < 0):
        # If any eigenvalue is negative, use linear approximation instead
        b, c = linear_approximation(X_v, y_v,beta)
        A = np.identity(len(b))

    ### Gaussian approximation ###
    # Calculate mean and covariance of the X_p and y_p
    mu = np.mean(X_p, axis = 0)
    Sigma = np.cov(X_p.T)

    ### Pre supremum term ###
    pre_sup = -beta*np.log(expectation(A, b, c, mu, Sigma))-delta*beta

    return pre_sup

def pre_sub_robust_estimator_prime_approx(X_p,y_p,X_v,y_v, delta,tol = 1e-3):
    return lambda beta : (pre_sub_robust_estimator(X_p,y_p,X_v,y_v,beta+tol,delta)- \
                          pre_sub_robust_estimator(X_p,y_p,X_v,y_v,beta-tol,delta))/ \
                          (2*tol)

def robust_estimator(X_p,y_p,X_v,y_v,delta):
    """
    X_p: 2D array of samples (s,a) corresponding to transistions.
    y_p: Transision probability p(s'|s,a)
    X_v: 2D array of samples (s,a) corresponding to state values in y_v
    y_v: State values V(s) =max_a Q(s,a)
    delta: Kullback liebler divergence distance
    """

    f = pre_sub_robust_estimator(X_p,y_p,X_v,y_v,delta)
    f_prime = pre_sub_robust_estimator_prime_approx(X_p,y_p,X_v,y_v,delta)

    beta_max = maximize(f_prime)
    return f(beta_max)

# mu = np.array([0, 0])
# Sigma = np.array([[3, 1], [1, 4]])

# A = np.array([[1, 0], [0, 1]])
# b = np.array([0, 0])
# c = 0

from scipy.stats import norm
import matplotlib.pyplot as plt

# # 1D dataset
# X_p = np.expand_dims(np.linspace(-1.5, 1.5, 100), axis=1)
# y_p = np.array([norm.pdf(x, loc = 0, scale = 1) + np.random.normal(0, 0.1, 1) for x in X_p])
# X_v = np.expand_dims(np.linspace(-1.5, 1.5, 100), axis=1)
# y_v = np.array([norm.pdf(x, loc = 0, scale = 1) + np.random.normal(0, 0.1, 1) for x in X_v])


# delta = 0.1
# plt.scatter(X_p.squeeze(), y_p.squeeze())
# print(robust_estimator(X_p,y_p,X_v,y_v,delta))

# # 2D dataset
# Sample randomly in 2D space
X_p = np.array([[np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)] for _ in range(100)])
# Sample randomly for 2D multivariate gaussian
y_p = multivariate_normal.pdf(X_p, mean = [0, 0], cov = [[3, 1], [1, 4]])
X_v = X_p
y_v = y_p

delta = 0.1
# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_p[:,0], X_p[:,1], y_p)

beta = 0.1
# 2D plot of quadratic approximation
A, b, c = quadratic_approximation(X_p, y_p,beta)
X = np.linspace(-1.5, 1.5, 10)
Y = np.linspace(-1.5, 1.5, 10)

A_inv = np.linalg.inv(A)
S = (-1/2)*A_inv
m = (-b/2)@A_inv
S_inv = np.linalg.inv(S)
S_det = np.linalg.det(S)
k = (np.exp(c)*np.sqrt(S_det*(2*np.pi)**len(m))/np.exp(-(1/2)*m.T@S_inv@m))


Z = np.array([[A[0,0]*x**2+2*A[0,1]*x*y+A[1,1]*y**2+b[0]*x+b[1]*y+c for x in X] for y in Y])
Z_ = np.array([[k*multivariate_normal.pdf([x,y], mean = m, cov = S) for x in X] for y in Y])
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, Z, alpha=0.2, color = 'blue')
ax.plot_surface(X, Y, Z_, alpha=0.2, color = 'red')
plt.show()
print(robust_estimator(X_p,y_p,X_v,y_v,delta))
pass




# Plot the quadratic approximation

A, b, c = quadratic_approximation(X_v, y_v)
X = np.linspace(-1.5, 1.5, 100)
# if A < 0: A = np.array(0)
y = np.array([A*x**2+b*x+c for x in X])
plt.plot(X.squeeze(), y.squeeze())

# Plot a gaussian function with loc = 0 and scale = 1
X = np.linspace(-1.5, 1.5, 100)
plt.scatter(X.squeeze(), y.squeeze())
plt.plot(X.squeeze(), y_v.squeeze())
plt.show()