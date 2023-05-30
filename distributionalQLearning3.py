import numpy as np
from scipy.stats import multivariate_normal
import sys
from scipy.optimize import minimize

def quadratic_approximation(X,y):
    ### Quadratic approximation using Bayesian linear regression ###
    # Hyperparameters: alpha = 0, beta = 1
    D = X.shape[1]

    # Compute the design matrix for the quadratic function for n-dimensions
    Phi = np.array([np.ones(X.shape[0]), *[X[:,i] for i in range(D)], *[X[:,i]*X[:,j] for i in range(D) for j in range(i,D)]]).T

    # Compute the posteriror using linear algebra
    Sigma = np.linalg.pinv(Phi.T@Phi)
    mu = Sigma@Phi.T@y # TODO: THIS HAS A TENDENCY TO RETURN NAN VALUES

    # Extract c, b, A from the fitted mean values.
    c = mu[0]
    b = mu[1:D+1]
    A = np.zeros((D,D))
    for i in range(D):
        A[i,i] = mu[D+1+i*2]
        for j in range(i+1,D):
            A[i,j] = mu[D+1+i+j]
            A[j,i] = A[i,j]

    # if np.isnan(A):
    #     i = 2

    return A, b, c

def linear_approximation(X,y):
    ### Linear approximation using Bayesian linear regression ###
    # Hyperparameters: alpha = 0, beta = 1
    D = X.shape[1]

    # Compute the design matrix for the linear function for n-dimensions
    Phi = np.array([np.ones(X.shape[0]), *[X[:,i] for i in range(D)]]).T

    # Compute the posteriror using linear algebra
    Sigma = np.linalg.inv(Phi.T@Phi)
    mu = Sigma@Phi.T@y

    c = mu[0]
    b = mu[1:]

    return b, c

def expectation(A, b, c, beta, mu, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_tilde = np.linalg.inv(Sigma_inv+(2/beta)*A)
    Sigma_tilde_inv = np.linalg.inv(Sigma_tilde)
    mu_tilde = (mu.T@Sigma_inv+(-b.T/beta))@Sigma_tilde
    k = np.sqrt(np.linalg.det(Sigma_tilde)/np.linalg.det(Sigma)) \
        * np.exp((-1/2)*mu.T@Sigma_inv@mu+(1/2)*mu_tilde.T@Sigma_tilde_inv@mu_tilde+(c/-beta))
    
    if k == np.array([[np.inf]]):
        k = np.array([[0]])

    return k

def maximize_f(f):
    
    # Find the maximum of f
    # Use a negative sign for maximization
    res = minimize(lambda x : -f(x), 1, method = 'Nelder-Mead', tol = 1e-5, bounds = [(1e-5, np.inf)])
    
    # Return the: f(x), x
    return -res.fun, res.x[0]


# def maximize(f_prime, tol = 1e-5):
#     """
#     Maximize f_stable with respect to x by using the derivative f_prime.
#     Also note that f_prime is either monotonic decreasing or only has one maximum.
#     Also note that x is always positive.
#     """
#
#     x_min = 2e-3
#     x_max = 10
#
#
#     # If f_prime(tol) < 0 the function is monotonic decreasing.
#     if f_prime(x_min) < 0:
#         return x_min
#
#     while True:
#         # If f_prime(10) > 0, adjust the maximum
#         if f_prime(x_max) > 0:
#             x_max *= 2
#         else:
#             break
#
#     # Find the maximum using devide and conquer
#     while True:
#         x_mid = (x_min + x_max) / 2
#         if x_max - x_min < tol:
#             return x_mid
#         f_prime_x_mid = f_prime(x_mid)
#         if f_prime_x_mid > 0 + tol:
#             x_min = x_mid
#         elif f_prime_x_mid < 0 - tol:
#             x_max = x_mid
#         else:
#             return x_mid

def pre_sub_robust_estimator(X_p,y_p,X_v,y_v, delta = 0.1, linear_only = False):
    
    if not linear_only:
        ### Quadratic approximation ###
        A, b, c = quadratic_approximation(X_v, y_v)
        ### Check if A is positive definite ###
        # We know it is semi- since A is always symmetric
        w, _ = np.linalg.eig(A)
        if np.any(w < 0):
            # If any eigenvalue is negative, use linear approximation instead
            b, c = linear_approximation(X_v, y_v)
            A = np.zeros((len(b),len(b)))
    else:
        b, c = linear_approximation(X_v, y_v)
        A = np.zeros((len(b),len(b)))
        
    b, c = linear_approximation(X_v, y_v)
    # A = np.zeros((len(b),len(b)))

    ### Gaussian approximation ###
    # Compute the mean and covariance of the samples X_p, y_p
    mu = np.mean(X_p, axis = 0)
    Sigma = np.cov(X_p.T)

    # For 1D environments expand the dimensions for the covariance
    if Sigma.shape == (): Sigma = np.expand_dims(np.expand_dims(Sigma, axis = 0),axis=0)

    def estimator(beta):
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_tilde = np.linalg.inv(Sigma_inv+(2/beta)*A)
        Sigma_tilde_inv = np.linalg.inv(Sigma_tilde)
        mu_tilde = ((mu.T@Sigma_inv+(-b.T/beta))@Sigma_tilde).flatten() # We flatten here to avoid a mishap with Sigma_tilde
        k1 = np.sqrt(np.linalg.det(Sigma_tilde)/np.linalg.det(Sigma))
        k2 = (-1/2)*mu.T@Sigma_inv@mu+(1/2)*mu_tilde.T@Sigma_tilde_inv@mu_tilde+(c/-beta)


        return - beta * (np.log(k1) + k2) - delta * beta

    return estimator

def pre_sub_robust_estimator_prime_approx(X_p,y_p,X_v,y_v, delta,tol = 1e-3, linear_only = False):
    return lambda beta : (pre_sub_robust_estimator(X_p,y_p,X_v,y_v,delta,linear_only)(beta+tol)- \
                          pre_sub_robust_estimator(X_p,y_p,X_v,y_v,delta,linear_only)(beta-tol))/ \
                          (2*tol)

def robust_estimator(X_p,y_p,X_v,y_v,delta, linear_only = False):
    """
    X_p: 2D array of samples (s,a) corresponding to transistions.
    y_p: Transision probability p(s'|s,a)
    X_v: 2D array of samples (s,a) corresponding to state values in y_v
    y_v: State values V(s) =max_a Q(s,a)
    delta: Kullback liebler divergence distance
    """

    f = pre_sub_robust_estimator(X_p,y_p,X_v,y_v,delta, linear_only)
    # f_prime = pre_sub_robust_estimator_prime_approx(X_p,y_p,X_v,y_v,delta, linear_only=linear_only)
    
    return maximize_f(f)
    # beta_max = maximize(f_prime)
    # return f(beta_max)[0][0], beta_max

# mu = np.array([0, 0])
# Sigma = np.array([[3, 1], [1, 4]])

# A = np.array([[1, 0], [0, 1]])
# b = np.array([0, 0])
# c = 0

# from scipy.stats import norm
# import matplotlib.pyplot as plt

# # # 1D dataset
# # X_p = np.expand_dims(np.linspace(-1.5, 1.5, 100), axis=1)
# # y_p = np.array([norm.pdf(x, loc = 0, scale = 1) + np.random.normal(0, 0.1, 1) for x in X_p])
# # X_v = np.expand_dims(np.linspace(-1.5, 1.5, 100), axis=1)
# # y_v = np.array([norm.pdf(x, loc = 0, scale = 1) + np.random.normal(0, 0.1, 1) for x in X_v])


# # delta = 0.1
# # plt.scatter(X_p.squeeze(), y_p.squeeze())
# # print(robust_estimator(X_p,y_p,X_v,y_v,delta))

# # # 2D dataset
# # Sample randomly in 2D space
# X_p = np.array([[np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)] for _ in range(100)])
# # Sample randomly for 2D multivariate gaussian
# y_p = multivariate_normal.pdf(X_p, mean = [0, 0], cov = [[3, 1], [1, 4]])
# X_v = X_p
# y_v = y_p

# delta = 0.1
# # 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_p[:,0], X_p[:,1], y_p)

# # 2D plot of quadratic approximation
# A, b, c = quadratic_approximation(X_p, y_p)
# X = np.linspace(-1.5, 1.5, 10)
# Y = np.linspace(-1.5, 1.5, 10)

# beta = 0.1
# S = (beta/2)*np.linalg.inv(A)
# m = (-b/2)@np.linalg.inv(A)
# S_inv = np.linalg.inv(S)
# S_det = np.linalg.det(S)
# k = (np.exp(c/-beta)*np.sqrt((2*np.pi)**len(m)*S_det))/np.exp(-(1/2)*m.T@S_inv@m)


# Z = np.array([[A[0,0]*x**2+2*A[0,1]*x*y+A[1,1]*y**2+b[0]*x+b[1]*y+c for x in X] for y in Y])
# Z_ = np.array([[multivariate_normal.pdf([x,y], mean = m, cov = S) for x in X] for y in Y])
# X, Y = np.meshgrid(X, Y)
# ax.plot_surface(X, Y, Z, alpha=0.2, color = 'blue')
# ax.plot_surface(X, Y, Z_, alpha=0.2, color = 'red')
# plt.show()
# print(robust_estimator(X_p,y_p,X_v,y_v,delta))
# pass




# # Plot the quadratic approximation

# A, b, c = quadratic_approximation(X_v, y_v)
# X = np.linspace(-1.5, 1.5, 100)
# # if A < 0: A = np.array(0)
# y = np.array([A*x**2+b*x+c for x in X])
# plt.plot(X.squeeze(), y.squeeze())

# # Plot a gaussian function with loc = 0 and scale = 1
# X = np.linspace(-1.5, 1.5, 100)
# plt.scatter(X.squeeze(), y.squeeze())
# plt.plot(X.squeeze(), y_v.squeeze())
# plt.show()


