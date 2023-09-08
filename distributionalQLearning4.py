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
        
    ### Gaussian approximation ###
    # Compute the mean and covariance of the samples X_p, y_p
    mu = np.mean(y_p, axis = 0)
    Sigma = np.cov(y_p.T)

    # For 1D environments expand the dimensions for the covariance
    if Sigma.shape == (): Sigma = np.expand_dims(np.expand_dims(Sigma, axis = 0),axis=0)

    def estimator(beta):
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_tilde = np.linalg.inv(Sigma_inv+(2/beta)*A)
        Sigma_tilde_inv = np.linalg.inv(Sigma_tilde)
        mu_tilde = (mu.T@Sigma_inv+(-b.T/beta))@Sigma_tilde
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
    
    return maximize_f(f)


