#%%
import numpy as np
from scipy.stats import multivariate_normal as normal
from scipy.optimize import minimize, Bounds

def quadratic_to_gaus(A, b, c):
    """
    Approximate a gaussian function with a quadratic function
    x.T@A@x + x.T@b + c ~ k*N(mu,Sigma)
    
    Returns: mu, Sigma, k, lambda(x)=k*scipy.stats.multivariate_normal.pdf(x,mu,Sigma)

    :param np.array A: Coefficient in quadratic function
    :param np.array b: Coefficient in quadratic function
    :param int c: Coefficient in quadratic funciton
    """
    if(A.shape[0] != 1):
        A_inv = np.linalg.inv(A)
        neg_half_A_det = np.linalg.det(-(1/2)*A_inv)
    else:
        A_inv = np.array([1/A,])
        neg_half_A_det = -(1/2)*A_inv
    
    Sigma = -(1/2)*A_inv
    mu = -(1/2)*b.T@Sigma
    k = c*np.sqrt((2*np.pi)**(mu.shape[0])*neg_half_A_det)/np.exp((1/4)*b.T@A_inv@b)
    
    return mu, Sigma, k, lambda x : k*normal.pdf(x, mu, Sigma)

def quadratic_approximation(X, y, beta = 1, alpha = 0):
    """
    Approximate a quadratic function to a dataset using bayesian inference.
    This function does not optimize the hyperparameters beta and alpha.
    Returns the parameter of a quadratic function.

    Returns: A, b, c
    
    :param np.array X: input data
    :param np.array y: output data
    :param beta: hyperparameter (is not optimized)
    :param alpha: hyperparameter (is not optimized)
    """

    d = X.shape[1]

    # Compute the design matrix for the quadratic function for n-dimensions
    Phi = np.array([np.ones(X.shape[0]), *[X[:,i] for i in range(d)], *[X[:,i]*X[:,j] for i in range(d) for j in range(i,d)]]).T

    # Compute the posteriror using linear algebra
    Sigma = np.linalg.inv(beta*Phi.T@Phi + alpha*np.eye(Phi.shape[1]))
    mu = beta*Sigma@Phi.T@y

    # Extract c, b, A from the fitted mean values.
    c = mu[0]
    b = mu[1:d+1]
    A = np.zeros((d,d))
    for i in range(d):
        A[i,i] = mu[d+1+i]
        for j in range(i+1,d):
            A[i,j] = mu[d+1+i+j]
            A[j,i] = A[i,j]
    return A, b, c

def int_gaus_exp_quadratic_product(mu, Sigma, k, A, b, c):
    """
    Cumpute the integral of the product between a gaussian function and a exponential quadratic function
        int N(mu,Sigma)*exp(x^TAx + b^Tx + c) dx

    :param np.array mean: Mean of gaussian function
    :param np.array Sigma: Covariance of gaussian function
    :param np.array k: Constant of gaussian function
    :param np.array A: Coefficient of quadratic function
    :param np.array b: Coefficient of quadratic function
    :param np.array c: Coefficient of quadratic function
    """
    
    # Convert the sqaured function to a gaussian function
    mu_, Sigma_, k_, _ = quadratic_to_gaus(A, b, c)

    # The product of two gaussian functions is a gaussian function
    #   k_1*N(x | mu_1, Sigma_1)*k_2*N(x | mu_2, Sigma_2) = k_1*k_2*k_3*N(x | mu_3, Sigma_3)
    #   where: k_3 = normal(mu_1 | mu_2, Sigma_1 + Sigma_2)

    # Compute the integral of the product of two gaussian functions 
    # by moving the constants outside the integral: int k*N(x | mu, Sigma) dx = k*int N(x | mu, Sigma) dx 
    integral = k*k_*normal.pdf(mu, mu_, Sigma+Sigma_)

    return integral

def int_two_exp_quadratic_product(A1, b1, c1, A2, b2, c2):
    """
    Compute the integral of product between two exponential quadratic function.
        int exp(x^TA1x + b1^Tx + c1)*exp(x^TA2x + b2^Tx + c2) dx

    Returns: result of integral (int)
    
    :param np.array A: Coefficient of the first quadratic function
    :param np.array b: Coefficient of the first quadratic function
    :param np.array c: Coefficient of the first quadratic function
    :param np.array A: Coefficient of the second quadratic function
    :param np.array b: Coefficient of the second quadratic function
    :param np.array c: Coefficient of the second quadratic function
    """
    # Convert the two squared sqaured function to a gaussian function
    mu1, Sigma1, k1, _ = quadratic_to_gaus(A1, b1, c1)
    mu2, Sigma2, k2, _ = quadratic_to_gaus(A2, b2, c2)

    # Compute the integral of the product of two gaussian
    integral = k1*k2*normal.pdf(mu1, mu2, Sigma1+Sigma2)

    return integral

def pre_supp_robust_estimator(mu, Sigma, X, y, beta, delta = 1):
    """
    Computes values inside the suppremum for the robust estimator.

    :param: np.array mu: Mean of the gaussian function
    :param: np.array Sigma: Covariance of the gaussian function
    :param np.array X: Explanatory variable for the state value function (p)
    :param np.array y: Response variable for the state value function (p)
    :param float beta: Hyperparameter
    :param float delta: Hyperparameter
    """

    # NOTE: Minus is missing (-y/beta) but the matrix becomes non-semi-definite.
    # NOTE: The overall difference may be that we use a infimum instead of suppremum.
    # NOTE: The shape of the resulting function -beta*np.log(area)-beta*delta is the same as in the paper. Negated
    A, b, c = quadratic_approximation(X, y/beta)
    area = int_gaus_exp_quadratic_product(mu, Sigma, 1, A, b, c)
    return -beta*np.log(area)-beta*delta

def robust_estimator(X_p, y_p, X_v, y_v, r, delta, gamma):
    """
    Estimate the robust_estimator using bayes inference to estimate, gaussian- and quadratic approximations.
    
    :param np.array X_p: Explanatory variable for the transition function, prev_obs and action: (s,a)
    :param np.array y_p: Response variable for the transition function, next_obs: s'
    :param np.array X_v: Explanatory variable for the state value function, next_obs: s'
    :param np.array y_v: Response variable for the state value function, value_function v(s')
    :param float delta: Max Kullback-Leibler divergence distance
    :param float gamma: Discount factor
    """
    suppremum = None
    mu = np.mean(y_p, axis=0)
    Sigma = np.cov(y_p.T)

    # Compute the suppremum for beta >= 0 of pre_supp_robust_estimator
    f = lambda beta: pre_supp_robust_estimator(mu, Sigma, X_v, y_v, beta, delta)
    bounds = Bounds(1e-03, np.inf)

    # NOTE: Minimize and not maximize is used. This results in a infimum.
    # NOTE: CHECK IF THIS IS CORRECT by using real data.
    suppremum = minimize(fun = f, x0 = 1, method='nelder-mead', options={'xatol': 1e-3}, bounds=bounds)

    return r + gamma*suppremum.fun


"""
If run as main script then toy examples are run.
This includes a 1D and 2D toy dataset.
The result of the toy dataset is also plotted.
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Seed for reproducibility
    np.random.seed(0)

    n = 100
    x_min = -1
    x_max = 1

    ### 1D toy dataset
    # Multivariate gaussain function plus gaussian noise
    X_1d = np.random.uniform(x_min,x_max,n)
    X_1d = np.expand_dims(X_1d, axis=1)
    y_1d = np.array([np.exp(-x**2) + np.random.normal(0,0.1) for x in X_1d])

    ### 2D toy dataset
    # Multivariate gaussain function plus gaussian noise
    X_2d = np.array([np.random.uniform(x_min,x_max,n), np.random.uniform(x_min,x_max,n)]).T
    y_2d = np.array([np.exp(-x[0]**2 - x[1]**2) + np.random.normal(0,0.1) for x in X_2d])

    A_1d, b_1d, c_1d = quadratic_approximation(X_1d, y_1d)

    mu_1d, Sigma_1d, _, f_gaus_1d = quadratic_to_gaus(A_1d, b_1d, c_1d)

    A_2d, b_2d, c_2d = quadratic_approximation(X_2d, y_2d)

    mu_2d, Sigma_2d, _, f_gaus_2d = quadratic_to_gaus(A_2d, b_2d, c_2d)

    # Visualize dataset, true_function, quadratic approximation, gaussian approximation
    X = np.expand_dims(np.linspace(x_min-1,x_max+1,n), axis = 1)
    plt.scatter(X_1d, y_1d) # Dataset
    plt.plot(X, np.array([np.exp(-x**2) for x in X])) # True function
    plt.plot(X, np.array([x.T@A_1d@x + b_1d@x + c_1d for x in X])) # Quadratic approximation
    plt.plot(X, np.array([f_gaus_1d(x).squeeze() for x in X])) # Gaussian approximation
    plt.legend(["Dataset",
                "True function",
                "Quadratic approximation",
                "Gaussian approximation",
                "Exponential quadratic approximation"])
    plt.show()

    # 2D visualization
    X = np.array([[x,y] for x in np.linspace(x_min,x_max,n) for y in np.linspace(x_min,x_max,n)])
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_2d[:,0], X_2d[:,1], y_2d) # Dataset
    ax.plot_surface(X[:,0].reshape(n,n), X[:,1].reshape(n,n),
                    np.array([np.exp(-x[0]**2 - x[1]**2) for x in X]).reshape(n,n),
                    alpha = 0.5) # True function
    ax.plot_surface(X[:,0].reshape(n,n), X[:,1].reshape(n,n),
                    np.array([x.T@A_2d@x + b_2d@x + c_2d for x in X]).reshape(n,n),
                    alpha = 0.5) # Quadratic approximation
    ax.plot_surface(X[:,0].reshape(n,n), X[:,1].reshape(n,n),
                    np.array([f_gaus_2d(x).squeeze() for x in X]).reshape(n,n),
                    alpha = 0.5) # Gaussian approximation
    plt.show()


    print(robust_estimator(X_1d, y_1d, X_1d, y_1d, 0, 0.1, 0.9))

