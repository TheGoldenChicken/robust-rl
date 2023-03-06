import numpy as np
import matplotlib.pyplot as plt

def normal(x, mu, Sigma):
    if Sigma.shape[0] != 1:
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
    else:
        Sigma_det = Sigma
        Sigma_inv = 1/Sigma
        
    return 1/(np.sqrt((2*np.pi)**(mu.shape[0])*Sigma_det))*np.exp(-(1/2)*(x-mu).T@Sigma_inv@(x-mu))

def gaus_to_expo_squared(mu, Sigma, k):
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)
    
    A = -Sigma_inv/2
    b = mu.T@Sigma_inv
    c = k/np.sqrt((2*np.pi)**(mu.shape[0])*Sigma_det) * np.exp((1/2)*mu.T@Sigma_inv@mu)
    
    return A, b, c, lambda x : np.exp(x*A@x + b@x + c)    
    
def expo_squared_to_gaus(A, b, c):
    if(A.shape[0] != 1):
        A_inv = np.linalg.inv(A)
        neg_half_A_det = np.linalg.det(-(1/2)*A_inv)
    else:
        A_inv = np.array([1/A,])
        neg_half_A_det = -(1/2)*A_inv
    
    Sigma = -(1/2)*A_inv
    mu = -(1/2)*b.T@A_inv
    k = np.exp(c)*np.sqrt((2*np.pi)**(mu.shape[0])*neg_half_A_det)/np.exp((1/4)*b.T@A_inv@b)
    
    return mu, Sigma, k, lambda x : k*normal(x, mu, Sigma)


beta = 1
step = 0.1

c_p = 2
c_v = -1/-beta
b_p = np.array([-3])
b_v = np.array([5])/-beta
A_p = np.array([[-3]])
A_v = np.array([[7]])/-beta

mu_p, Sigma_p, k_p, f_p = expo_squared_to_gaus(A_p, b_p, c_p)
mu_v, Sigma_v, k_v, f_v = expo_squared_to_gaus(A_v, b_v, c_v)

X = np.arange(-10,10,step)

plt.plot(X, np.array([f_p(np.array([x,])) for x in X]).squeeze())
plt.plot(X, np.array([f_v(np.array([x,])) for x in X]).squeeze())
plt.legend(["Normal: p", "Normal: v"])
plt.plot(X, np.array([np.exp(A_p*(x**2) + b_p*x + c_p) for x in X]).squeeze())
plt.plot(X, np.array([np.exp(A_v*(x**2) + b_v*x + c_v) for x in X]).squeeze())
plt.legend(["Squared: p", "Squared: v"])
plt.show()

# Computed area by analytically integrating the product of two gaussian functions
area_com = sum([f_p(x)*f_v(x) for x in X])*step

# Derived area using the product of two gaussian functions
area_der = k_p*k_v*normal(mu_p, mu_v, Sigma_p+Sigma_v)


print("Computed area: " + str(area_com.squeeze()))
print("Derived area: " + str(area_der.squeeze()))