import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


c_p = 2
c_v = 1
b_p = np.array([3])
b_v = np.array([1])
A_p = np.array([[-1]])
A_v = np.array([[-0.5]])

minus_beta = 2

bv_beta = b_v/minus_beta

Ap_inv = np.linalg.inv(A_p)
Av_inv = np.linalg.inv(A_v/minus_beta)

det_Ap = np.linalg.det(-(1/2)*Ap_inv)
det_Av = np.linalg.det(-(1/2)*Av_inv)

Sigma_p = -(1/2)*Ap_inv
Sigma_v = -(1/2)*Av_inv
mu_p = b_p.T@Sigma_p
mu_v = bv_beta.T@Sigma_v
k_p = c_p*np.sqrt((2*np.pi)**(A_p.shape[0])*np.linalg.det(Sigma_p))/np.exp((1/4)*mu_p.T@Sigma_p@mu_p)
k_v = c_v*np.sqrt((2*np.pi)**(A_v.shape[0])*np.linalg.det(Sigma_v))/np.exp((1/4)*mu_v.T@Sigma_v@mu_v)

normal = lambda x, mu, Sigma : 1/np.sqrt((2*np.pi)**(x.shape[0])*np.linalg.det(Sigma))*np.exp(-(1/2)*((x-mu).T@np.linalg.inv(Sigma)@(x-mu)))

area = k_p*k_v*normal(mu_p, mu_v, Sigma_p+Sigma_v)


step = 0.01
X = np.arange(-10,10,step)
# Gaussian function using Sigma_p and mu_p
fy_p = lambda x : k_p/np.sqrt((2*np.pi)**(A_p.shape[0])*np.linalg.det(Sigma_p))* \
            np.exp(-(1/2)*(x-mu_p).T@np.linalg.inv(Sigma_p)@(x-mu_p))
y_p = np.array([[fy_p(x)] for x in X])

fy_v = lambda x : k_v/np.sqrt((2*np.pi)**(A_v.shape[0])*np.linalg.det(Sigma_v))* \
            np.exp(-(1/2)*(x-mu_v).T@np.linalg.inv(Sigma_v)@(x-mu_v))
y_v = np.array([[fy_v(x)] for x in X])

y_res = np.array([[fy_p(x)*fy_v(x)] for x in X])

plt.plot(X, y_p)
plt.plot(X, y_v)
plt.plot(X, y_res)
plt.show()

print("area: ", area)
print("area res: ", sum(y_res[:,0])*step)
