import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


c_p = 2
c_v = 1
b_p = np.array([3])
b_v = np.array([5])
A_p = np.array([[-3]])
A_v = np.array([[7]])

minus_beta = -1

bv_beta = lambda beta : b_v/-beta

Ap_inv = lambda : np.linalg.inv(A_p)
Av_inv = lambda beta : np.linalg.inv(A_v/-beta)

det_Ap = lambda : np.linalg.det(-(1/2)*Ap_inv())
det_Av = lambda beta : np.linalg.det(-(1/2)*Av_inv(beta))

Sigma_p = lambda : -(1/2)*Ap_inv()
Sigma_v = lambda beta : -(1/2)*Av_inv(beta)
mu_p = lambda : b_p.T@Sigma_p()
mu_v = lambda beta : bv_beta(beta).T@Sigma_v(beta)
k_p = lambda : c_p*np.sqrt((2*np.pi)**(A_p.shape[0])*np.linalg.det(Sigma_p()))/np.exp((1/4)*mu_p().T@Sigma_p()@mu_p())
k_v = lambda beta : c_v*np.sqrt((2*np.pi)**(A_v.shape[0])*np.linalg.det(Sigma_v(beta)))/np.exp((1/4)*mu_v(beta).T@Sigma_v(beta)@mu_v(beta))

normal = lambda x, mu, Sigma : 1/np.sqrt((2*np.pi)**(x.shape[0])*np.linalg.det(Sigma))*np.exp(-(1/2)*((x-mu).T@np.linalg.inv(Sigma)@(x-mu)))

area = lambda beta : k_p()*k_v(beta)*normal(mu_p(), mu_v(beta), Sigma_p()+Sigma_v(beta))

b = 1
step = 0.01
X = np.arange(-10,10,step)
# Gaussian function using Sigma_p and mu_p
fy_p = lambda x : k_p()/np.sqrt((2*np.pi)**(A_p.shape[0])*np.linalg.det(Sigma_p()))* \
            np.exp(-(1/2)*(x-mu_p()).T@np.linalg.inv(Sigma_p())@(x-mu_p()))
y_p = np.array([[fy_p(x)] for x in X])

fy_v = lambda x, beta : k_v(beta)/np.sqrt((2*np.pi)**(A_v.shape[0])*np.linalg.det(Sigma_v(beta)))* \
            np.exp(-(1/2)*(x-mu_v(beta)).T@np.linalg.inv(Sigma_v(beta))@(x-mu_v(beta)))
y_v = np.array([[fy_v(x, b)] for x in X])

y_res = np.array([[fy_p(x)*fy_v(x, b)] for x in X])

plt.plot(X, y_p)
plt.plot(X, y_v)
plt.plot(X, y_res)
plt.show()

print("area: ", area(b))
print("area res: ", sum(y_res[:,0])*step)
plt.show()

x = np.linspace(0.1, 20, 100)
plt.plot(x, [-beta*np.log(area(beta))-beta*0.1 for beta in x])
