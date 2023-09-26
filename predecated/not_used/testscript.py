import numpy as np
import matplotlib.pyplot as plt


m = [1,3,7,9]
b = [1,1.5,2,2.5]
n = 10

demand_ratios = np.zeros((len(m),len(b),n+1))

for i in range(len(m)):
    for j in range(len(b)):
        for x in range(n+1):
            demand_ratios[i,j,x] = (b[j]+1)/(n+1) \
                                    if x == m[i] or x == m[i] + 1 \
                                    else (n - 1 - 2*b[j])/(n**2 - 1)

# 4x2 grid of subplots. Row is m = 3 and b = [1,1.5,2,2.5]
# Column is m = [3,4,5,6] and b = 1
fig, axs = plt.subplots(2, 4, figsize=(16,5))

# Add vertical and horizontal distance between the plots
fig.subplots_adjust(hspace = .5, wspace=.0)

for j in range(len(b)):
    axs[0,j].bar(np.arange(n+1), demand_ratios[0,j,:])
    axs[0,j].set_xlabel("Demand")
    axs[0,j].set_title(f"b = {b[j]}, m = 1")
    axs[0,j].set_ylim([0,0.35])
    # remove yticks
    if j != 0: axs[0,j].set_yticks([])
    else: axs[0,j].set_ylabel("Probability")

for i in range(len(m)):
    axs[1,i].bar(np.arange(n+1), demand_ratios[i,0,:])
    axs[1,i].set_xlabel("Demand")
    axs[1,i].set_title(f"b = 1, m = {m[i]}")
    
    # remove yticks
    if i != 0: axs[1,i].set_yticks([])
    else: axs[1,i].set_ylabel("Probability")
    
plt.show()




m = [0,1,2,3,4,5,5,6,7,8,9]
b = [1,2,3,3.5]
demand_ratios = np.zeros((len(m),len(b),n+1))

for i in range(len(m)):
    for j in range(len(b)):
        for x in range(n+1):
            demand_ratios[i,j,x] = (b[j]+1)/(n+1) \
                                    if x == m[i] or x == m[i] + 1 \
                                    else (n - 1 - 2*b[j])/(n**2 - 1)

uniform_p = np.ones(n+1)/(n+1)

# Calculate the KL-divergence between the uniform distribution and the demand distribution KL(unifrom || demand)
KL_divergence = np.zeros((len(m),len(b)))

for i in range(len(m)):
    for j in range(len(b)):
        KL_divergence[i,j] = np.sum(uniform_p * np.log(uniform_p/demand_ratios[i,j,:]))
        
print(KL_divergence)



# plt.bar(np.arange(n+1), demand_ratio)
# plt.xlabel("Demand")
# plt.ylabel("Probability")
# plt.title(f"Demand probability distribution, b = {b} and m = {m}")
# plt.show()