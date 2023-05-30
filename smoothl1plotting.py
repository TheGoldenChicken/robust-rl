import numpy as np
import matplotlib.pyplot as plt

diffs = np.linspace(-3, 3, 1000)

def smoothl1(diff, beta):
    if abs(diff) < beta:
        return (0.5 * (diff) ** 2) / beta
    else:
        return (abs(diff)) - 0.5 * beta

def l1(diff):
    return abs(diff)

def l2(diff):
    return 0.5*(diff ** 2)

lossesl1 = [l1(i) for i in diffs]
lossesl2 = [l2(i) for i in diffs]
lossessmooth = [smoothl1(i, 1.0) for i in diffs]

plt.plot(diffs, lossesl1, label='L1 Loss')
plt.plot(diffs, lossesl2, label='L2 Loss')
plt.plot(diffs, lossessmooth, label='Smooth L1 Loss')
plt.title('Smooth-L1 Loss')
plt.xlabel('(x - y)')
plt.ylabel('Loss')
plt.ylim(0, 3)  # Set the y-axis limit to show values up to 4
plt.xlim(-3, 3)
plt.legend()
plt.show()