import numpy as np
import matplotlib.pyplot as plt

goal_x = 1125
goal_y = 333.3

xs, ys = np.linspace(0,2000, 1125), np.linspace(0,2000, 334)

reward_func = lambda x,y: sum([-x**2+2*goal_x*x, -y**2+2*goal_y*y])
reward_normalizer = lambda val: (val-reward_func(0,0))/(reward_func(1000, 1000)-reward_func(0,0))

rewards = [reward_normalizer(reward_func(x,y)) for x, y in zip(xs, ys)]
#
# plt.plot(rewards)
# plt.show()
X, Y = np.meshgrid(xs, ys)
Z = reward_normalizer(reward_func(X,Y))
max_idx = np.unravel_index(np.argmax(Z), Z.shape)
min_idx = np.unravel_index(np.argmin(Z), Z.shape)

min_x, min_y, min_z = xs[min_idx[1]], ys[min_idx[0]], np.min(Z)
max_x, max_y, max_z = xs[max_idx[1]], ys[max_idx[0]], np.max(Z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.scatter(max_x, max_y, max_z, color='red', marker='o', s=500)

print(max_x, max_y)
print(reward_normalizer(reward_func(1000,1000)))
print(min_x, min_y)
print(reward_normalizer(reward_func(0,1300)))

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Reward')
ax.set_title('Reward values as a function of X and Y')

# Show the plot
plt.show()