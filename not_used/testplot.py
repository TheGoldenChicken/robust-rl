import matplotlib.pyplot as plt
import numpy as np

# Generate random points
np.random.seed(0)
random_points = np.random.rand(50, 2) * 100

# Generate the extra central point
central_point = np.array([[10, 90]])

# Create scatter plot
plt.scatter(random_points[:, 0], random_points[:, 1], color='blue', label='Replay Buffer states')
plt.scatter(central_point[:, 0], central_point[:, 1], color='red', label='Central Point')

# Set plot properties
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Replay Buffer')
plt.legend()

# Display the plot
plt.show()

# Generate grid
x_grid = np.arange(0, 101, 33.3)
y_grid = np.arange(0, 101, 33.3)

# Create scatter plot with grid lines
# plt.scatter(random_points[:, 0], random_points[:, 1], color='blue', label='Replay Buffer states')
# plt.scatter(central_point[:, 0], central_point[:, 1], color='red', label='Central Point')

# Plot vertical grid lines
for x in x_grid:
    plt.axvline(x, color='grey', linestyle='--')


# Plot horizontal grid lines
for y in y_grid:
    plt.axhline(y, color='grey', linestyle='--')

dims = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
i = 0
for x in x_grid[:-1]:
    for y in y_grid[:-1]:
        plt.text(x=-5+x+(33.3)/2, y=y+(33.3)/2, s=dims[i])
        i += 1

# Set plot properties
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('')
plt.legend()

# Display the plot
plt.grid(False)
plt.show()

x_grid = np.arange(0, 101, 11.1)
y_grid = np.arange(0, 101, 33.3)

# Create scatter plot with grid lines
# plt.scatter(random_points[:, 0], random_points[:, 1], color='blue', label='Replay Buffer states')
# plt.scatter(central_point[:, 0], central_point[:, 1], color='red', label='Central Point')

# Plot vertical grid lines
for x in x_grid:
    plt.axvline(x, color='grey', linestyle='--')


# Plot horizontal grid lines
# for y in y_grid:
#     plt.axhline(y, color='grey', linestyle='--')

dims = np.arange(10)
i = 0
for x in x_grid[:]:
    plt.text(x=-5+x+(11.1)/2, y=10, s=dims[i])
    i += 1

# Set plot properties
plt.xlim(0, 100)
plt.ylim(0, 20)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('')
plt.legend()
plt.axis('off')
# Display the plot
plt.grid(False)
plt.show()