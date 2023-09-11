
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches

import sys
import os

# Add current folder to path
sys.path.append('..')
# Set the current working directory to the folder this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def generate_heatmap(x, y, std = 8):
    layout_size = [11,7]
    y = layout_size[1] - y
    # x = layout_size[1] - x

    bins = int(np.prod(layout_size)**(1.5))
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=std)
    heatmap = np.log(heatmap+0.1)+1 # Bringing it into a nicer scale

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet)

    plt.show()
    return heatmap.T, extent


delta = 3
seed = 1000
factor = -1
linear = True

q_vals =  np.load(f'test_results/linear={linear}-seed={seed}-factor={factor}/{delta}-q_vals.npy', allow_pickle=True).reshape(150,150,5)

# Transpose the first two dimentions
q_vals = np.transpose(q_vals, axes=(1,0,2))

# Flip the second dimention
q_vals = np.flip(q_vals, axis=0)

# Make a image where each pixel represents which of the 5 actions is best
# action 1 (noopt): red
# action 2 (down): green
# action 3 (left): blue
# action 4 (right): yellow
# action 5 (up): purple

noopt = 0
down = 1
left = 2
right = 3
up = 4
best_actions = np.argmax(q_vals, axis=2)

# Make a color map
colors = np.zeros((150,150,3))
colors[best_actions == noopt] = [255,0,0]
colors[best_actions == down] = [0,255,0]
colors[best_actions == left] = [0,0,255]
colors[best_actions == right] = [255,255,0]
colors[best_actions == up] = [255,0,255]



# Plot the color map
plt.imshow(colors.astype(np.uint8))
# plt.show()

goal_plot = plt.scatter([112.5],[150-33.3],label = 'Goal')
start_plot = plt.scatter([33.3],[150-33.3],label = 'Start')

# Dotted Horizontal lin through image at y = 150 - 30
plt.plot([0,149],[120,120],'--',color='black')

opopt_patch = mpatches.Patch(color='red', label='OpOpt')
down_patch = mpatches.Patch(color='green', label='Up')
left_patch = mpatches.Patch(color='blue', label='Left')
right_patch = mpatches.Patch(color='yellow', label='Right')
up_patch = mpatches.Patch(color='purple', label='Down')

plt.legend(handles=[opopt_patch, down_patch, left_patch, right_patch, up_patch, start_plot, goal_plot], loc='upper left')

# Change the x and y ticks to be between 0 and 1
plt.xticks(np.around(np.linspace(0,149,6),decimals=2), np.around(np.linspace(0,1,6), decimals=2))
plt.yticks(np.around(np.linspace(0,149,6),decimals=2), np.around(np.linspace(0,1,6), decimals=2))

# x and y labels
plt.xlabel('x')
plt.ylabel('y')

# Change the title
plt.title(f'Cliff Car Q-Values, delta: {delta}')

plt.show()

print("done")





