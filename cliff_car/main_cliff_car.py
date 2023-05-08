import cliff_car
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

env = cliff_car.Env(max_duration = 50, render = True)
agent = cliff_car.PlayMode(env)

episodes = 30
for _ in range(episodes):
    env.reset()
    while not agent.next():
        pass

def generate_heatmap(x, y, std = 8):
    layout_size = np.array(env.layout).shape
    y = layout_size[1] - y
    # x = layout_size[1] - x

    bins = int(np.prod(layout_size)**(1.5))
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=std)
    heatmap = np.log(heatmap+0.1)+1 # Bringing it into a nicer scale

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

data = np.array(agent.visited_states)
x = data[:,0]
y = data[:,1]

smooth_scale = 8

img, extent = generate_heatmap(x, y, smooth_scale)
plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet)

plt.show()

# manager.run()
