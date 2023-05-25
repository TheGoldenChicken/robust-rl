
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter



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

# Load .npy file from test_results

data = np.load('cliff_car/test_results/Cliffcar-newoptim-linear-True-test_seed_6969_robust_factor_-1/0.01-test_data.npy', allow_pickle=True)


run = np.array([[d[0], d[1]] for d in data[0,:,0] if not np.isnan(d).any()])
x = run[:,0]
y = run[:,1]

smooth_scale = 8

plt.scatter(x,y)
plt.show()
# generate_heatmap(x, y, smooth_scale)




# manager.run()
