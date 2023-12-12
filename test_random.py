import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load value_iter_V.npy from value_iteration/maximize/noise-0
path_max = Path('value_iteration/maximize/noise-0/')
path_min = Path('value_iteration/minimize/noise-0/')

start = np.array([1,15], dtype=np.float32) # has to be discrete for discrete agent to work
goal = np.array([15,1], dtype=np.float32)

# two 3d plots
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(10,5))
fig.tight_layout(pad=3)

for i, path in enumerate([path_max, path_min]):

    V_dict = np.load(path/'value_iter_V.npy', allow_pickle=True)

    V_items = V_dict.item()

    X = list(V_items.keys())
    y = list(V_items.values())


    # Argsort the X values
    X = np.array(X)
    idx = np.argsort(X[:,0])
    X = X[idx]
    y = np.array(y)[idx]

    axs[i].plot_trisurf(X[:,0], X[:,1], y, cmap=plt.cm.viridis, linewidth=0.2)
    
    # Rotate the axes 90 degrees around the vertical axis
    axs[i].view_init(40, 240) 
    

    axs[i].set_title(f'V(x,y), {path.parts[-2]}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')
    axs[i].set_zlabel('V')
    
plt.savefig('value_iteration/V_min_max.png', dpi=300)
plt.show()
