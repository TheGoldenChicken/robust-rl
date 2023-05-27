from cliff_car import cliff_car_ensemble
from cliff_car.cliff_car_agent import CliffCarAgent
from cliff_car.cliff_car_again import CliffCar
from matplotlib import pyplot as plt
import sum_plot
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches

env = CliffCar()

def load_agents(paths):
    
    agents = []
    for path in paths:
        agent = CliffCarAgent(env, replay_buffer=None, epsilon_decay=None, model_path=path)
        agent.is_test = True
        agents.append(agent)
        
    return agents

def generate_heatmap(x, y, axis, std = 8):
    # layout_size = [11,7]
    # y = layout_size[1] - y
    # x = layout_size[1] - x

    bins = int(75*75)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 150], [0, 150]])
    heatmap = gaussian_filter(heatmap, sigma=std)
    heatmap[heatmap <= 0] = np.nan
    
    heatmap = np.log(heatmap+0.1)+1 # Bringing it into a nicer scale

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    #flip the heatmap
    heatmap = np.flip(heatmap, axis=1)
    
    # Make all bins with 0 values transparent
    
    
    heat_plot = axis.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet, alpha=0.5, label = 'States visited')

    return heat_plot
    # plt.show()
    # return heatmap.T, extent

def plot_q_vals(seeds, delta, axis):
    
    data = []
    for seed in seeds:
        d = np.load(f'cliff_car/test_results/Cliffcar-newoptim-linear-False-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy', allow_pickle=True).reshape(150,150,5)
        data.append(d)
    
    q_vals = np.array([d.reshape(150,150,5) for d in data])
    
    # Move the first dimension to the back and ranspose the second and third dimenstion dimentions
    # (0,1,2,3) -> (2,1,0,3)
    q_vals = np.transpose(q_vals, axes=(2,1,0,3))

    # Make a image where each pixel represents which of the 5 actions is best
    noopt = 0 # red
    down = 1 # green
    left = 2 # blue
    right = 3 # yellow
    up = 4 # up
    
    # Perform majority viting between the models
    # q_vals = np.argmax(q_vals, axis=2)
    # order = q_vals
    ranks = q_vals.argsort(axis=2).argsort(axis=2)
    
    best_actions = np.argmax(np.apply_along_axis(np.bincount,-1,np.argmax(q_vals,axis=-1),minlength=5),axis=-1)

    # Make a color map
    colors = np.zeros((150,150,3))
    colors[best_actions == noopt] = [255,0,0]
    colors[best_actions == right] = [0,255,0]
    colors[best_actions == left] = [0,0,255]
    colors[best_actions == down] = [255,255,0]
    colors[best_actions == up] = [255,0,255]

    # Plot the color map
    axis.imshow(colors.astype(np.uint8))
    
    goal_plot = axis.scatter([112.5],[33.3],label = 'Goal')
    start_plot = axis.scatter([33.3],[33.3],label = 'Start')

    # Dotted Horizontal lin through image at y = 150 - 30
    cliff = axis.plot([0,149],[30,30],'--',color='black', label = 'Cliff')

    return goal_plot, start_plot, cliff
    


deltas = [0.001,0.005]#,0.005,0.01,0.05,0.1,0.5,1,2]
# deltas = []
seeds = [9005,9006]
linear = False

fig, axs = plt.subplots(1, len(deltas), figsize=(len(deltas)*5, 4))

for i, delta in enumerate(deltas):
    

    paths_linear = [f'Cliff_car/test_results/Cliffcar-newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-model'
                    for seed in seeds]

    agents = load_agents(paths_linear)

    ensemble_agent = cliff_car_ensemble.EnsembleCliffTestAgent(env, agents)

    all_sar = ensemble_agent.test(test_games=100)
    
    
    x = []
    y = []
    for game in all_sar:
        for step in game:
            if step[0] is not np.nan:
                x.append(step[0][0])
                y.append(step[0][1])
    x = np.array(x)/10
    y = np.abs(150-np.array(y)/10)
    
    goal, start, cliff = plot_q_vals(seeds, delta, axs[i])
    heat_plot = generate_heatmap(x, y, axs[i])
    
    opopt_patch = mpatches.Patch(color='red', label='OpOpt')
    down_patch = mpatches.Patch(color='green', label='Up')
    left_patch = mpatches.Patch(color='blue', label='Left')
    right_patch = mpatches.Patch(color='yellow', label='Right')
    up_patch = mpatches.Patch(color='purple', label='Down')

    axs[i].legend(handles=[opopt_patch, down_patch, left_patch, right_patch, up_patch, goal, start], loc='upper left')

    # # Change the x and y ticks to be between 0 and 1
    # plt.xticks(np.around(np.linspace(0,149,6),decimals=2), np.around(np.linspace(0,1,6), decimals=2))
    # plt.yticks(np.around(np.linspace(0,149,6),decimals=2), np.around(np.linspace(0,1,6), decimals=2))

    # x and y labels
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')

    # Change the title
    axs[i].set_title(f'Delta: {delta}')
    
    # plot_average_of_seeds(seeds, [delta], linear = linear)    
    
    # ax1.legend(loc = 2)
    # ax2.legend(loc = 4)
    # ax1.set_ylabel('Q-value')
    # ax2.set_ylabel('State density')
    # ax1.set_xlabel('State')
    # plt.title(f'Q-values for converged agent, delta={delta}')
    # plt.show()

plt.show()
    
    