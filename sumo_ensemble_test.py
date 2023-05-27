from sumo import sumo_ensemble
from sumo.sumo_agent import SumoAgent
from sumo.sumo_pp import SumoPPEnv
from matplotlib import pyplot as plt
import sum_plot
import numpy as np
import matplotlib.patches as mpatches

env = SumoPPEnv()

def load_agents(paths):
    
    agents = []
    for path in paths:
        agent = SumoAgent(env, replay_buffer=None, epsilon_decay=None, model_path=path)
        agent.is_test = True
        agents.append(agent)
        
    return agents

# deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5,1,2]
deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2,
          0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
# deltas = []
seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
linear = False

def plot_q_vals(q_vals, q_vals_var, delta=None, same_plot=True, vertical_lines=False, axis = None):

    q_mean_1 = q_vals[:, 0]
    q_mean_2 = q_vals[:, 1]
    q_mean_3 = q_vals[:, 2]
    
    q_var_1 = q_vals_var[:, 0]
    q_var_2 = q_vals_var[:, 1]
    q_var_3 = q_vals_var[:, 2]

    axis.fill_between(np.arange(len(q_mean_1)), q_mean_1 - q_var_1, q_mean_1 + q_var_1, alpha=0.2, color='red')
    axis.fill_between(np.arange(len(q_mean_2)), q_mean_2 - q_var_2, q_mean_2 + q_var_2, alpha=0.2, color='blue')
    axis.fill_between(np.arange(len(q_mean_3)), q_mean_3 - q_var_3, q_mean_3 + q_var_3, alpha=0.2, color='green')

    axis.plot(q_mean_1, color='red', label='Q-value: NoOp')
    axis.plot(q_mean_2, color='blue', label='Q-value: Right')
    axis.plot(q_mean_3, color='green', label='Q-value: Left')
    # axis.title(f'Q-values for converged agent with Delta {delta}')

    if vertical_lines:
        # Add lines to indicate where right becomes better action than left
        indices = np.where(q_mean_3 > q_mean_2)[0]
        indices1 = np.where(q_mean_1 > q_mean_2)[0]

        # axis.axvline(x=indices1[0], color='green', linestyle='--', label = f'Left > Right: {indices[0]}')
        # axis.axvline(x=indices[0], color='blue', linestyle='--', label = f'NoOp > Right: {indices1[0]}')
        axis.axvline(x=1000, color='black', linestyle='--', label=f'Cliff at {1000}')
        axis.axvline(x=240, color='red', linestyle='--', label=f'Start Pos at {240}')

        # ax1.text(500, 0.2, f'Left > Right: {indices[0]}\nNoOp > Right: {indices1[0]}\nCliff: {1000}\nStart Pos: 240',
        #          rotation=0, va='bottom')

def plot_average_q_of_seeds(seeds, delta_vals, linear=True, axis = None):

    paths = [[f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy'
                     for seed in seeds] for delta in delta_vals]

    for i in range(len(paths)):
        q_vals = np.array([sum_plot.get_q_vals(path=path) for path in paths[i]])
        plot_q_vals(np.mean(q_vals, axis=0), np.var(q_vals, axis=0), delta=delta_vals[i], vertical_lines=True, axis = axis)

def get_image_from_q(q_vals, width = 80):
    # Majority voting for the q-values between all seeds
    actions = np.argmax(np.apply_along_axis(np.bincount,-1,np.argmax(q_vals,axis=-1), minlength=3),axis=-1)
    
    NOOPT = 0; RIGHT = 1; LEFT = 2
    
    colors = np.zeros((1200,3))
    colors[actions == NOOPT] = [1,0,0]
    colors[actions == RIGHT] = [0,0,1]
    colors[actions == LEFT] = [0,1,0]

    # copy the first axis 80 times
    img = np.broadcast_to(colors, (80,1200,3))
    
    return img

def plot_q_and_state_hist(test_games = 200):
    for delta in deltas:
        print("Starting delta: ", delta)
        fig, (ax1, ax3) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [16, 5]})
    

        ax2 = ax1.twinx()
        
        paths_linear = [f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-model'
                        for seed in seeds]

        q_paths = [f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy'
                     for seed in seeds]

        q_vals = np.array([sum_plot.get_q_vals(path=path) for path in q_paths])
        q_vals = np.transpose(q_vals, axes = (1,0,2))
    
        
        ax3.imshow(get_image_from_q(q_vals), aspect='auto')

        agents = load_agents(paths_linear)

        ensemble_agent = sumo_ensemble.EnsembleSumoTestAgent(env, agents)

        all_sar = ensemble_agent.test(test_games=test_games)

        # Create a bar plot of all the states visisted in the test_games
        ax2.hist(all_sar[:,:,0].flatten(), bins=100, label = f"State distribution", alpha=0.5, density=True)
        
        plot_average_q_of_seeds(seeds, [delta], linear = linear, axis = ax1)    
        
        patch1 = mpatches.Patch(color='red', label='Majority vote: Noop')
        patch2 = mpatches.Patch(color='blue', label='Majority vote: Right')
        patch3 = mpatches.Patch(color='green', label='Majority vote: Left')
        
        ax3.legend(handles=[patch1, patch2, patch3], loc = 2)
        
        # Remove xticks from ax1 and ax2
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax3.set_yticks([])
        
        ax1.legend(loc = 2)
        ax2.legend(loc = 4)
        ax1.set_ylabel('Q-value')
        ax2.set_ylabel('State density')
        ax3.set_xlabel('State')
        plt.title(f'Q-values for converged agent, delta={delta}')
        
        # Adjust the vertical spacing
        plt.subplots_adjust(hspace=0, wspace=0)
        # plt.show(
        
        # Increase surrounding white space of plot
        plt.tight_layout()

        plt.savefig(f'plots/q-vals/ensemble-q-and-state-{linear}-{delta}.png', dpi=300)
        


def plot_state_hist_multiple_delta(test_games = 200):
    
    fig, axs = plt.subplots(len(deltas), 1)
    
    # Remove vertical distance between plots
    fig.subplots_adjust(hspace=0)
    
    for i, delta in enumerate(deltas):
        
        print("Stating: ", delta)
        paths_linear = [f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-model'
                        for seed in seeds]

        agents = load_agents(paths_linear)

        ensemble_agent = sumo_ensemble.EnsembleSumoTestAgent(env, agents)

        all_sar = ensemble_agent.test(test_games=test_games)

        # Create a bar plot of all the states visisted in the test_games
        bins = 75
        img = np.array([np.histogram(all_sar[:,:,0].flatten(), bins = bins)[0],
                        np.histogram(all_sar[:,:,0].flatten(), bins = bins)[0]])
        axs[i].imshow(img, label = f"Delta={delta}")
        
        patch = mpatches.Patch(color='blue', label=f"Delta={delta}")
        # Remove x ticks
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        
        axs[i].legend(handles=[patch], loc = 2)
        
        
    # plt.legend(loc = 2)
    # plt.ylabel('State density')
    # plt.xlabel('State')
    # plt.title(f'State histogram')

    # Save the figure
    plt.tight_layout()

    plt.savefig(f'plots/q-vals/ensemble-all-delta-state-{linear}.png', dpi=300)
    
    
def plot_q_as_image_multiple_delta():
    
    # Figure with aspect ratio
    fig, axs = plt.subplots(len(deltas), 1)

    # Remove vertical distance between plots
    fig.subplots_adjust(hspace=0)
    
    
    axs[0].set_title(f'State/optimal action diagram')
    
    for i, delta in enumerate(deltas):
        paths = [f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy'
                        for seed in seeds]
        
        q_vals = np.array([sum_plot.get_q_vals(path=path) for path in paths])

        q_vals = np.transpose(q_vals, axes = (1,0,2))
        
        img = get_image_from_q(q_vals)
        
        
        axs[i].imshow(img, label = f"Delta={delta}")
        
        
        # Custom label without colored box. Only text saying "Delta=0.1"
        # patch = mpatches.Patch(color='None', label=f"Delta={delta}")


        # axs[i].legend(handles=[patch], loc = 2)
        
        # Remove x ticks
        axs[i].set_yticks([40],[f"Delta: {delta}"])
        
        if i != len(deltas) - 1 :
            axs[i].set_xticks([])
        else:
            axs[i].set_xlabel('State')
            
        cliff_line = axs[i].axvline(x=1000, color='k', linestyle='--', label = f"Cliff")
        start_line = axs[i].axvline(x=240, color='r', linestyle='--', label = f"Start")
        
        
    patch0 = mpatches.Patch(color='red', label=f"NoOp")
    patch2 = mpatches.Patch(color='blue', label=f"Right")
    patch1 = mpatches.Patch(color='green', label=f"Left")
    plt.legend(handles=[patch0, patch1, patch2, cliff_line,start_line], loc = 4)
    
    
    # plt.legend(loc = 2)
    # plt.ylabel('State density')
    # plt.xlabel('State')
    # plt.show()
    
    # Save the figure in plots/q-vals

    # Draw a vertical dotted line at state 1000 (cliff)
    

    # plt.tight_layout()

    plt.savefig(f'plots/q-vals/ensemble-all-delta-q-vals-{linear}.png', dpi=300)


# plot_q_and_state_hist()
plot_q_as_image_multiple_delta()
# plot_state_hist_multiple_delta()