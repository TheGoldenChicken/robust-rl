from sumo import sumo_ensemble
from sumo.sumo_agent import SumoAgent
from sumo.sumo_pp import SumoPPEnv
from matplotlib import pyplot as plt
import sum_plot
import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import AxesGrid
from sumo import DQN_sumo_agent
from sumo import DQN_sumo_ensemble
import os

env = SumoPPEnv()

def load_agents(paths):
    
    agents = []
    for path in paths:
        agent = SumoAgent(env, replay_buffer=None, epsilon_decay=None, model_path=path)
        agent.is_test = True
        agents.append(agent)
        
    return agents

# deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2,
#           0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
# seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]

# deltas = [0.1,0.2]
# seeds = [22]
training_type = "newoptim"
linear = False
robust_factor = -1


deltas = [ 3, 5]#0.001, 0.005, 0.01, 0.05, 0.1, 0.2,
         # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2,
seeds = [4242, 6942, 6969, 123, 420, 5318008, 22, 23, 99, 10]
# training_type = "longertraining"
# linear = False


DQN_iter = 12000

def plot_q_vals(q_vals, q_vals_var = None, same_plot=True, vertical_lines=False, axis = None):

    q_mean_1 = q_vals[:, 0]
    q_mean_2 = q_vals[:, 1]
    q_mean_3 = q_vals[:, 2]
    
    if q_vals_var is not None:
        q_var_1 = q_vals_var[:, 0]
        q_var_2 = q_vals_var[:, 1]
        q_var_3 = q_vals_var[:, 2]

        axis.fill_between(np.arange(len(q_mean_1)), q_mean_1 - q_var_1, q_mean_1 + q_var_1, alpha=0.2, color='red')
        axis.fill_between(np.arange(len(q_mean_2)), q_mean_2 - q_var_2, q_mean_2 + q_var_2, alpha=0.2, color='blue')
        axis.fill_between(np.arange(len(q_mean_3)), q_mean_3 - q_var_3, q_mean_3 + q_var_3, alpha=0.2, color='green')

    axis.plot(q_mean_1, color='red', label='Q-value: NoOp')
    axis.plot(q_mean_2, color='blue', label='Q-value: Right')
    axis.plot(q_mean_3, color='green', label='Q-value: Left')

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

def plot_average_q_of_seeds(seeds, delta_vals, linear=True, axis = None, DQN = False):
    

    if not DQN:
        paths = [[f'sumo/test_results/{training_type}-linear-{linear}-test_seed_{seed}_robust_factor_{robust_factor}/{delta}-q_vals.npy'
                        for seed in seeds] for delta in delta_vals]
    else:
        paths = [[f'sumo/test_results/DQN_sumo-{DQN_iter}-frames/seed-{seed}-q_vals.npy' for seed in seeds]]
        
    for i in range(len(paths)):
        q_vals = np.array([sum_plot.get_q_vals(path=path) for path in paths[i]])
        plot_q_vals(np.mean(q_vals, axis=0), np.var(q_vals, axis=0), vertical_lines=True, axis = axis)

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

def plot_q_and_state_hist(test_games = 200, DQN = False):
    
    
    for delta in deltas:
        print("Starting delta: ", delta)
        fig, (ax1, ax3) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [16, 5]})
    
        ax2 = ax1.twinx()
        
        if not DQN:
            paths_linear = [f'sumo/test_results/{training_type}-linear-{linear}-test_seed_{seed}_robust_factor_{robust_factor}/{delta}-model'
                            for seed in seeds]

            q_paths = [f'sumo/test_results/{training_type}-linear-{linear}-test_seed_{seed}_robust_factor_{robust_factor}/{delta}-q_vals.npy'
                        for seed in seeds]
            
            agents = load_agents(paths_linear)
            ensemble_agent = sumo_ensemble.EnsembleSumoTestAgent(env, agents)
        else:
            paths_linear = [f'sumo/test_results/DQN_sumo-{DQN_iter}-frames/seed-{seed}-model'
                            for seed in seeds]

            q_paths = [f'sumo/test_results/DQN_sumo-{DQN_iter}-frames/seed-{seed}-q_vals.npy'
                        for seed in seeds]

            agents = load_agents(paths_linear)
            ensemble_agent = DQN_sumo_ensemble.EnsembleDQNSumoTestAgent(env, agents)
            
        q_vals = np.array([sum_plot.get_q_vals(path=path) for path in q_paths])
        q_vals = np.transpose(q_vals, axes = (1,0,2))
    
        ax3.imshow(get_image_from_q(q_vals), aspect='auto')

        all_sar = ensemble_agent.test(test_games=test_games)

        if DQN:
            # os.mkdir("sumo/test_results/sumo_ensemble")
            np.save(f"sumo/test_results/sumo_ensemble/DQN-all_sar-games-{test_games}.npy", all_sar, allow_pickle=True)
        else:
            # os.mkdir("sumo/test_results/sumo_ensemble")
            np.save(f"sumo/test_results/sumo_ensemble/{delta}-all_sar-games-{test_games}.npy", all_sar, allow_pickle=True)

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
        
        if not DQN:
            plt.title(f'Q-values for converged agent, delta={delta}')
            
            # Adjust the vertical spacing
            plt.subplots_adjust(hspace=0, wspace=0)
            # plt.show(
            
            # Increase surrounding white space of plot
            plt.tight_layout()
            plt.savefig(f'plots/q-vals/{training_type}-ensemble-q-and-state-{linear}-{delta}.png', dpi=300)
        else:
            plt.title(f'Q-values for ensemble DQN agent')
            
            # Adjust the vertical spacing
            plt.subplots_adjust(hspace=0, wspace=0)
            # plt.show(
            
            # Increase surrounding white space of plot
            plt.tight_layout()
            print("saving plot")
            plt.savefig(f'plots/q-vals/{training_type}-DQN-{DQN_iter}-ensemble-q-and-state.png', dpi=300)
            break
            


def plot_state_hist_multiple_delta(test_games = 200):
    
    
    figure = plt.figure()

    grid = AxesGrid(figure, 111,
                    nrows_ncols=(len(deltas) + 1, 1),
                    axes_pad=0,
                    share_all=True,
                    label_mode="all",
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="0.5%",
                    cbar_pad = 0.1,
                    )
    
    paths_linear = [f'sumo/test_results/DQN_sumo-{DQN_iter}-frames/seed-{seed}-model'
                        for seed in seeds]

    agents = load_agents(paths_linear)

    ensemble_agent = sumo_ensemble.EnsembleSumoTestAgent(env, agents)

    all_sar = ensemble_agent.test(test_games=test_games)

    # Replace all nan values with 0
    all_sar = np.nan_to_num(all_sar)

    # Create a bar plot of all the states visisted in the test_games
    bins = 50
    
    img_height = 80
    
    img = np.array([np.histogram(all_sar[:,:,0].flatten(), bins = bins, range=(0,1200), density = True)[0],
                    np.histogram(all_sar[:,:,0].flatten(), bins = bins, range=(0,1200), density = True)[0],
                    np.histogram(all_sar[:,:,0].flatten(), bins = bins, range=(0,1200), density = True)[0]])
    im = grid[0].imshow(img, label = f"DQN")
    
    # Vertical line at the cliff (1000)
    cliff_line = grid[0].axvline(x=50*0.83, color='black', linestyle='--', label = 'Cliff')
    
    # Vertical line at the start position (240)
    start_line = grid[0].axvline(x=50*0.2, color='red', linestyle='--', label = 'Start')
    
    # Create text box with the delta value
    textstr = f'DQN'
    props = dict(boxstyle='round', facecolor='white', alpha=1)

    # place a text box in upper left in axes coords
    grid[0].text(0.02,0.7, textstr, transform=grid[0].transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    
    grid[0].set_yticks([])
    grid[0].set_xticks([])
    
    grid[0].set_title("State distribution heatmap")
    
    
    for i, (delta, ax) in enumerate(zip(deltas,grid[1:])):
        
        print("Stating: ", delta)
        
        paths_linear = [f'sumo/test_results/{training_type}-linear-{linear}-test_seed_{seed}_robust_factor_{robust_factor}/{delta}-model'
                        for seed in seeds]

        agents = load_agents(paths_linear)

        ensemble_agent = sumo_ensemble.EnsembleSumoTestAgent(env, agents)

        all_sar = ensemble_agent.test(test_games=test_games)

        # Replace all nan values with 0
        all_sar = np.nan_to_num(all_sar)

        # Create a bar plot of all the states visisted in the test_games
        bins = 50
        
        img_height = 80
        
        img = np.array([np.histogram(all_sar[:,:,0].flatten(), bins = bins, range=(0,1200), density = True)[0],
                        np.histogram(all_sar[:,:,0].flatten(), bins = bins, range=(0,1200), density = True)[0],
                        np.histogram(all_sar[:,:,0].flatten(), bins = bins, range=(0,1200), density = True)[0]])
        im = ax.imshow(img, label = f"Delta={delta}")
        
        # Vertical line at the cliff (1000)
        cliff_line = ax.axvline(x=50*0.83, color='black', linestyle='--', label = 'Cliff')
        
        # Vertical line at the start position (240)
        start_line = ax.axvline(x=50*0.2, color='red', linestyle='--', label = 'Start')
        
        # Create text box with the delta value
        textstr = f'Delta={delta}'
        props = dict(boxstyle='round', facecolor='white', alpha=1)

        # place a text box in upper left in axes coords
        ax.text(0.02,0.7, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        
        ax.set_yticks([])

        if i == len(deltas)-1:
            ax.set_xlabel('State')
            ax.set_xticks([0,8.3,16.6,24.9,33.2,41.5],[0,200,400,600,800,1000])
        else:
            ax.set_xticks([])
            
        
    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.set_yticks([])
        cax.toggle_label(False)
        
    
        

    plt.savefig(f'plots/q-vals/{training_type}-ensemble-all-delta-state-{linear}.png', dpi=300)
    
    
def plot_q_as_image_multiple_delta():
    
    # Figure with aspect ratio
    fig, axs = plt.subplots(len(deltas) + 1, 1)

    # Remove vertical distance between plots
    fig.subplots_adjust(hspace=0)
    
    
    axs[0].set_title(f'State/optimal action diagram')
    
    # Plot the q-values for the DQN {DQN_iter} agent
    paths = [f'sumo/test_results/DQN_sumo-{DQN_iter}-frames/seed-{seed}-q_vals.npy' for seed in seeds]
    q_vals = np.array([sum_plot.get_q_vals(path=path) for path in paths])
    q_vals = np.transpose(q_vals, axes = (1,0,2))  
    
    img = get_image_from_q(q_vals)
    
    axs[0].imshow(img, label = f"DQN")
    
    # Create text box with the delta value
    textstr = f'DQN'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    
    # place a text box in upper left in axes coords
    axs[0].text(0.02,0.7, textstr, transform=axs[0].transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    
    # Remove x ticks
    axs[0].set_yticks([])

    
    for i, delta in enumerate(deltas):
        paths = [f'sumo/test_results/{training_type}-linear-{linear}-test_seed_{seed}_robust_factor_{robust_factor}/{delta}-q_vals.npy'
                        for seed in seeds]
        
        q_vals = np.array([sum_plot.get_q_vals(path=path) for path in paths])

        q_vals = np.transpose(q_vals, axes = (1,0,2))
        
        img = get_image_from_q(q_vals)
        
        
        axs[i+1].imshow(img, label = f"Delta={delta}")
        
        # Create text box with the delta value
        textstr = f'Delta={delta}'
        props = dict(boxstyle='round', facecolor='white', alpha=1)

        # place a text box in upper left in axes coords
        axs[i+1].text(0.02,0.7, textstr, transform=axs[i+1].transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        # Custom label without colored box. Only text saying "Delta=0.1"
        # patch = mpatches.Patch(color='None', label=f"Delta={delta}")


        # axs[i+1].legend(handles=[patch], loc = 2)
        
        # Remove x ticks
        axs[i+1].set_yticks([])
        
        if i == len(deltas)-1:
            axs[i+1].set_xlabel('State')
        else:
            axs[i+1].set_xticks([])
            
            
        cliff_line = axs[i+1].axvline(x=1000, color='k', linestyle='--', label = f"Cliff")
        start_line = axs[i+1].axvline(x=240, color='r', linestyle='--', label = f"Start")
        

    cliff_line = axs[0].axvline(x=1000, color='k', linestyle='--', label = f"Cliff")
    start_line = axs[0].axvline(x=240, color='r', linestyle='--', label = f"Start")
        
        
    patch0 = mpatches.Patch(color='red', label=f"NoOp")
    patch2 = mpatches.Patch(color='blue', label=f"Right")
    patch1 = mpatches.Patch(color='green', label=f"Left")
    plt.legend(handles=[patch0, patch1, patch2, cliff_line,start_line], loc = 4)
    
    plt.show()
    # plt.savefig(f'plots/q-vals/{training_type}-ensemble-all-delta-q-vals-{linear}.png', dpi=300)


def plot_DQN_performance(test_games = 200):
    
    
    for seed in seeds:
        print("Starting delta: ", seed)
        fig, (ax1, ax3) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [16, 5]})
    

        ax2 = ax1.twinx()
        
        model_path = f'sumo/test_results/DQN_sumo-{DQN_iter}-frames/seed-{seed}-model'

        q_path = f'sumo/test_results/DQN_sumo-{DQN_iter}-frames/seed-{seed}-q_vals.npy'

        q_vals = sum_plot.get_q_vals(path=q_path)
        # q_vals = np.transpose(q_vals, axes = (1,0,2))
        
        actions = np.argmax(q_vals, axis = 1)
        
        NOOPT = 0; RIGHT = 1; LEFT = 2
    
        colors = np.zeros((1200,3))
        colors[actions == NOOPT] = [1,0,0]
        colors[actions == RIGHT] = [0,0,1]
        colors[actions == LEFT] = [0,1,0]

        # copy the first axis 80 times
        img = np.broadcast_to(colors, (80,1200,3))
        
        
        ax3.imshow(img, aspect='auto')

        
        DQN_agent = DQN_sumo_agent.DQNSumoAgent(env, replay_buffer=None, epsilon_decay=None, target_update = 0, model_path=model_path)

        all_sar = DQN_agent.test(test_games=test_games)
        

        # Create a bar plot of all the states visisted in the test_games
        ax2.hist(all_sar[:,:,0].flatten(), bins=100, label = f"State distribution", alpha=0.5, density=True)
        
        plot_q_vals(q_vals, vertical_lines=True, axis = ax1)
        
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
        plt.title(f'Q-values for DQN agent with seed {seed}')
        
        # Adjust the vertical spacing
        plt.subplots_adjust(hspace=0, wspace=0)
        # plt.show(
        
        # Increase surrounding white space of plot
        plt.tight_layout()

        plt.savefig(f'plots/q-vals/{training_type}-DQN-{DQN_iter}_performance-{seed}.png', dpi=300)

def plot_sumo_states_individual(seeds, delta_vals, linear=True):
    
    paths = [[f'sumo/test_results/{training_type}-linear-{linear}-test_seed_{seed}_robust_factor_{robust_factor}/{delta}-test_data.npy'
                     for seed in seeds] for delta in delta_vals]
    
    fig, axs = plt.subplots(1,len(delta_vals),figsize=(13,4))
    
    fig.tight_layout(pad = 1.5)
    
    # Combine (not average) the states from all seeds
    for i, paths_ in enumerate(paths):
        for j, path in enumerate(paths_):
            sar_data = np.load(path)
            sar_data = sar_data[:,:,0].flatten()
            
            axs[i].hist(sar_data, bins=100, label = f"seed: {seeds[j]}", alpha=0.5)
        axs[i].set_title(f"Delta: {delta_vals[i]}")
        axs[i].set_xlabel("State")
        axs[i].legend()
    
    plt.savefig(f'plots/q-vals/{training_type}-individual-states-{linear}.png', dpi=300)
    
    plt.show()
    
def print_acum_return(seeds, delta_vals, linear=True):
    
    paths = [[f'sumo/test_results/{training_type}-linear-{linear}-test_seed_{seed}_robust_factor_{robust_factor}/{delta}-test_data.npy'
                     for seed in seeds] for delta in delta_vals]
    
    fig, axs = plt.subplots(1,len(delta_vals),figsize=(15,4))
    
    # Combine (not average) the states from all seeds
    for i, paths_ in enumerate(paths):
        acum_returns = []
        for j, path in enumerate(paths_):
            sar_data = np.load(path)
            acum_return = np.mean(np.sum(np.nan_to_num(sar_data[:,:,2]),axis=-1))
            acum_returns.append(acum_return)
            print(f"seed:{seeds[j]},delta:{delta_vals[i]},{acum_return}")
        print(f"delta:{delta_vals[i]},mean:{np.mean(acum_returns)},std:{np.std(acum_returns)}")
        

# seeds = [4242, 6942, 420, 5318008, 23]
# deltas = [0.01,0.05,0.1, 0.5,1]
    
# print_acum_return([6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10], [0.001,0.01,0.1,1,2], linear=False)    q 
    
# PLotting the state distributions individually
# plot_sumo_states_individual([4242, 6942, 420, 5318008, 23],[0.01,0.05,0.1, 0.5,1],linear=False)

# Plot all q values and state distributions for all seeds seperately
# plot_DQN_performance()

# Plot all q values and state distributions as an ensemble method
plot_q_and_state_hist(DQN = True, test_games = 100)

# Plot all q values and state distributions as an ensemble method
plot_q_and_state_hist(test_games = 100)

# Plot the Q values as decision diagrams for all delta values (ensemble method)
# plot_q_as_image_multiple_delta()

# Plot the state distributions as heatmap for all delta values (ensemble method)
# plot_state_hist_multiple_delta()

