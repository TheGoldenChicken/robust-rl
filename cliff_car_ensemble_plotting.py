#%%
import os
import sys

sys.path.append('.')
# Set the current working directory to the folder this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from cliff_car import cliff_car_ensemble
from cliff_car.cliff_car_agent import CliffCarAgent
from cliff_car.cliff_car_again import CliffCar
from matplotlib import pyplot as plt
import sum_plot
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
import torch

env = CliffCar()

after_exam = True

if after_exam:
    addon = 'After-Exam-'

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
    
    
    heat_plot = axis.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet, alpha=0.6, label = 'States visited')

    return heat_plot
    # return heatmap.T, extent

def plot_q_vals(seeds, delta, axis, DQN_qvals = None):
    
    data = []
    if DQN_qvals is None:
        for seed_ in seeds:
            d = np.load(f'cliff_car/test_results/{addon}Cliffcar-newoptim-linear-{linear}-test_seed_{seed_}_robust_factor_-1/{delta}-q_vals.npy', allow_pickle=True).reshape(150,150,5)
            data.append(d)
        
        q_vals = np.array([d.reshape(150,150,5) for d in data])
    else:
        q_vals = DQN_qvals.reshape(len(seeds),150,150,5)
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
    colors[best_actions == noopt] = [200,0,0]
    colors[best_actions == right] = [0,200,0]
    colors[best_actions == left] = [0,200,200]
    colors[best_actions == down] = [200,200,0]
    colors[best_actions == up] = [200,0,200]

    # Plot the color map
    axis.imshow(colors.astype(np.uint8))
    
    axis.scatter([112.5],[33.3], color = "black", marker = "*", s = 70)
    axis.scatter([33.3],[33.3],label = 'Start', color = "black", marker = "o", s = 45)
    goal_plot = axis.scatter([112.5],[33.3],label = 'Goal', color = "white", marker = "*", s = 60)
    start_plot = axis.scatter([33.3],[33.3],label = 'Start', color = "white", marker = "o", s = 35)
    
    # Dotted Horizontal lin through image at y = 150 - 30
    cliff = axis.plot([0,149],[30,30],'--',color='black', label = 'Cliff')

    return goal_plot, start_plot, cliff
    


deltas = [0.05,0.5,0.1,0.5,1,2,3]
# deltas = []
seeds = [22,23,99,123,420,4242,6942,6969]#,9000,9001,9002,9003,9004,9005,

# seeds = [7777]
# deltas = [0.1]
linear = False
DQN_frames = 12000


def plot_all_seeds(DQN = False):
    for i, delta in enumerate(deltas):
        
        fig, axs = plt.subplots(1, 1)

        if DQN:
            paths_linear = [f'cliff_car/test_results/DQN_cliffcar-{DQN_frames}-frames/seed-{seed}-model'
                            for seed in seeds]
        else:
            paths_linear = [f'cliff_car/test_results/{addon}Cliffcar-newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-model'
                            for seed in seeds]

        agents = load_agents(paths_linear)

        ensemble_agent = cliff_car_ensemble.EnsembleCliffTestAgent(env, agents)

        all_sar = ensemble_agent.test(test_games=100)
        
        
        x = []
        y = []
        for game in all_sar:
            for step in game:
                if step[0] is not np.nan and step[0][0] != 0 and step[0][1] != 0:
                    x.append(step[0][0])
                    y.append(step[0][1])
        x = np.array(x)/10
        y = np.abs(150-np.array(y)/10)
        
        if DQN:
            DQN_qvals = []
            for agent in agents:
                X, Y = np.mgrid[0:1:150j, 0:1:150j]
                xy = np.vstack((X.flatten(), Y.flatten())).T
                states = torch.FloatTensor(xy).to(agent.device)

                DQN_qvals.append(agent.get_q_vals(states))
            DQN_qvals = np.array(DQN_qvals)
            
            goal, start, cliff = plot_q_vals(seeds, delta, axs, DQN_qvals = DQN_qvals)
        else:
            # goal, start, cliff = plot_q_vals(seeds, delta, axs[i])
            goal, start, cliff = plot_q_vals(seeds, delta, axs, DQN_qvals = None)
            
        # heat_plot = generate_heatmap(x, y, axs[i])
        heat_plot = generate_heatmap(x, y, axs)
        
        opopt_patch = mpatches.Patch(color='red', label='OpOpt')
        down_patch = mpatches.Patch(color='green', label='Up')
        left_patch = mpatches.Patch(color='cyan', label='Left')
        right_patch = mpatches.Patch(color='yellow', label='Right')
        up_patch = mpatches.Patch(color='purple', label='Down')

        # axs[i].legend(handles=[opopt_patch, down_patch, left_patch, right_patch, up_patch, goal, start], loc='upper left')
        axs.legend(handles=[opopt_patch, down_patch, left_patch, right_patch, up_patch, goal, start], loc='upper left')

        # # Change the x and y ticks to be between 0 and 1
        plt.xticks(np.around(np.linspace(0,149,6),decimals=2), np.around(np.linspace(0,1,6), decimals=2))
        plt.yticks(np.around(np.linspace(0,149,6),decimals=2), np.around(np.linspace(0,1,6), decimals=2))

        # x and y labels
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        # Change the title
        # axs[i].set_title(f'Delta: {delta}')
        
        # plot_average_of_seeds(seeds, [delta], linear = linear)    
        fig.colorbar(heat_plot, ticks=[], label='State density')
        # ax1.legend(loc = 2)
        # ax2.legend(loc = 4)
        # ax1.set_ylabel('Q-value')
        # ax2.set_ylabel('State density')
        # ax1.set_xlabel('State')
        if DQN:
            plt.title(f'DQN Cliff Car. Decision plane')
            plt.savefig(f'plots/q-vals/Cliffcar-DQN-{DQN_frames}-frames.png')
            plt.close()
            break
        else:
            plt.title(f'Ensemble Cliff Car. Decision plane, delta={delta}')
            
            # Save the figure
            
            plt.savefig(f'plots/q-vals/{addon}Cliffcar-ensemble-{linear}-test-{delta}.png')
            plt.close()

        # plt.show()
    
def plot_individual_seeds(DQN = False):
    for seed in seeds:
        for i, delta in enumerate(deltas):
            
            fig, axs = plt.subplots(1, 1)
            DQN_q_vals = None
            if DQN:
                paths_linear = [f'cliff_car/test_results/DQN_cliffcar-{DQN_frames}-frames/seed-{seed}-model']
                
                
            else:
                paths_linear = [f'cliff_car/test_results/{addon}Cliffcar-newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-model']

            agents = load_agents(paths_linear)

            if DQN:
                DQN_q_vals = []
                X, Y = np.mgrid[0:1:150j, 0:1:150j]
                xy = np.vstack((X.flatten(), Y.flatten())).T
                states = torch.FloatTensor(xy).to(agents[0].device)

                DQN_q_vals.append(agents[0].get_q_vals(states))

                DQN_q_vals = np.array(DQN_q_vals)
                
            ensemble_agent = cliff_car_ensemble.EnsembleCliffTestAgent(env, agents)

            all_sar = ensemble_agent.test(test_games=100)
            
            
            x = []
            y = []
            for game in all_sar:
                for step in game:
                    if step[0] is not np.nan and step[0][0] != 0 and step[0][1] != 0:
                        x.append(step[0][0])
                        y.append(step[0][1])
            x = np.array(x)/10
            y = np.abs(150-np.array(y)/10)
            
            # goal, start, cliff = plot_q_vals(seeds, delta, axs[i])
            goal, start, cliff = plot_q_vals([seed], delta, axs, DQN_qvals = DQN_q_vals)
            # heat_plot = generate_heatmap(x, y, axs[i])
            heat_plot = generate_heatmap(x, y, axs)
            
            opopt_patch = mpatches.Patch(color='red', label='OpOpt')
            down_patch = mpatches.Patch(color='green', label='Up')
            left_patch = mpatches.Patch(color='cyan', label='Left')
            right_patch = mpatches.Patch(color='yellow', label='Right')
            up_patch = mpatches.Patch(color='purple', label='Down')

            # axs[i].legend(handles=[opopt_patch, down_patch, left_patch, right_patch, up_patch, goal, start], loc='upper left')
            axs.legend(handles=[opopt_patch, down_patch, left_patch, right_patch, up_patch, goal, start], loc='upper left')

            # # Change the x and y ticks to be between 0 and 1
            plt.xticks(np.around(np.linspace(0,149,6),decimals=2), np.around(np.linspace(0,1,6), decimals=2))
            plt.yticks(np.around(np.linspace(0,149,6),decimals=2), np.around(np.linspace(0,1,6), decimals=2))

            # x and y labels
            axs.set_xlabel('x')
            axs.set_ylabel('y')
            # Change the title
            # axs[i].set_title(f'Delta: {delta}')
            
            # plot_average_of_seeds(seeds, [delta], linear = linear)    
            fig.colorbar(heat_plot, ticks=[], label='State density')
            # ax1.legend(loc = 2)
            # ax2.legend(loc = 4)
            # ax1.set_ylabel('Q-value')
            # ax2.set_ylabel('State density')
            # ax1.set_xlabel('State')
            if DQN:
                plt.title(f'DQN Cliff Car. Decision plane, seed={seed}, delta={delta}')
        
                plt.savefig(f'plots/q-vals/DQN-{DQN_frames}-cliffcar-seed-{seed}-{linear}-test-{delta}.png')
                plt.close()
                break
            else:
                plt.title(f'Robust Cliff Car. Decision plane, seed={seed}, delta={delta}')
        
                plt.savefig(f'plots/q-vals/{addon}cliffcar-seed-{seed}-{linear}-test-{delta}.png')
                plt.close()
            
            # plt.show()

# import linear regression
from sklearn.linear_model import LinearRegression

def plot_optimal_action_distribution_mean_seeds():
    
    
    
    delta_optimal_actions = np.zeros((len(deltas), 5))
    for delta in deltas:
        for i, seed in enumerate(seeds):

            q_vals = np.load(f'cliff_car/test_results/{addon}Cliffcar-newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy', allow_pickle=True)
            
            delta_optimal_actions[i] += np.bincount(np.argmax(q_vals, axis=-1), minlength=5)/(250*250*len(seeds))
    
    action_label = {0:"noopt",1:"right",2:"left",3:"up",4:"down"}
    action_color = {0:"red",1:"orange",2:"blue",3:"limegreen",4:"magenta"}
    
    # Plot the ratios of actions for each delta
    for i in range(5):
        if(i in [1,3,4]):
            alpha = 1
            marker = "-o"
        else:
            alpha = 0.3
            marker = "-"
        axs[0].plot(np.arange(10),delta_optimal_actions[:,i],marker,color = action_color[i],label=action_label[i],alpha=alpha)

    
    # Plot a linear regression for each of the 5 actions
    lin_reg = LinearRegression()
    for i in [1,3,4]:
        lin_reg.fit(np.arange(10).reshape(-1,1),delta_optimal_actions[:,i])
        axs[0].plot(np.arange(10),lin_reg.predict(np.arange(10).reshape(-1,1)),"--",color = action_color[i])

    axs[0].set_xticks(np.arange(10),deltas)
    axs[0].set_xlabel("dDlta")
    axs[0].set_ylabel("Action ratios")
    
    axs[0].set_title("Invidual (mean)")
    
    axs[0].legend(loc = "center left")
        
def plot_optimal_action_distribution_ensemble():
    delta_optimal_actions = np.zeros((len(deltas), 5))
    for i, delta in enumerate(deltas):
        data = []
        for seed in seeds:    
            d = np.load(f'cliff_car/test_results/{addon}Cliffcar-newoptim-linear-False-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy', allow_pickle=True).reshape(150,150,5)
            data.append(d)
            
        q_vals = np.array([d.reshape(150,150,5) for d in data])

        q_vals = np.transpose(q_vals, axes=(2,1,0,3))
        
        best_actions = np.argmax(np.apply_along_axis(np.bincount,-1,np.argmax(q_vals,axis=-1),minlength=5),axis=-1)

        delta_optimal_actions[i] += np.bincount(best_actions.flatten(), minlength=5)/(250*250)
        
    action_label = {0:"noopt",1:"right",2:"left",3:"up",4:"down"}
    action_color = {0:"red",1:"orange",2:"blue",3:"limegreen",4:"magenta"}

    
    # Plot the ratios of actions for each delta
    for i in range(5):
        if(i in [1,3,4]):
            alpha = 1
            marker = "-o"
        else:
            alpha = 0.3
            marker = "-"
        axs[1].plot(np.arange(10),delta_optimal_actions[:,i],marker,color = action_color[i],label=action_label[i],alpha=alpha)
    
    
    # Plot a linear regression for each of the 5 actions
    lin_reg = LinearRegression()
    for i in [1,3,4]:
        lin_reg.fit(np.arange(10).reshape(-1,1),delta_optimal_actions[:,i])
        axs[1].plot(np.arange(10),lin_reg.predict(np.arange(10).reshape(-1,1)),"--",color = action_color[i])

    axs[1].set_xticks(np.arange(10),deltas)
    axs[1].set_xlabel("Delta")
    axs[1].set_ylabel("Action ratio")
    
    # Change the view 
    axs[1].set_ylim([0,0.35])
    axs[1].set_title("Ensemble (majority vote)")
    
    axs[1].legend(loc = "center left")
        

# for DQN_frames in [8000,10000,12000]:
#     plot_individual_seeds(DQN = True)
#     plot_all_seeds(DQN = True)
    
plot_individual_seeds()
# plt.show()
plot_all_seeds()

# fig, axs = plt.subplots(1,2,figsize=(12,5))
# plot_optimal_action_distribution_mean_seeds()
# plot_optimal_action_distribution_ensemble()
# plt.show()
    
    