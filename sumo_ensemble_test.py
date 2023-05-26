from sumo import sumo_ensemble
from sumo.sumo_agent import SumoAgent
from sumo.sumo_pp import SumoPPEnv
from matplotlib import pyplot as plt
import sum_plot
import numpy as np

env = SumoPPEnv()

def load_agents(paths):
    
    agents = []
    for path in paths:
        agent = SumoAgent(env, replay_buffer=None, epsilon_decay=None, model_path=path)
        agent.is_test = True
        agents.append(agent)
        
    return agents
    

# q_vals = sum_plot.get_q_vals(path = f'sumo/test_results/newoptim-linear-True-test_seed_{6969}_robust_factor_-1/{0.001}-q_vals.npy')

# sum_plot.plot_q_vals(q_vals, delta=0.001, vertical_lines=True)

deltas = [0.001, 0.005,0.01,0.05,0.1,0.5,1,2]
# deltas = []
seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
linear = True

def plot_q_vals(q_vals, delta=None, same_plot=True, vertical_lines=False):

    column1 = q_vals[:, 0]
    column2 = q_vals[:, 1]
    column3 = q_vals[:, 2]

    if not same_plot:
        ax1.figure(figsize=(15, 5))
        ax1.subplot(131)
        ax1.plot(column1)
        ax1.title(f'Action 0, {delta}')
        ax1.subplot(132)
        ax1.plot(column2)
        ax1.title('Action 1')
        ax1.subplot(133)
        ax1.plot(column3)
        ax1.title('Action 2')
        ax1.tight_layout()
    else:
        ax1.plot(column1, color='red', label='0: Noop')
        ax1.plot(column2, color='blue', label='1: Right (towards cliff)')
        ax1.plot(column3, color='green', label='2: Left (away from cliff)')
        # ax1.title(f'Q-values for converged agent with Delta {delta}')

    if vertical_lines:
        # Add lines to indicate where right becomes better action than left
        indices = np.where(column3 > column2)[0]
        indices1 = np.where(column1 > column2)[0]

        ax1.axvline(x=indices1[0], color='green', linestyle='--', label = f'Left > Right: {indices[0]}')
        ax1.axvline(x=indices[0], color='blue', linestyle='--', label = f'NoOp > Right: {indices1[0]}')
        ax1.axvline(x=1000, color='black', linestyle='--', label=f'Cliff: {1000}')
        ax1.axvline(x=240, color='red', linestyle='--', label=f'Start Pos: {240}')

        # ax1.text(500, 0.2, f'Left > Right: {indices[0]}\nNoOp > Right: {indices1[0]}\nCliff: {1000}\nStart Pos: 240',
        #          rotation=0, va='bottom')

def plot_average_of_seeds(seeds, delta_vals, linear=True):

    paths = [[f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-q_vals.npy'
                     for seed in seeds] for delta in delta_vals]

    for i in range(len(paths)):
        q_vals = np.array([sum_plot.get_q_vals(path=path) for path in paths[i]])
        plot_q_vals(np.mean(q_vals, axis=0), delta=delta_vals[i], vertical_lines=True)


for delta in deltas:
    
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    
    paths_linear = [f'sumo/test_results/newoptim-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-model'
                    for seed in seeds]

    agents = load_agents(paths_linear)

    ensemble_agent = sumo_ensemble.EnsembleSumoTestAgent(env, agents)

    all_sar = ensemble_agent.test(test_games=10)

    # Create a bar plot of all the states visisted in the test_games
    ax2.hist(all_sar[:,:,0].flatten(), bins=100, label = f"State distribution", alpha=0.5, density=True)
    
    plot_average_of_seeds(seeds, [delta], linear = linear)    
    
    ax1.legend(loc = 2)
    ax2.legend(loc = 4)
    ax1.set_ylabel('Q-value')
    ax2.set_ylabel('State density')
    ax1.set_xlabel('State')
    plt.title(f'Q-values for converged agent, delta={delta}')
    plt.show()
    
    
    