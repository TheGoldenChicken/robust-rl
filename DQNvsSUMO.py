#%% TESTING DQN vs SUMO

import sys
import os

sys.path.append('sumo') # Add current folder to path
sys.path.append('cliff_car') # Add current folder to path
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set the current working directory to the folder this file is in

# Environment
from sumo.sumo_pp import SumoPPEnv
import cliff_car_again

# Agents
from sumo import sumo_ensemble
from sumo import sumo_agent
from sumo import DQN_sumo_agent
from sumo import DQN_sumo_ensemble
import cliff_car_ensemble
import cliff_car_agent
import cliff_car_DQN

# Utils
import numpy as np
import pickle

def load_agents(env, DQN_agent_class, DQN_ensemble_agent_class, ensemble_agent_class, agent_class, DQN_paths, paths, deltas):
    """
    Load all agents in a specific environment
    """
    # Load all agents
    DQN_agents = [DQN_agent_class(env, replay_buffer=None, epsilon_decay=None, model_path=path, target_update = 0) for path in DQN_paths]
    DQN_ensemble_agent = DQN_ensemble_agent_class(env, DQN_agents)

    agents = {deltas[i] : [agent_class(env, replay_buffer = None, epsilon_decay = None, model_path = path) for path in same_delta_paths]
                for i, same_delta_paths in enumerate(paths)}
    ensemble_agents = {delta : ensemble_agent_class(env, same_seed_agents) for delta, same_seed_agents in agents.items()}

    # Set is_test to True for all agents
    for agent in DQN_agents:
        agent.is_test = True

    DQN_ensemble_agent.is_test = True

    for delta, agents_ in agents.items():
        for agent in agents_:
            agent.is_test = True

    for delta, agent in ensemble_agents.items():
        agent.is_test = True
        
    return DQN_agents, DQN_ensemble_agent, agents, ensemble_agents

def test_agent_group(agent_group, n_games = 10):
    DQN_agents, DQN_ensemble_agent, agents, ensemble_agents = agent_group
    
    print(">>> Testing (1/4): Testing DQN Agents")    
    asr_DQN_agents = np.array([agent.test(test_games = n_games) for agent in DQN_agents])
    print(">>> Testing (2/4): Testing DQN Ensemble Agent")
    asr_DQN_ensemble_agent = DQN_ensemble_agent.test(test_games = n_games)
    print(">>> Testing (3/4): Testing Agents")
    asr_agents = {delta : np.array([agent.test(test_games = n_games) for agent in agents]) for delta, agents in agents.items()}
    print(">>> Testing (4/4): Testing Ensemble Agents")
    asr_ensemble_agents = {delta : agent.test(test_games = n_games) for delta, agent in ensemble_agents.items()}
    print(">>> Testing Done!")
    
    return asr_DQN_agents, asr_DQN_ensemble_agent, asr_agents, asr_ensemble_agents

def accum_reward(asr_group):
    asr_DQN_agents, asr_DQN_ensemble_agent, asr_agents, asr_ensemble_agents = asr_group
    
    DQN_accum_r = np.sum(np.apply_along_axis(np.nan_to_num, -1, asr_DQN_agents[:,:,:,2,0]),axis=-1)
    DQN_ensemble_accum_r = np.sum(np.nan_to_num(asr_DQN_ensemble_agent[:,:,2,0]),axis=-1)
    accum_r = {delta : np.sum(np.apply_along_axis(np.nan_to_num, -1, agents[:,:,:,2,0]),axis=-1) for delta, agents in asr_agents.items()}
    ensemble_accum_r = {delta : np.sum(np.nan_to_num(agent[:,:,2,0]),axis=-1) for delta, agent in asr_ensemble_agents.items()}
    
    return DQN_accum_r, DQN_ensemble_accum_r, accum_r, ensemble_accum_r

def get_accum(env_type,
              DQN_agent_class, DQN_ensemble_agent_class, ensemble_agent_class, agent_class,
              DQN_paths, paths,
              std_env, noise_envs,
              deltas, n_games):
    no_added_noise_agents = load_agents(std_env, DQN_agent_class, DQN_ensemble_agent_class, ensemble_agent_class, agent_class, DQN_paths, paths, deltas)
    # Load all agents in all environments
    mean_shift_agents = {}
    for mean_shift in noise_envs[0]:
        mean_shift_agents[mean_shift.noise_mean] = load_agents(mean_shift, DQN_agent_class, DQN_ensemble_agent_class, ensemble_agent_class, agent_class, DQN_paths, paths, deltas)

    var_shift_agents = {}
    for var_shift in noise_envs[1]:
        var_shift_agents[var_shift.noise_var] = load_agents(var_shift, DQN_agent_class, DQN_ensemble_agent_class, ensemble_agent_class, agent_class, DQN_paths, paths, deltas)

    ### TEST ALL AGENTS ###

    # Test all agents in all environments
    print(f"Testing in ENV: mean_shift=STD, var_shift=STD")
    no_noise_asr = test_agent_group(no_added_noise_agents, n_games)

    mean_shift_asr = {}
    for noise_mean, agent_group in mean_shift_agents.items():
        print(f"Testing in ENV: mean_shift={noise_mean}, var_shift=STD")
        mean_shift_asr[noise_mean] = test_agent_group(agent_group, n_games)

    var_shift_asr = {}
    for noise_var, agent_group in var_shift_agents.items():
        print(f"Testing in ENV: mean_shift=STD, var_shift={noise_var}")
        var_shift_asr[noise_var] = test_agent_group(agent_group, n_games)
    print("Testing Done!")


    ### CALCULATE ACCUMULATED REWARDS ###

    print("Calculating accumulated rewards")
    no_noise_accum = accum_reward(no_noise_asr)

    mean_shift_accum = {}
    for noise_mean, asr_group in mean_shift_asr.items():
        mean_shift_accum[noise_mean] = accum_reward(asr_group)
        
    var_shift_accum = {}
    for noise_var, asr_group in var_shift_asr.items():
        var_shift_accum[noise_var] = accum_reward(asr_group)

    print("accum rewards calculated!")
    
    print("Saving results")
    with open(f'dqnVsumoAcliff_test_results/accum_reward_data/{env_type}-no_noise_accum.pkl', 'wb') as f:
        pickle.dump(no_noise_accum, f)
    with open(f'dqnVsumoAcliff_test_results/accum_reward_data/{env_type}-mean_shift_accum.pkl', 'wb') as f:
        pickle.dump(mean_shift_accum, f)
    with open(f'dqnVsumoAcliff_test_results/accum_reward_data/{env_type}-var_shift_accum.pkl', 'wb') as f:
        pickle.dump(var_shift_accum, f)
    print("Results saved!")
    
    return no_noise_accum, mean_shift_accum, var_shift_accum

### SUMO ENVIRONMENT ###
    
# Define mean and variance of environment noise
# sumo_noise_mean = []#0.14, 0.32, 0.45, 1.0, 1.41, 2.0, 2.45, 2.83, 3.16, 3.46, 3.74, 4.0, 4.24, 4.47, 6.32]
# sumo_noise_var = 10+np.array([0.65, 1.55, 2.3, 6.21, 10.28, 18.59, 28.1, 39.42, 53.06, 69.59, 89.71, 114.23, 144.11, 180.59, 1464.1])

# # Load all environmnets
# sumo_envs = [[SumoPPEnv(noise_mean=mean, noise_var=0) for mean in sumo_noise_mean],
#         [SumoPPEnv(noise_mean=0, noise_var=var) for var in sumo_noise_var]]
# sumo_std_env = SumoPPEnv()

# # Define all deltas and seeds
# sumo_deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2,
#           0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5]
# sumo_seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
# sumo_linear = False
# sumo_n_games = 100

# sumo_training_length = 8000

# if sumo_training_length == 12000: sumo_training_type = "longertraining"
# else: sumo_training_type = "newoptim"

# # Get path to all agents
# DQN_sumo_model_paths = [f'sumo/test_results/DQN_sumo-{sumo_training_length}-frames/seed-{seed}-model' for seed in sumo_seeds]
# sumo_model_paths = [[f'sumo/test_results/{sumo_training_type}-linear-{sumo_linear}-test_seed_{seed}_robust_factor_-1/{delta}-model'
#                         for seed in sumo_seeds] for delta in sumo_deltas]

# sumo_no_noise_accum, sumo_mean_shift_accum, sumo_var_shift_accum = \
#     get_accum("sumo",
#               DQN_sumo_agent.DQNSumoAgent, DQN_sumo_ensemble.EnsembleDQNSumoTestAgent, sumo_ensemble.EnsembleSumoTestAgent, sumo_agent.SumoAgent,
#               DQN_sumo_model_paths, sumo_model_paths,
#               sumo_std_env, sumo_envs,
#               sumo_deltas, sumo_n_games)

## CLIFF CAR ENVIRONMENT ###

# Define mean and variance of environment noise
cliff_car_noise_mean = []#0.14, 0.32, 0.45, 1.0, 1.41, 2.0, 2.45, 2.83, 3.16, 3.46, 3.74, 4.0, 4.24, 4.47, 6.32]
cliff_car_noise_var = np.array([6.98,7.38,7.7,9.29,10.8,22.07,41.99,126.92,356.9,2680.17])

# Load all environmnets
cliff_car_envs = [[cliff_car_again.CliffCar(noise_mean=mean, noise_var=0) for mean in cliff_car_noise_mean],
        [cliff_car_again.CliffCar(noise_mean=0, noise_var=var) for var in cliff_car_noise_var]]
cliff_car_std_env = cliff_car_again.CliffCar()

# Define all deltas and seeds
cliff_car_deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]
cliff_car_seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
cliff_car_linear = False
cliff_car_n_games = 100

cliff_car_training_length = 12000

cliff_car_training_type = "newoptim"

# Get path to all agents
DQN_cliff_car_model_paths = [f'cliff_car/test_results/DQN_cliffcar-{cliff_car_training_length}-frames/seed-{seed}-model' for seed in cliff_car_seeds]
cliff_car_model_paths = [[f'cliff_car/test_results/Cliffcar-{cliff_car_training_type}-linear-{cliff_car_linear}-test_seed_{seed}_robust_factor_-1/{delta}-model'
                        for seed in cliff_car_seeds] for delta in cliff_car_deltas]

cliff_car_no_noise_accum, cliff_car_mean_shift_accum, cliff_car_var_shift_accum = \
    get_accum("cliff_car",
              cliff_car_DQN.DQNCliffCarAgent, cliff_car_ensemble.EnsembleCliffTestAgent, cliff_car_ensemble.EnsembleCliffTestAgent, cliff_car_agent.CliffCarAgent,
              DQN_cliff_car_model_paths, cliff_car_model_paths,
              cliff_car_std_env, cliff_car_envs,
              cliff_car_deltas, cliff_car_n_games)



### no_noise_accum architecture
# no_noise_accum = (DQN_accum_r, DQN_ensemble_accum_r, accum_r, ensemble_accum_r)
#   DQN_accum_r.shape = (n_seeds, n_games) = (10, 100)
#   DQN_ensemble_accum_r.shape = (n_games) = (100,)
#   accum_r.shape = {n_deltas : (n_seeds, n_games} = {16/10 : (10, 100)}
#   ensemble_accum_r.shape = {n_deltas : (n_games} = {16/10 : (100,)}

### mean_shift_accum architecture
# mean_shift_accum = {noise_mean : (DQN_accum_r, DQN_ensemble_accum_r, accum_r, ensemble_accum_r)}
#   DQN_accum_r.shape = (n_seeds, n_games) = (10, 100)
#   DQN_ensemble_accum_r.shape = (n_games) = (100,)
#   accum_r.shape = {n_deltas : (n_seeds, n_games} = {16/10 : (10, 100)}
#   ensemble_accum_r.shape = {n_deltas : (n_games} = {16/10 : (100,)}

### var_shift_accum architecture
# var_shift_accum = {noise_var : (DQN_accum_r, DQN_ensemble_accum_r, accum_r, ensemble_accum_r)}
#   DQN_accum_r.shape = (n_seeds, n_games) = (10, 100)
#   DQN_ensemble_accum_r.shape = (n_games) = (100,)
#   accum_r.shape = {n_deltas : (n_seeds, n_games} = {16/10 : (10, 100)}
#   ensemble_accum_r.shape = {n_deltas : (n_games} = {16/10 : (100,)}


#%% PLOTTING
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# ### SUMO PARAMETERS ###
deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2,
          0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5]
seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
noise_var = 10+np.array([0.65, 1.55, 2.3, 6.21, 10.28, 18.59, 28.1, 39.42, 53.06, 69.59, 89.71, 114.23, 144.11, 180.59, 1464.1])

env_type = "sumo"

# # CLIFF CAR PARAMETERS ###
# deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]
# seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
# noise_var = np.array([6.98,7.38,7.7,9.29,10.8,22.07,41.99,126.92,356.9,2680.17])

# env_type = "cliff_car"

### LOAD ALL RESULTS ###
print("Loading results")
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/{env_type}-no_noise_accum.pkl', 'rb') as f:
    no_noise_accum = pickle.load(f)
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/{env_type}-mean_shift_accum.pkl', 'rb') as f:
    mean_shift_accum = pickle.load(f)
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/{env_type}-var_shift_accum.pkl', 'rb') as f:
    var_shift_accum = pickle.load(f)
    
print("Results loaded!")

### PLOT ALL RESULTS ###
print("Plotting results")

### DQN NO NOISE BOXPLOT PERFORMANACE ###

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Plotting for all seeds
for i, seed in enumerate(seeds):
    ax.boxplot(no_noise_accum[0][i], positions=[i], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})

# Plotting for ensemble
ax.boxplot(no_noise_accum[1], positions=[i+1], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})

ax.set_title(f"{env_type}, DQN accumulated reward, No augmentation: var_shift=STD")
ax.set_xlabel("Seed/model")
ax.set_ylabel("Accumulated Reward")
ax.set_xticks(range(len(seeds) + 1))
ax.set_xticklabels(seeds + ["Ensemble"])
plt.savefig(f'dqnVsumoAcliff_test_results/{env_type}_DQN_all_seeds_no_noise_accum_boxplot.png', dpi=300)
plt.show()

### DQN AND SUMO ENSAMBLE NO NOISE BOXPLOT PERFORMANACE ###

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

# Plotting for all delta values
for i, delta in enumerate(deltas):
    ax.boxplot(no_noise_accum[3][delta], positions=[i], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})

    # Make a t-test between DQN and SUMO
    t, p = ttest_ind(no_noise_accum[3][delta], no_noise_accum[1])
    
    # Plot the p-value just above the x-axis aligned with the boxplots
    ax.text(i + 0.05, 0.98, f"p={p:.2f}", ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=10)
    
# Plotting for ensemble
ax.boxplot(no_noise_accum[1], positions=[i+1], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})

# Plot a horizontal line for the mean of the DQN ensemble
plt.axhline(y=np.mean(no_noise_accum[1]), color='gray', linestyle='--', label='DQN ensemble mean', linewidth=1.5, alpha=0.5)

ax.set_title(f"{env_type} ensemble. Robust and DQN. No Augmentation: var=20; KL(p||p_0)={0}")
ax.set_xlabel("Agent (delta value)")
ax.set_ylabel("Accumulated Reward")
ax.set_xticks(range(len(deltas) + 1))
ax.set_xticklabels(deltas + ["DQN"])
plt.savefig(f'dqnVsumoAcliff_test_results/{env_type}_DQN_ensemble_no_noise_accum_boxplot.png', dpi=500)
plt.show()


### DQN AND SUMO ENSAMBLE VARIANCE NOISE ENVIRONMENTS

# p_values = np.zeros((len(deltas), len(deltas)))

for i, (kl, var) in enumerate(zip(deltas,noise_var)):
    
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    
    # Plotting for all delta values
    for j, delta in enumerate(deltas):
        ax.boxplot(var_shift_accum[var][3][delta], positions=[j], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})
        
        # Make a t-test between DQN and SUMO
        t, p = ttest_ind(var_shift_accum[var][3][delta], var_shift_accum[var][1])
        
        # p_values[i,j] = (p>0.05)
        
        # Plot the p-value just above the x-axis aligned with the boxplots
        ax.text(j + 0.05, 0.98, f"p={p:.2f}", ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=10)
        
    # Plotting for ensemble
    ax.boxplot(var_shift_accum[var][1], positions=[j+1], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})
    
    # Plot a horizontal line for the mean of the DQN ensemble
    plt.axhline(y=np.mean(var_shift_accum[var][1]), color='gray', linestyle='--', label='DQN ensemble mean', linewidth=1.5, alpha=0.5)
    

    ax.set_title(f"{env_type} ensemble. Robust and DQN. Augmentation: var={np.round(var,2)}; KL(p||p_0)={kl}")
    ax.set_xlabel("Agent (delta value)")
    ax.set_ylabel("Accumulated Reward")
    ax.set_xticks(range(len(deltas) + 1))
    ax.set_xticklabels(deltas + ["DQN"])
    plt.savefig(f'dqnVsumoAcliff_test_results/{env_type}_DQN_ensemble_var-{np.round(var,2)}_accum_boxplot.png', dpi = 500)
    plt.show()
    
# Plot the q values as a heatmap
# fig, ax = plt.subplots(1, 1, figsize=(12,5))

# # ax.imshow(p_values, cmap='hot')

# fig.colorbar(ax.imshow(p_values), ax=ax)

# plt.show()




