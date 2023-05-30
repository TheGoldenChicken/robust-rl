#%% TESTING DQN vs SUMO vs CLIFF CAR

import sys
import os

sys.path.append('sumo') # Add current folder to path
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set the current working directory to the folder this file is in

# Environment
from sumo.sumo_pp import SumoPPEnv

# Agents
from sumo import sumo_ensemble
from sumo import sumo_agent
from sumo import DQN_sumo_agent
from sumo import DQN_sumo_ensemble

# Utils
import numpy as np
import pickle

def load_agents(env, DQN_sumo_path, sumo_path, deltas):
    """
    Load all agents in a specific environment
    """
    # Load all agents
    DQN_sumo_agents = [DQN_sumo_agent.DQNSumoAgent(env, replay_buffer=None, epsilon_decay=None, model_path=path, target_update = 0) for path in DQN_sumo_path]
    DQN_ensemble_sumo_agent = DQN_sumo_ensemble.EnsembleDQNSumoTestAgent(env, DQN_sumo_agents)

    sumo_agents = {deltas[i] : [sumo_agent.SumoAgent(env, replay_buffer = None, epsilon_decay = None, model_path = path) for path in same_delta_paths]
                for i, same_delta_paths in enumerate(sumo_path)}
    sumo_ensemble_agents = {delta : sumo_ensemble.EnsembleSumoTestAgent(env, same_seed_agents) for delta, same_seed_agents in sumo_agents.items()}

    # Set is_test to True for all agents
    for agent in DQN_sumo_agents:
        agent.is_test = True

    DQN_ensemble_sumo_agent.is_test = True

    for delta, agents in sumo_agents.items():
        for agent in agents:
            agent.is_test = True

    for delta, agent in sumo_ensemble_agents.items():
        agent.is_test = True
        
    return DQN_sumo_agents, DQN_ensemble_sumo_agent, sumo_agents, sumo_ensemble_agents

def test_agent_group(agent_group, n_games = 10):
    DQN_sumo_agents, DQN_ensemble_sumo_agent, sumo_agents, sumo_ensemble_agents = agent_group
    
    print(">>> Testing (1/4): Testing DQN Sumo Agents")    
    asr_DQN_sumo_agents = np.array([agent.test(test_games = n_games) for agent in DQN_sumo_agents])
    print(">>> Testing (2/4): Testing DQN Sumo Ensemble Agent")
    asr_DQN_ensemble_sumo_agent = DQN_ensemble_sumo_agent.test(test_games = n_games)
    print(">>> Testing (3/4): Testing Sumo Agents")
    asr_sumo_agents = {delta : np.array([agent.test(test_games = n_games) for agent in agents]) for delta, agents in sumo_agents.items()}
    print(">>> Testing (4/4): Testing Sumo Ensemble Agents")
    asr_sumo_ensemble_agents = {delta : agent.test(test_games = n_games) for delta, agent in sumo_ensemble_agents.items()}
    print(">>> Testing Done!")
    
    return asr_DQN_sumo_agents, asr_DQN_ensemble_sumo_agent, asr_sumo_agents, asr_sumo_ensemble_agents

def accum_reward(asr_group):
    asr_DQN_sumo_agents, asr_DQN_ensemble_sumo_agent, asr_sumo_agents, asr_sumo_ensemble_agents = asr_group
    
    DQN_sumo_accum_r = np.sum(np.apply_along_axis(np.nan_to_num, -1, asr_DQN_sumo_agents[:,:,:,2]),axis=-1)
    DQN_ensemble_sumo_accum_r = np.sum(np.nan_to_num(asr_DQN_ensemble_sumo_agent[:,:,2]),axis=-1)
    sumo_accum_r = {delta : np.sum(np.apply_along_axis(np.nan_to_num, -1, agents[:,:,:,2]),axis=-1) for delta, agents in asr_sumo_agents.items()}
    sumo_ensemble_accum_r = {delta : np.sum(np.nan_to_num(agent[:,:,2]),axis=-1) for delta, agent in asr_sumo_ensemble_agents.items()}
    
    return DQN_sumo_accum_r, DQN_ensemble_sumo_accum_r, sumo_accum_r, sumo_ensemble_accum_r

# Define mean and variance of environment noise
noise_mean = []#0.14, 0.32, 0.45, 1.0, 1.41, 2.0, 2.45, 2.83, 3.16, 3.46, 3.74, 4.0, 4.24, 4.47, 6.32]
noise_var = 10+np.array([0.65, 1.55, 2.3, 6.21, 10.28, 18.59, 28.1, 39.42, 53.06, 69.59, 89.71, 114.23, 144.11, 180.59, 1464.1])

# Load all environmnets
envs = [[SumoPPEnv(noise_mean=mean, noise_var=0) for mean in noise_mean],
        [SumoPPEnv(noise_mean=0, noise_var=var) for var in noise_var]]
std_env = SumoPPEnv()

# Define all deltas and seeds
deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2,
          0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]
seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
linear = False
n_games = 100

training_length = 8000

if training_length == 12000: training_type = "longertraining"
else: training_type = "newoptim"

# Get path to all agents
DQN_sumo_model_paths = [f'sumo/test_results/DQN_sumo-{training_length}-frames/seed-{seed}-model' for seed in seeds]
sumo_model_paths = [[f'sumo/test_results/{training_type}-linear-{linear}-test_seed_{seed}_robust_factor_-1/{delta}-model'
                        for seed in seeds] for delta in deltas]

### LOAD ALL AGENTS ###

# Load all agents in standard environment (no extra noise)
no_added_noise_sumo_agents = load_agents(std_env, DQN_sumo_model_paths, sumo_model_paths, deltas)

# Load all agents in all environments
mean_shift_sumo_agents = {}
for mean_shift in envs[0]:
    mean_shift_sumo_agents[mean_shift.noise_mean] = load_agents(mean_shift, DQN_sumo_model_paths, sumo_model_paths, deltas)

var_shift_sumo_agents = {}
for var_shift in envs[1]:
    var_shift_sumo_agents[var_shift.noise_var] = load_agents(var_shift, DQN_sumo_model_paths, sumo_model_paths, deltas)

### TEST ALL AGENTS ###

# Test all agents in all environments
print(f"Testing in ENV: mean_shift=STD, var_shift=STD")
no_noise_asr = test_agent_group(no_added_noise_sumo_agents, n_games)

mean_shift_sumo_asr = {}
for noise_mean, agent_group in mean_shift_sumo_agents.items():
    print(f"Testing in ENV: mean_shift={noise_mean}, var_shift=STD")
    mean_shift_sumo_asr[noise_mean] = test_agent_group(agent_group, n_games)

var_shift_sumo_asr = {}
for noise_var, agent_group in var_shift_sumo_agents.items():
    print(f"Testing in ENV: mean_shift=STD, var_shift={noise_var}")
    var_shift_sumo_asr[noise_var] = test_agent_group(agent_group, n_games)
print("Testing Done!")


### CALCULATE ACCUMULATED REWARDS ###

print("Calculating accumulated rewards")
no_noise_accum = accum_reward(no_noise_asr)

mean_shift_accum = {}
for noise_mean, asr_group in mean_shift_sumo_asr.items():
    mean_shift_accum[noise_mean] = accum_reward(asr_group)
    
var_shift_accum = {}
for noise_var, asr_group in var_shift_sumo_asr.items():
    var_shift_accum[noise_var] = accum_reward(asr_group)

print("accum rewards calculated!")

### no_noise_accum architecture
# no_noise_accum = (DQN_sumo_accum_r, DQN_ensemble_sumo_accum_r, sumo_accum_r, sumo_ensemble_accum_r)
#   DQN_sumo_accum_r.shape = (n_seeds, n_games) = (10, 12)
#   DQN_ensemble_sumo_accum_r.shape = (n_games) = (12,)
#   sumo_accum_r.shape = {n_deltas : (n_seeds, n_games} = {16 : (10, 12)}
#   sumo_ensemble_accum_r.shape = {n_deltas : (n_games} = {16 : (12,)}

### mean_shift_accum architecture
# mean_shift_accum = {noise_mean : (DQN_sumo_accum_r, DQN_ensemble_sumo_accum_r, sumo_accum_r, sumo_ensemble_accum_r)}
#   DQN_sumo_accum_r.shape = (n_seeds, n_games) = (10, 12)
#   DQN_ensemble_sumo_accum_r.shape = (n_games) = (12,)
#   sumo_accum_r.shape = {n_deltas : (n_seeds, n_games} = {16 : (10, 12)}
#   sumo_ensemble_accum_r.shape = {n_deltas : (n_games} = {16 : (12,)}

### var_shift_accum architecture
# var_shift_accum = {noise_var : (DQN_sumo_accum_r, DQN_ensemble_sumo_accum_r, sumo_accum_r, sumo_ensemble_accum_r)}
#   DQN_sumo_accum_r.shape = (n_seeds, n_games) = (10, 12)
#   DQN_ensemble_sumo_accum_r.shape = (n_games) = (12,)
#   sumo_accum_r.shape = {n_deltas : (n_seeds, n_games} = {16 : (10, 12)}
#   sumo_ensemble_accum_r.shape = {n_deltas : (n_games} = {16 : (12,)}

### Saving all results using Pickle ###
print("Saving results")
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/no_noise_accum.pkl', 'wb') as f:
    pickle.dump(no_noise_accum, f)
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/mean_shift_accum.pkl', 'wb') as f:
    pickle.dump(mean_shift_accum, f)
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/var_shift_accum.pkl', 'wb') as f:
    pickle.dump(var_shift_accum, f)
print("Results saved!")

#%% PLOTTING
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2,
          0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]
seeds = [6969, 4242, 6942, 123, 420, 5318008, 23, 22, 99, 10]
noise_var = 10+np.array([0.65, 1.55, 2.3, 6.21, 10.28, 18.59, 28.1, 39.42, 53.06, 69.59, 89.71, 114.23, 144.11, 180.59, 1464.1])


### LOAD ALL RESULTS ###
print("Loading results")
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/no_noise_accum.pkl', 'rb') as f:
    no_noise_accum = pickle.load(f)
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/mean_shift_accum.pkl', 'rb') as f:
    mean_shift_accum = pickle.load(f)
with open(f'dqnVsumoAcliff_test_results/accum_reward_data/var_shift_accum.pkl', 'rb') as f:
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

ax.set_title("DQN accumulated reward, No augmentation: var_shift=STD")
ax.set_xlabel("Seed/model")
ax.set_ylabel("Accumulated Reward")
ax.set_xticks(range(len(seeds) + 1))
ax.set_xticklabels(seeds + ["Ensemble"])
plt.savefig(f'dqnVsumoAcliff_test_results/sumo_DQN_all_seeds_no_noise_accum_boxplot.png', dpi=300)
plt.show()

### DQN AND SUMO ENSAMBLE NO NOISE BOXPLOT PERFORMANACE ###

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Plotting for all delta values
for i, delta in enumerate(deltas):
    ax.boxplot(no_noise_accum[3][delta], positions=[i], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})

    # Make a t-test between DQN and SUMO
    t, p = ttest_ind(no_noise_accum[3][delta], no_noise_accum[1])
    
    # Plot the p-value just above the x-axis aligned with the boxplots
    ax.text(i + 0.05, 0.98, f"p={p:.3f}", ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=8)

# Plotting for ensemble
ax.boxplot(no_noise_accum[1], positions=[i+1], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})

# Plot a horizontal line for the mean of the DQN ensemble
plt.axhline(y=np.mean(no_noise_accum[1]), color='gray', linestyle='--', label='DQN ensemble mean', linewidth=1.5, alpha=0.5)

ax.set_title(f"Sumo ensemble. Robust and DQN. No Augmentation: var=20; KL(p||p_0)={0}")
ax.set_xlabel("Agent (delta value)")
ax.set_ylabel("Accumulated Reward")
ax.set_xticks(range(len(deltas) + 1))
ax.set_xticklabels(deltas + ["DQN"])
plt.savefig(f'dqnVsumoAcliff_test_results/SUMO_DQN_ensemble_no_noise_accum_boxplot.png', dpi=500)
plt.show()


### DQN AND SUMO ENSAMBLE VARIANCE NOISE ENVIRONMENTS

p_values = np.zeros((len(deltas), len(noise_var)))

for i, (kl, var) in enumerate(zip(deltas,noise_var)):
    
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    
    # Plotting for all delta values
    for j, delta in enumerate(deltas):
        ax.boxplot(var_shift_accum[var][3][delta], positions=[j], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})
        
        # Make a t-test between DQN and SUMO
        t, p = ttest_ind(var_shift_accum[var][3][delta], var_shift_accum[var][1])
        
        p_values[j,i] = p
        
        # Plot the p-value just above the x-axis aligned with the boxplots
        ax.text(j + 0.05, 0.98, f"p={p:.3f}", ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=8)
        
    # Plotting for ensemble
    ax.boxplot(var_shift_accum[var][1], positions=[j+1], widths=0.6, showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2})
    
    # Plot a horizontal line for the mean of the DQN ensemble
    plt.axhline(y=np.mean(var_shift_accum[var][1]), color='gray', linestyle='--', label='DQN ensemble mean', linewidth=1.5, alpha=0.5)
    

    ax.set_title(f"Sumo ensemble. Robust and DQN. Augmentation: var={np.round(var,2)}; KL(p||p_0)={kl}")
    ax.set_xlabel("Agent (delta value)")
    ax.set_ylabel("Accumulated Reward")
    ax.set_xticks(range(len(deltas) + 1))
    ax.set_xticklabels(deltas + ["DQN"])
    plt.savefig(f'dqnVsumoAcliff_test_results/SUMO_DQN_ensemble_var-{np.round(var,2)}_accum_boxplot.png', dpi = 500)
    plt.show()
    
# Plot the q values as a heatmap
fig, ax = plt.subplots(1, 1, figsize=(10,5))

# ax.imshow(p_values, cmap='hot')

fig.colorbar(ax.imshow(p_values), ax=ax)

plt.show()

### DQN AND SUMO ENSAMBLE MEAN accum OVER ENV VARIANCE
#%%

fig, ax = plt.subplots(1, 1, figsize=(30,12))

means = np.zeros((len(noise_var), len(deltas)))
conf_int = np.zeros((len(noise_var), len(deltas), 2))

DQN_means = np.zeros(len(noise_var))
DQN_conf_int = np.zeros((len(noise_var), 2))

for i, (kl, var) in enumerate(zip(deltas,noise_var)):
    
    # Plotting for all delta values
    for j, delta in enumerate(deltas):
        
        means[i][j] = np.mean(var_shift_accum[var][3][delta])
        # print(np.percentile(var_shift_accum[var][3][delta], [2.5, 97.5]))
        conf_int[i][j] = np.percentile(var_shift_accum[var][3][delta], [2.5, 97.5])
    
    DQN_means[i] = np.mean(var_shift_accum[var][1])
    DQN_conf_int[i] = np.percentile(var_shift_accum[var][1], [2.5, 97.5])

for i, (m, cis) in enumerate(zip(means.T, np.transpose(conf_int,(1,0,2)))):
    m_ = np.vstack((m,DQN_means))
    cis_ = np.hstack((cis,DQN_conf_int))
    for j, (diff, ci) in enumerate(zip(m_.T,cis_)):
        ax.plot([i*2 - (1-j/20),i*2+0.1 + j/20],diff,'-o',color='r')
        # Draw the confidence intervals
        ax.plot([i*2 - (1-j/20), i*2 - (1-j/20)],ci[:2],'--',color='b')
        ax.plot([i*2+0.1 + j/20,i*2+0.1 + j/20],ci[2:],'--',color='g')
        
    # print(m_.shape)
    # ax.plot(np.arange(len(m)), m, '-o')
ax.set_xticks(range(15))
ax.set_xticklabels(deltas)
plt.show()




