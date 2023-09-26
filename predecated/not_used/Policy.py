import numpy as np
from collections import defaultdict

def epsilon_greedy(env, state, q, e = 0.95):
    pi = defaultdict(lambda : 0)
    
    actions = env.A(state)
    best_action_idx = 0 
    for idx, action in enumerate(actions):
        if(q[(state,action)] > q[(state,actions[best_action_idx])]):
            best_action_idx = idx
    
    for idx, action in enumerate(actions):
        if(idx == best_action_idx):
            pi[action] = 0.95
        else:
            pi[action] = (1-e)/(len(actions)-1)
    return pi

def greedy(env, state, q):
    pi = defaultdict(lambda : 0)
    
    actions = env.A(state)
    best_action_idx = 0 
    for idx, action in enumerate(actions):
        if(q[(state,action)] > q[(state,actions[best_action_idx])]):
            best_action_idx = idx
    
    for idx, action in enumerate(actions):
        if(idx == best_action_idx):
            pi[action] = 1
        else:
            pi[action] = 0
    return pi

def random(env, state, q):
    pi = defaultdict(lambda : 0)
    actions = env.A(state)
    for action in actions:
        pi[action] = 1/len(actions)
        
    return pi
    