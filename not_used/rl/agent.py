from collections import defaultdict
import Policy as policy
import random
import numpy as np
import matplotlib.pyplot as plt
# import pytorch as torch

class Agent:
    
    def __init__(self, env) -> None:
        self.env = env
        
        # replay_buffer = {(state, action) : (state_, reward)}
        self.replay_buffer = defaultdict(lambda : tuple(list, float))
        
        # visisted_states = {state}
        self.visisted_states = set()
        
        self.state = self.env.reset()
    
    # Decide an action from a given state
    # Returns True if the environment is done (won or lost)
    def next(self) -> bool:
        return False
    
# Shallow agent for discrete environments or continuous environments with a small state space
class ShallowAgent(Agent):
    
    def __init__(self, env) -> None:
        super().__init__(env)
        
        # self.v = {state : value}
        self.V = defaultdict(lambda : 0)
        
        # self.q = {(state, action) : value}
        self.Q = defaultdict(lambda : 0)

# Deep agent for continuous environments or discrete environments with a large state space
class DeepAgent(Agent):
    
    def __init__(self, env, weight_size) -> None:
        super().__init__(env)
        
        # self.w = torch.rand(weight_size)
    
    









class DiscreteQLearningAgent(Agent):
    
    def __init__(self, env) -> None:
        super().__init__(env)
        


class DiscreteActionValueIterationAgent(Agent):
    
    def __init__(self, env, policy = policy.random) -> None:
        super().__init__(env)
        self.policy = policy
        
    def next(self) -> bool:
        # Action value iteration function
        q = defaultdict(lambda : 0)
        
        for state in self.env.get_states():
            for action in self.env.A(state):
                trans_prob = self.env.get_transistion_probabilities(state, action)
                
                temp = 0
                for next_state, reward in trans_prob:
                    pi = self.policy(self.env, next_state, self.q)
                    temp += trans_prob[(next_state, reward)]*(reward + 0.99*sum([pi[action_] * self.q[(next_state,action_)] for action_ in self.env.A(next_state)]))
                
                q[(state,action)] = temp
        
        self.q = q
        
        # Value iteration function
        v = defaultdict(lambda : 0)
        for state in self.env.get_states():
            actions = self.env.A(state)
            for action in actions:
                pi = 1/len(actions)
                temp = 0
                trans_prob = self.env.get_transistion_probabilities(state, action)
                
                # Get reward, next state    
                for next_state, reward in trans_prob.keys():
                    
                    temp += trans_prob[(next_state,reward)]*(reward + 0.99*self.v[next_state])
                
                pi = self.policy(self.env, state, self.q)
                v[state] += pi[action] * temp
                
        self.v = v
        
        return True

class ManualAgent(Agent):
    
    def __init__(self, env) -> None:
        super().__init__(env)
        
    def next(self):
        
        while(True):
            print("Current state: " + str(self.env.state))
            print("Total reward: " + str(self.env.get_accumulative_reward(self)))
            print("---------------")
            print("Enter next action")
            print("Avaliable actions: " + str(self.env.A(self.state)))
            try:
                action = input()
                
                if action == "exit":
                    return False
                
                action = int(action)
                
                print("\n")
                
                if(action <= self.env.A(self.state)[-1]):
                    
                    next_state, reward = self.env.step(self.state, action)
                    
                    self.state = next_state
                    self.visited_states.append(self.state)
                    self.previous_actions.append(action)
                    self.obtained_rewards.append(reward)
        
                    print("---------------")
                    print("Reward: " + str(reward))
                    print("---------------")
                    return True
                else:
                    print("Invalid action")
                
                
            except ValueError:
                print("The provided string is not a valid representation of an integer.\n"+
                        "Please enter a valid integer in the action space")
    
    