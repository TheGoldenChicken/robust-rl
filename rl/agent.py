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
    
    # Returns True if the environment is done (won or lost)
    def next(self) -> bool:
        return False
    
# Shallow agent for discrete environments or continuous environments with a small state space
class ShallowAgent(Agent):
    
    def __init__(self, env) -> None:
        super().__init__(env)
        
        # self.v = {state : value}
        self.v = defaultdict(lambda : 0)
        
        # self.q = {(state, action) : value}
        self.q = defaultdict(lambda : 0)

# Deep agent for continuous environments or discrete environments with a large state space
class DeepAgent(Agent):
    
    def __init__(self, env, weight_size) -> None:
        super().__init__(env)
        
        # self.w = torch.rand(weight_size)
    
    

class DiscreteQLearningAgent(Agent):
    
    def __init__(self, env) -> None:
        super().__init__(env)
        
class DiscreteDestributionalMLMCRobustAgent(Agent):
    
    def __init__(self, env) -> None:
        super().__init__(env)
        
        self.gamma = 0.5
        self.delta = 1
        self.lr = lambda t : 1/(1+(1-self.gamma)*(t-1))
        self.e = 0.5
        
        self.t = 0
        
    def next(self) -> bool:
        
        def newtonMethod(f, f_prime, N, r, x0, epsilon=0.01):
            x = x0
            while abs(f_prime(N, r, x)) > epsilon:
                x = x + 0.1*f_prime(N, r, x)
                if(x < 0):
                    x = 0.1
                    if(f_prime(N, r, x) < 0):
                        break
            return x
        
        def f_Delta_r(N, samples):

            temp_1 = lambda N, r, alpha : -alpha*np.log((1/(2**(N+1)))*sum([np.exp(-r[i][1]/(alpha + 1e-2)) for i in range(int(2**(N+1)))]) + 1e-2) - alpha*self.delta
            temp_2 = lambda N, r, alpha : -alpha*np.log((1/(2**(N)))*sum([np.exp(-r[2*i][1]/(alpha + 1e-2)) for i in range(int(2**(N)))]) + 1e-2) - alpha*self.delta
            temp_3 = lambda N, r, alpha : -alpha*np.log((1/(2**(N)))*sum([np.exp(-r[2*i-1][1]/(alpha + 1e-2)) for i in range(int(2**(N)))]) + 1e-2) - alpha*self.delta

            # diff_temp_1 = lambda N, r, alpha : - \
            #     np.log(sum([np.exp(-r[i][1]/(alpha + 1e-2)) for i in range(int(2**(N+1)))])/(2**(N+1))) - \
            #         (alpha*((sum([(r[i][1]*np.exp(-r[i][1]/(alpha + 1e-2))) / (alpha**2 + 1e-2) for i in range(int(2**(N+1)))])))) / \
            #             (sum([np.exp(-r[i][1]/(alpha + 1e-2)) for i in range(int(2**(N+1)))])) - self.delta
            
            # diff_temp_2 = lambda N, r, alpha : - \
            #     np.log(sum([np.exp(-r[2*i][1]/(alpha + 1e-2)) for i in range(int(2**(N)))])/(2**(N))) - \
            #         (alpha*((sum([(r[2*i][1]*np.exp(-r[2*i][1]/(alpha + 1e-2))) / (alpha**2 + 1e-2) for i in range(int(2**(N)))])))) / \
            #             (sum([np.exp(-r[2*i][1]/(alpha + 1e-2)) for i in range(int(2**(N)))])) - self.delta
                        
            # diff_temp_3 = lambda N, r, alpha : - \
            #     np.log(sum([np.exp(-r[2*i-1][1]/(alpha + 1e-2)) for i in range(int(2**(N)))])/(2**(N))) - \
            #         (alpha*((sum([(r[2*i-1][1]*np.exp(-r[2*i-1][1]/(alpha + 1e-2))) / (alpha**2 + 1e-2) for i in range(int(2**(N)))])))) / \
            #             (sum([np.exp(-r[2*i-1][1]/(alpha + 1e-2)) for i in range(int(2**(N)))])) - self.delta
            
            # alpha1 = newtonMethod(temp_1, diff_temp_1, N, samples, 1)
            # alpha2 = newtonMethod(temp_2, diff_temp_2, N, samples, 1)
            # alpha3 = newtonMethod(temp_3, diff_temp_3, N, samples, 1)
            
            # Delta_r = temp_1(N, samples, alpha1) - 1/2*temp_2(N, samples, alpha2) - 1/2*temp_3(N, samples, alpha3)
            
            Delta_r = max([temp_1(N,samples, a_) for a_ in np.arange(0.1, 10, 0.01)])
            Delta_r -= 1/2*max([temp_2(N,samples, a_) for a_ in np.arange(0.1, 10, 0.01)])
            Delta_r -= 1/2*max([temp_3(N,samples, a_) for a_ in np.arange(0.1, 10, 0.01)])
            
            return Delta_r
        
        def f_Delta_q(N, samples):
            
            temp_1 = lambda N, s_, beta : -beta*np.log((1/(2**(N+1))) *
                                                       sum([np.exp(-max([self.q[(s_[i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[i][0])])) for i in range(int(2**(N+1)))]) + 1e-2) - beta*self.delta
            temp_2 = lambda N, s_, beta : -beta*np.log((1/(2**(N))) *
                                                       sum([np.exp(-max([self.q[(s_[2*i][0], b)]/(beta + 1e-2) for b in self.env.A(s_[2*i][0])])) for i in range(int(2**(N)))]) + 1e-2) - beta*self.delta
            temp_3 = lambda N, s_, beta : -beta*np.log((1/(2**(N))) *
                                                       sum([np.exp(-max([self.q[(s_[2*i-1][0], b)]/(beta + 1e-2) for b in self.env.A(s_[2*i-1][0])])) for i in range(int(2**(N)))]) + 1e-2) - beta*self.delta
            
            # diff_temp_1 = lambda N, s_, beta : - \
            #     np.log(sum([np.exp(-max([self.q[(s_[i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[i][0])])/(beta + 1e-2)) for i in range(int(2**(N+1)))])/(2**(N+1))) - \
            #         (beta*((sum([(max([self.q[(s_[i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[i][0])])*np.exp(-max([self.q[(s_[i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[i][0])])/(beta + 1e-2))) / (beta**2 + 1e-2) for i in range(int(2**(N+1)))])))) / \
            #             (sum([np.exp(-max([self.q[(s_[i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[i][0])])/(beta + 1e-2)) for i in range(int(2**(N+1)))])) - self.delta
            
            # diff_temp_2 = lambda N, s_, beta : - \
            #     np.log(sum([np.exp(-max([self.q[(s_[2*i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[2*i][0])])/(beta + 1e-2)) for i in range(int(2**(N)))])/(2**(N))) - \
            #         (beta*((sum([(max([self.q[(s_[2*i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[2*i][0])])*np.exp(-max([self.q[(s_[2*i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[2*i][0])])/(beta + 1e-2))) / (beta**2 + 1e-2) for i in range(int(2**(N)))])))) / \
            #             (sum([np.exp(-max([self.q[(s_[2*i][0], b)] / (beta + 1e-2) for b in self.env.A(s_[2*i][0])])/(beta + 1e-2)) for i in range(int(2**(N)))])) - self.delta
                        
            # diff_temp_3 = lambda N, s_, beta : - \
            #     np.log(sum([np.exp(-max([self.q[(s_[2*i-1][0], b)] / (beta + 1e-2) for b in self.env.A(s_[2*i-1][0])])/(beta + 1e-2)) for i in range(int(2**(N)))])/(2**(N))) - \
            #         (beta*((sum([(max([self.q[(s_[2*i-1][0], b)] / (beta + 1e-2) for b in self.env.A(s_[2*i-1][0])])*np.exp(-max([self.q[(s_[2*i-1][0], b)] / (beta + 1e-2) for b in self.env.A(s_[2*i-1][0])])/(beta + 1e-2))) / (beta**2 + 1e-2) for i in range(int(2**(N)))])))) / \
            #             (sum([np.exp(-max([self.q[(s_[2*i-1][0], b)] / (beta + 1e-2) for b in self.env.A(s_[2*i-1][0])])/(beta + 1e-2)) for i in range(int(2**(N)))])) - self.delta
            
            # alpha1 = newtonMethod(temp_1, diff_temp_1, N, samples, 1)
            # alpha2 = newtonMethod(temp_2, diff_temp_2, N, samples, 1)
            # alpha3 = newtonMethod(temp_3, diff_temp_3, N, samples, 1)
            
            #Delta_q = temp_1(N, samples, alpha1) - 1/2*temp_2(N, samples, alpha2) - 1/2*temp_3(N, samples, alpha3)
            
            Delta_q = max([temp_1(N,samples, a_) for a_ in np.arange(0.1, 10, 0.01)])
            Delta_q -= 1/2*max([temp_2(N,samples, a_) for a_ in np.arange(0.1, 10, 0.01)])
            Delta_q -= 1/2*max([temp_3(N,samples, a_) for a_ in np.arange(0.1, 10, 0.01)])
            
            return Delta_q
        
        
        self.t += 1
        alpha_t = self.lr(self.t)
        q_ = defaultdict(lambda : 0)
        
        for state in self.env.get_states():
            actions = self.env.A(state)
            for action in actions:
                p = lambda n : self.e*(1-self.e)**(n)
                cp = lambda n : 1/p(n)
                N = cp(random.random())
                samples = [self.env.step(state, action) for _ in range(int(2**(N+1)))]

                Delta_r = f_Delta_r(N, samples)
                Delta_q = f_Delta_q(N, samples)
                R_rob = samples[0][1] + Delta_r/p(N)
                T_rob = max([self.q[(samples[0][0], b)] for b in self.env.A(samples[0][0])]) + Delta_q/p(N)
                T_rob_e = R_rob + self.gamma*T_rob
                
                if(type((1-alpha_t)*self.q[(state, action)] + alpha_t*T_rob_e) != np.float64):
                    print("Error")
                
                q_[(state,action)] = (1-alpha_t)*self.q[(state, action)] + alpha_t*T_rob_e
        self.q = q_
                
        return True

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
    
    