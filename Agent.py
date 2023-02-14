from collections import defaultdict
import Policy as policy
import random
import numpy as np

class Agent:
    
    def __init__(self, env) -> None:
        self.env = env
        
        self.v = defaultdict(lambda : 0)
        self.q = defaultdict(lambda : 0)
        
        self.visited_states = []
        self.previous_actions = []
        self.obtained_rewards = []
        
        self.state = env.reset()
        
    def next(self) -> bool:
        return True

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
        
        def f_Delta_r(N, samples):
            alpha_max = 10
            alpha_min = 0.1
            alpha_step = 0.1

            temp_1 = lambda N, r, alpha : -alpha*np.log((1/(2**(N+1)))*sum([np.exp(-r[i][1]/(alpha + 1e-10)) for i in range(int(2**(N+1)))]) + 1e-10) - alpha*self.delta
            temp_2 = lambda N, r, alpha : -alpha*np.log((1/(2**(N)))*sum([np.exp(-r[2*i][1]/(alpha + 1e-10)) for i in range(int(2**(N)))]) + 1e-10) - alpha*self.delta
            temp_3 = lambda N, r, alpha : -alpha*np.log((1/(2**(N)))*sum([np.exp(-r[2*i-1][1]/(alpha + 1e-10)) for i in range(int(2**(N)))]) + 1e-10) - alpha*self.delta

            Delta_r = max([temp_1(N, samples, alpha) for alpha in np.arange(alpha_min, alpha_max, alpha_step)]) - 1/2*max([temp_2(N, samples, alpha) for alpha in np.arange(alpha_min, alpha_max, alpha_step)]) - 1/2*max([temp_3(N, samples, alpha) for alpha in np.arange(alpha_min, alpha_max, alpha_step)])
            
            return Delta_r
        
        def f_Delta_q(N, samples):
            beta_max = 10
            beta_min = 0.1
            beta_step = 0.1
            
            
            temp_1 = lambda N, s_, beta : -beta*np.log((1/(2**(N+1)))*sum([np.exp(-max([self.q[(s_[i][0], b)]/(beta + 1e-10) for b in self.env.A(s_[i][0])])) for i in range(int(2**(N+1)))]) + 1e-10) - beta*self.delta
            temp_2 = lambda N, s_, beta : -beta*np.log((1/(2**(N)))*sum([np.exp(-max([self.q[(s_[2*i][0], b)]/(beta + 1e-10) for b in self.env.A(s_[2*i][0])])) for i in range(int(2**(N)))]) + 1e-10) - beta*self.delta
            temp_3 = lambda N, s_, beta : -beta*np.log((1/(2**(N)))*sum([np.exp(-max([self.q[(s_[2*i-1][0], b)]/(beta + 1e-10) for b in self.env.A(s_[2*i-1][0])])) for i in range(int(2**(N)))]) + 1e-10) - beta*self.delta
            
            Delta_q = max([temp_1(N, samples, beta) for beta in np.arange(beta_min, beta_max, beta_step)]) - 1/2*max([temp_2(N, samples, beta) for beta in np.arange(beta_min, beta_max, beta_step)]) - 1/2*max([temp_3(N, samples, beta) for beta in np.arange(beta_min, beta_max, beta_step)])
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
    
    