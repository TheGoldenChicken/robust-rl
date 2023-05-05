import rl.agent
import numpy as np
from collections import defaultdict
from scipy.stats import geom

class robust_distributional_agent(rl.agent.ShallowAgent):
    
    def __init__(self, env, gamma = 0.9, delta = 1, epsilon = 0.5, tol = 0.05):
        super().__init__(env)
        
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.tol = tol
        
        self.lr = lambda t : 1/(1+(1-self.gamma)*(t-1))
        self.t = 0
        
        self.total_samples = 0
        
        self.Q = defaultdict(lambda : 0)

    def f_stable(self, x, samples, K):
        """
        The function to be maximized.
        This version is more numerically stable.
        """

        sample_star = np.max(samples)
        N = len(samples)
        tmp1 = -x*(sample_star - np.log(N))# np.log(2**K))
        tmp2 = -x*np.logaddexp.reduce(-samples/x - sample_star)
        tmp3 = -x*self.delta

        return tmp1 + tmp2 + tmp3
    
    def f_prime_approx(self, x, samples, K, tol):
        """
        An approximation of the derivative of f_stable.
        This should be used instead of f_prime due to numerical instability
        """

        return (self.f_stable(x+tol, samples, K) - self.f_stable(x-tol, samples, K)) / (2*tol)

    def maximize(self, samples, K, tol = 1e-5):
        """
        Maximize f_stable with respect to x by using the derivative f_prime.
        Also note that f_prime is either monotonic decreasing or only has one maximum.
        Also note that x is always positive.
        """
        
        x_min = tol*2
        x_max = 10
        
        # If f_prime(tol) < 0 the function is monotonic decreasing.
        if self.f_prime_approx(x_min, samples, K, tol) < 0:
            return x_min

        while True:
            # If f_prime(10) > 0, adjust the maximum
            if self.f_prime_approx(x_max, samples, K, tol) > 0:
                x_max *= 2
            else: 
                break
        
        # Find the maximum using devide and conquer
        while True:
            x_mid = (x_min + x_max) / 2
            if x_max - x_min < tol:
                return x_mid
            f_prime_x_mid = self.f_prime_approx(x_mid, samples, K, tol)
            if f_prime_x_mid > 0 + tol:
                x_min = x_mid
            elif f_prime_x_mid < 0 - tol:
                x_max = x_mid
            else:
                return x_mid

    def fDelta_q(self, N, state_):
        """
        The Delta_q function.
        """
        # Term 1
        # s_ = state_[:2**(N+1)+1]
        s_ = state_[::2]
        Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])

        # Calculate the supremum of the first term
        sup_1 = self.f_stable(self.maximize(Q_max, N+1), Q_max, N+1)

        # Term 2
        # s_ = state_[1:(2**N)+1:2]
        s_ = state_[1::2]
        Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])

        # Calculate the supremum of the second term
        sup_2 = self.f_stable(self.maximize(Q_max, N), Q_max, N)

        # Term 3
        # s_ = state_[:(2**N)+1:2]
        s_ = state_[::2]
        Q_max = np.array([max([self.Q[s_i, b] for b in self.env.A(s_i)]) for s_i in s_])

        # Calculate the supremum of the third term
        sup_3 = self.f_stable(self.maximize(Q_max, N), Q_max, N)

        return sup_1 - (1/2)*sup_2 - (1/2)*sup_3

    def fDelta_r(self, N, reward):
        """
        The Delta_r function.
        """
        # Term 1
        # r = reward[:2**(N+1)+1]
        r = reward

        # Calculate the supremum of the first term
        sup_1 = self.f_stable(self.maximize(r, N+1), r, N+1)

        # Term 2
        # r = reward[1:(2**N)+1:2]
        r = reward[1::2]

        # Calculate the supremum of the second term
        sup_2 = self.f_stable(self.maximize(r, N), r, N)

        # Term 3
        # r = reward[:(2**N)+1:2]
        r = reward[::2]

        # Calculate the supremum of the third term
        sup_3 = self.f_stable(self.maximize(r, N), r, N)

        return sup_1 - (1/2)*sup_2 - (1/2)*sup_3

    def stop_rnd(self, e):
        N = 0 # Pretty sure paper is buggy.
        while True:
            if np.random.rand() < e:
                return N
            N = N + 1

    def stop_log_p(self, e, N):
        return np.log(e) + np.log(1-e) * N

    def next(self):
        self.t += 1
        alpha_t = self.lr(self.t)
        # Q_ = defaultdict(lambda : 0)
        
        total_inf_norm = 0
        for state in self.env.get_states():
            actions = self.env.A(state)
            for action in actions:
                # Shift the location -1 to make it e(1-e)^N instead of e(1-e)^(N+1)
                loc = -1
                # Get N
                N = geom.rvs(p = self.epsilon, loc = loc, size = 1)[0]
                # N = self.stop_rnd(self.epsilon)
                # p_N = self.epsilon * (1-self.epsilon)**N
                # p_N = np.exp(self.stop_log_p(self.epsilon, N))

                # Get 2**(N+1) samples from the environment
                samples = np.array([self.env.step(state, action) for _ in range(2**(N+1))])

                # Track the total number of samples to reproduce the results in the paper
                self.total_samples += 2**(N+1)
                
                # Calculate the Delta_q and Delta_r functions
                Delta_q = self.fDelta_q(N, samples[:,0])
                Delta_r = self.fDelta_r(N, samples[:,1])
                
                # Calculate the probability of N
                p_N = geom.pmf(N, p = self.epsilon, loc = loc)

                # Calculate the robust estimate
                R_rob = samples[0][1] + Delta_r/p_N
                T_rob = max([self.Q[(samples[0][0], b)] for b in self.env.A(samples[0][0])]) + Delta_q/p_N
                T_rob_e = R_rob + self.gamma*T_rob
                
                # Update the Q function
                new_Q = (1-alpha_t)*self.Q[state, action] + alpha_t*T_rob_e
                
                inf_norm = np.abs(new_Q - self.Q[state, action])
                if inf_norm > total_inf_norm:
                    total_inf_norm = inf_norm
                
                self.Q[state, action] = new_Q

        # # Check for convergence
        # Q_diffs = []
        # for key in self.Q.keys():
        #     Q_diffs.append(np.abs(self.Q[key]-Q_[key]))
        # distance = np.max(Q_diffs)
        
        # Print the convergence information
        if total_inf_norm < self.tol:
            print(">>> (CONVERGED) Diff Inf Norm:", total_inf_norm, "| Total Samples:", self.total_samples)
            return True
        elif(self.t%100 == 0):
            print(">>> Diff Inf Norm:", total_inf_norm)
        
        # # Update the old Q values with the new Q values
        # self.Q = Q_
        
        return False