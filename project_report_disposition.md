
* **Abstract**  
Abstract
* **Introduction**  
Introduce the theory of Robust RL.  
Outline the two ways of doing Robust RL (Pure robust and in context learning)
. Motivate our project.
* **Environment**  
General requirements about the environment
and how the transistion function needs to
be normally distributed.
    * **Ware house environment**  
    Make this short. Explain the Warehouse 
    environment. State space, action space ect.
    * **Sumo-PP**  
    Explain the Sumo-PP environment. 
    State space, action space
    ect. Include an image.
    * **Brokken Cliff Car**  
    Explain the Broken Cliff Car environment. 
    State space, action space ect. 
    Include an Image
* **Method**  
Light introduction to RL in general and Robust 
RL. Motivate the theory by discussing why RL has its 
*can't take it out of the lab* downsides.
    * **Reinforcement learning**  
        * **Standard RL**  
        Cover standard RL
        * **Robust RL**  
        Describe Robust learning af the difference from Standard RL
    * **Distributional Robust Q-learning**  
    Explain the Distributional Robust Q-learning paper. 
    Why is it relevant? What are the Upsides? What are 
    the downsides?
        * **Kull-back Leibler divergence**  
        Explaing the Kull-back Leibler divergence 
        and how it is used to prove convergence when
        changing $\delta$
        * **Making it numerical stable**  
        Show what went into making their method numerically
        stable.
    * **Continues Distributional Robust Q-leaning**  
    Explain our method. Include how we can move away from Monte-Carlo 
    estimation by calculating the expectation using the squared- and 
    normal distribution assumptions.
        * **Deriation of the update function**  
        Full derivation of the update function.  
        Include the derivative (only if we can derive it :/ )
        * **Cooler Replay Buffer**  
        Detailed explanation of the *cooler replay buffer* and how it works.  
        Include illustrations.
        * **Kull-back Leibler divergence**  
        Derive the kull-back leibler divergence for two normal distributions.
* **Results/Discussion**
    * **Distributional Robust Q-learning reproduction: Warehouse environment**  
    Reproduce the policies and the graphs given in the Distributional Robust Q-learning paper
    * **1D-case: Sumo-PP**  
    Investigate the distribution of the value- and transistion function.
    Show statistics about convergence and how it influenced changes by
    changing $\delta$.  
    Show other interesting findings.  
        * Discuss the results
    * **2D-case: Broken Cliff Car**
    Investigate the distribution of the value- and transistion function.
    Show statistics about convergence and how it influenced changes by
    changing $\delta$
    Show other interesting findings.  
        * Discuss the results
* **Conclusion**  
Conclude on our findings