# robust-rl
Robust reinforcement learning implementation for bachelor's project


* **Abstract**
* **Introduction**
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
    Explain 
        * **Deriation of the update function**
        * **Cooler Replay Buffer**
* **Results/Discussion**
    * **Distributional Robust Q-learning reproduction: Warehouse environment**
    * **1D-case: Sumo-PP**
    * **2D-case: Broken Cliff Car**
* **Conclusion**
