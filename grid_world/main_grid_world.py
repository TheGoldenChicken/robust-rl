import rl.manager
import grid_world
from agent import first_visit_monte_carlo as fvmc


env = grid_world.Env()
policy = grid_world.PlayModePolicy(env)
agent = fvmc.FirstVisitMonteCarlo(env, policy)

manager = rl.manager.Manager(env, agent, render = True)

manager.run()
