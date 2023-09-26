import rl.manager
import grid_world
from agent import first_visit_monte_carlo as fvmc
from agent import every_visit_monte_carlo as evmc
from agent import td_zero
import rl.policy
import grid_world_levels

env = grid_world.Env(layout = grid_world_levels.player_only_1)
# policy = grid_world.PlayModePolicy(env)
policy = rl.policy.EpsilonGreedy(env, epsilon = 0.05, decay = 1)
# agent = fvmc.FirstVisitMonteCarlo(env, policy)
# agent = evmc.EveryVisitMonteCarlo(env, policy)
agent = td_zero.TDZero(env, policy)

manager = rl.manager.Manager(agent, render = True)

manager.run()
