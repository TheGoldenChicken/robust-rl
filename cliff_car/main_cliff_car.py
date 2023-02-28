import rl.manager
import cliff_car


env = cliff_car.Env()
agent = cliff_car.PlayMode(env)

manager = rl.manager.Manager(env, agent, render = True)

manager.run()
