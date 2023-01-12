import gym
import evogym.envs
from evogym import sample_robot
from modular_envs.wrappers.modular_wrapper import modular_env
import numpy as np

if __name__ == '__main__':
    # Setting
    mode = "modular"
    body_size = (3,3)
    body, connections = sample_robot(body_size)
    
    # Env
    env = gym.make("Walker-v0", body=body)
    # If you want to use ModularEvoGym
    env = modular_env(env=env, body=body)
    obs = env.reset()

    # Rollout
    while True:
        if mode == 'modular':
            action = np.random.uniform(low=0.6, high=1.6, size=body_size[0]*body_size[1])-1
        else:
            action = env.action_space.sample()-1
        ob, reward, done, info = env.step(action)
        # env.render()
        if done:
            break
    env.close()