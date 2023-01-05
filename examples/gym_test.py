import gym
import modularevogym.envs
from modularevogym import sample_robot
import numpy as np


if __name__ == '__main__':
    mode = "normal"
    body, connections = sample_robot((3,3))
    print(body)
    env = gym.make('Walker-v0', body=body, mode=mode)
    obs = env.reset()
    print(obs)

    while True:
        if mode == 'modular':
            action = np.random.uniform(low=0.6, high=1.6, size=9)-1
        else:
            action = env.action_space.sample()-1
        ob, reward, done, info = env.step(action)
        print(reward)
        # env.render()
        if done:
            break

    env.close()
