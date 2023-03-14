import gym
import evogym.envs
from evogym import sample_robot
import numpy as np
from evogym.utils import MODULAR_ENV_NAMES

if __name__ == '__main__':
    # Setting
    mode = "modular"
    body_size = (5,5)
    
    for env_name in MODULAR_ENV_NAMES:
        print("MODULAR ENV TEST: ", env_name)
        body, connections = sample_robot(body_size)
        # ModularEvoGym is compatible with EvolutionGym, if you just want to use EvoGym
        env = gym.make(env_name, body=body)
        # If you want to use ModularEvoGym, add mode='modular' and env_id=env_name
        env = gym.make(env_name, body=body, mode='modular', env_id=env_name)
        obs = env.reset()
    
        # Just Test: Update the orignal env
        new_body, new_connections = sample_robot(body_size)
        env.update(body=new_body, connections=new_connections)
        obs = env.reset()

        # Rollout
        while True:
            if mode == 'modular':
                action = np.random.uniform(low=-1.0, high=1.0, size=body_size[0]*body_size[1])
            else:
                action = env.action_space.sample()
            ob, reward, done, info = env.step(action)
            # env.render()
            if done:
                break
        env.close()
        print("Done!")