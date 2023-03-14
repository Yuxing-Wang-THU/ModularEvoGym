import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from evogym import *
from evogym.envs import BenchmarkBase

import random
import math
import numpy as np
import os

class Balance(BenchmarkBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Balancer-v0.json'))
        self.world.add_from_array('robot', body, 15, 3, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(1 + num_robot_points,), dtype=np.float)
        
        ####### Added by Yuxing Wang
        self.mode = mode
        self.env_id = env_id
        if self.mode == "modular":
            self.set_modular_attributes(body, nca_setting, init_nca_design)
        #################################################

    def get_obs(self, pos_final):
        com_final = np.mean(pos_final, 1)

        return np.array([
            17*self.VOXEL_SIZE - com_final[0],
            5.5*self.VOXEL_SIZE - com_final[1],
        ])

    def get_reward(self, pos_init, pos_final):
        com_init = np.mean(pos_init, 1)
        com_final = np.mean(pos_final, 1)
        
        reward = abs(17*self.VOXEL_SIZE - com_init[0]) - abs(17*self.VOXEL_SIZE - com_final[0])
        reward += (abs(5.0*self.VOXEL_SIZE - com_init[1]) - abs(5.0*self.VOXEL_SIZE - com_final[1]))
        return reward

    def step(self, action):
        ####### Added by Yuxing Wang
        if self.mode == "modular":
            if self.stage == 'design':
                return self.design_step(action)
            elif self.stage == 'act':
                action = action[~self.act_mask.astype(bool)] 
        #################################################

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        
        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        ort = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            np.array([ort]),
            self.get_relative_pos_obs("robot"),
            ))
        
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = self.get_reward(pos_1, pos_2)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        
        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {'design_success': True, 'stage': 'act'}

    def reset(self):
        ####### Added by Yuxing Wang
        if self.mode == "modular":
            if self.nca_setting is not None:
                return self.nca_reset()
            else:
                return self.modular_reset()
        #################################################

        super().reset()

        # observation
        obs = np.concatenate((
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        return obs
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Balancer-v0.json'))
        self.world.add_from_array('robot', body, 15, 3, connections=connections)
        # init sim
        BenchmarkBase.__init__(self, self.world)
        super().reset()
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 
    
        
class BalanceJump(BenchmarkBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Balancer-v1.json'))
        self.world.add_from_array('robot', body, 10, 1, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(1 + num_robot_points,), dtype=np.float)
        
        ####### Added by Yuxing Wang
        self.mode = mode
        self.env_id = env_id
        if self.mode == "modular":
            self.set_modular_attributes(body, nca_setting, init_nca_design)
        #################################################

    def get_obs(self, pos_final):
        com_final = np.mean(pos_final, 1)

        return np.array([
            17.5*self.VOXEL_SIZE - com_final[0],
            6*self.VOXEL_SIZE - com_final[1],
        ])

    def get_reward(self, pos_init, pos_final):
        com_init = np.mean(pos_init, 1)
        com_final = np.mean(pos_final, 1)
        
        reward = abs(17.5*self.VOXEL_SIZE - com_init[0]) - abs(17.5*self.VOXEL_SIZE - com_final[0])
        reward += (abs(6*self.VOXEL_SIZE - com_init[1]) - abs(6*self.VOXEL_SIZE - com_final[1]))

        return reward

    def step(self, action):
        ####### Added by Yuxing Wang
        if self.mode == "modular":
            if self.stage == 'design':
                return self.design_step(action)
            elif self.stage == 'act':
                action = action[~self.act_mask.astype(bool)] 
        #################################################

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        
        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        ort = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            np.array([ort]),
            self.get_relative_pos_obs("robot"),
            ))
        
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = self.get_reward(pos_1, pos_2)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        
        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {'design_success': True, 'stage': 'act'}

    def reset(self):
        ####### Added by Yuxing Wang
        if self.mode == "modular":
            if self.nca_setting is not None:
                return self.nca_reset()
            else:
                return self.modular_reset()
        #################################################

        super().reset()

        # observation
        obs = np.concatenate((
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        return obs
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Balancer-v1.json'))
        self.world.add_from_array('robot', body, 10, 1, connections=connections)
        # init sim
        BenchmarkBase.__init__(self, self.world)
        super().reset()
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 