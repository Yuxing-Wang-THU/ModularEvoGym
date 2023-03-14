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

class WalkingFlat(BenchmarkBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + num_robot_points,), dtype=np.float)
        
        ####### Added by Yuxing Wang
        self.mode = mode
        self.env_id = env_id
        if self.mode == "modular":
            self.set_modular_attributes(body, nca_setting, init_nca_design)
        #################################################

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

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
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
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
            
        # check goal met
        if com_2[0] > 99*self.VOXEL_SIZE:
            done = True
            reward += 1.0
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
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))
        return obs
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
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

class SoftBridge(BenchmarkBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'BridgeWalker-v0.json'))
        self.world.add_from_array('robot', body, 2, 5, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + 1 + num_robot_points,), dtype=np.float)
        
        ####### Added by Yuxing Wang
        self.mode = mode
        self.env_id = env_id
        if self.mode == "modular":
            self.set_modular_attributes(body, nca_setting, init_nca_design)
        #################################################

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

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
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
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check goal met
        if com_2[0] > (60)*self.VOXEL_SIZE:
            done = True
            reward += 1.0

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
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        return obs
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'BridgeWalker-v0.json'))
        self.world.add_from_array('robot', body, 2, 5, connections=connections)
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

class Duck(BenchmarkBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'CaveCrawler-v0.json'))
        self.world.add_from_array('robot', body, 1, 2, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + num_robot_points + 2*(self.sight_dist*2 +1),), dtype=np.float)
        
        ####### Added by Yuxing Wang
        self.mode = mode
        self.env_id = env_id
        if self.mode == "modular":
            self.set_modular_attributes(body, nca_setting, init_nca_design)
        #################################################

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

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["terrain"], self.sight_dist),
            self.get_ceil_obs("robot", ["terrain"], self.sight_dist),
            ))

        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check goal met
        if com_2[0] > (69)*self.VOXEL_SIZE:
            done = True
            reward += 1.0

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
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["terrain"], self.sight_dist),
            self.get_ceil_obs("robot", ["terrain"], self.sight_dist),
            ))

        return obs
        ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'CaveCrawler-v0.json'))
        self.world.add_from_array('robot', body, 1, 2, connections=connections)
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