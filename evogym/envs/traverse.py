from evogym.envs.base import EvoGymBase
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

class StairsBase(BenchmarkBase):
    
    def __init__(self, world):
        super().__init__(world)

    def get_reward(self, robot_pos_init, robot_pos_final):
        
        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        reward = (robot_com_pos_final[0] - robot_com_pos_init[0])
        return reward

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
        robot_ort = self.object_orientation_at_time(self.get_time(), "robot")
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))

        return obs


class StepsUp(StairsBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'UpStepper-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
        
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
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort_final]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        ####### Added by Yuxing Wang, Modular observation   
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        #error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        #check termination conditions
        com_pos = np.mean(robot_pos_final, axis=1)
        if com_pos[0] > (69)*self.VOXEL_SIZE:
            done = True
            reward += 2.0        
        if robot_ort_final > (math.pi/2 - math.pi/12) and robot_ort_final < (3*math.pi/2 + math.pi/12):
            if self.mode == "modular":
                pass
            else:
                done = True
                reward -= 3.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {'design_success': True, 'stage': 'act'}
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'UpStepper-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        # init sim
        StairsBase.__init__(self, self.world)
        EvoGymBase.reset(self)
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 
    
class StepsDown(StairsBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'DownStepper-v0.json'))
        self.world.add_from_array('robot', body, 1, 11, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
        
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
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort_final]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check termination conditions
        com_pos = np.mean(robot_pos_final, axis=1)
        if com_pos[0] > (74)*self.VOXEL_SIZE:
            done = True
            reward += 2.0        
        if robot_ort_final > math.pi/2 and robot_ort_final < 3*math.pi/2:
            done = True
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {'design_success': True, 'stage': 'act'}
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'DownStepper-v0.json'))
        self.world.add_from_array('robot', body, 1, 11, connections=connections)
        # init sim
        StairsBase.__init__(self, self.world)
        EvoGymBase.reset(self)
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 
    
class WalkingBumpy(StairsBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ObstacleTraverser-v0.json'))
        self.world.add_from_array('robot', body, 2, 1, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
        
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
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort_final]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        
        # check termination condition
        com_pos = np.mean(robot_pos_final, axis=1)
        if com_pos[0] > (79)*self.VOXEL_SIZE:
            done = True
            reward += 2.0 
        if robot_ort_final > math.pi/2 and robot_ort_final < 3*math.pi/2:
            done = True
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {'design_success': True, 'stage': 'act'}
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ObstacleTraverser-v0.json'))
        self.world.add_from_array('robot', body, 2, 1, connections=connections)
        # init sim
        StairsBase.__init__(self, self.world)
        EvoGymBase.reset(self)
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 
    
class WalkingBumpy2(StairsBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ObstacleTraverser-v1.json'))
        self.world.add_from_array('robot', body, 2, 4, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
        
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
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

         # check termination condition
        com_pos = np.mean(robot_pos_final, axis=1)
        if com_pos[0] > (59)*self.VOXEL_SIZE:
            done = True
            reward += 2.0 

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {'design_success': True, 'stage': 'act'}
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ObstacleTraverser-v1.json'))
        self.world.add_from_array('robot', body, 2, 4, connections=connections)
        # init sim
        StairsBase.__init__(self, self.world)
        EvoGymBase.reset(self)
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 

class VerticalBarrier(StairsBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):
        
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Hurdler-v0.json'))
        self.world.add_from_array('robot', body, 2, 4, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
        
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
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        if robot_ort_final > math.pi/2 and robot_ort_final < 3*math.pi/2:
            done = True
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {'design_success': True, 'stage': 'act'}
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Hurdler-v0.json'))
        self.world.add_from_array('robot', body, 2, 4, connections=connections)
        # init sim
        StairsBase.__init__(self, self.world)
        EvoGymBase.reset(self)
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 
    
class FloatingPlatform(StairsBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'PlatformJumper-v0.json'))
        self.world.add_from_array('robot', body, 1, 6, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)

        # terrain
        self.terrain_list = ["platform_1", "platform_2", "platform_3", "platform_4", "platform_5", "platform_6", "platform_7"]
        
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
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))
        
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check if y coordinate is below lowest platform
        com = np.mean(robot_pos_final, 1)
        if com[1] < 0.4: # force stop
            reward -= 3.0
            done = True

        if robot_ort_final > math.pi/2 and robot_ort_final < 3*math.pi/2:
            done = True
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

        EvoGymBase.reset(self)

        # observation
        robot_ort = self.object_orientation_at_time(self.get_time(), "robot")
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))

        return obs
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'PlatformJumper-v0.json'))
        self.world.add_from_array('robot', body, 1, 6, connections=connections)
        # init sim
        StairsBase.__init__(self, self.world)
        EvoGymBase.reset(self)
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 
    
class Gaps(StairsBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'GapJumper-v0.json'))
        self.world.add_from_array('robot', body, 1, 6, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)

        # terrain
        self.terrain_list = ["platform_1", "platform_2", "platform_3", "platform_4", "platform_5"]
        
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
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check if y coordinate is below lowest platform
        com = np.mean(robot_pos_final, 1)
        if com[1] < (4)*self.VOXEL_SIZE:
            reward -= 3.0
            done = True

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
        EvoGymBase.reset(self)

        # observation
        robot_ort = self.object_orientation_at_time(self.get_time(), "robot")
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))

        return obs
    
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'GapJumper-v0.json'))
        self.world.add_from_array('robot', body, 1, 6, connections=connections)
        # init sim
        StairsBase.__init__(self, self.world)

        EvoGymBase.reset(self)
        
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 
    
class BlockSoup(StairsBase):

    def __init__(self, body, connections=None, mode=None, nca_setting=None, init_nca_design=None, env_id=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Traverser-v0.json'))
        self.world.add_from_array('robot', body, 1, 9, connections=connections)

        self.blocks = [
            [20, 1, 3],
            [24, 1, 3],
            [27, 1, 2],
            [31, 1, 3],
            [34, 1, 3],
            [38, 1, 2],
            [41, 1, 3],
            [45, 1, 3],
            [21, 4, 3],
            [24, 4, 2],
            [26, 4, 1],
            [29, 4, 2],
            [31, 4, 3],
            [34, 4, 2],
            [36, 4, 2],
            [40, 4, 3],
            [43, 4, 2],
            [46, 4, 3],
        ]
        for i in range(20, 45, 3):
            self.blocks.append([i, 7, 2])

        count = 0
        self.terrain_list = ["ground"]
        for x, y, size in self.blocks:
            temp_obj = WorldObject.from_json(os.path.join(self.DATA_PATH, f'rigid_{size}x{size}.json'))
            temp_obj.set_pos(x, y)
            temp_obj.rename(f'block_{count}')
            self.world.add_object(temp_obj)
            self.terrain_list.append(f'block_{count}')
            count += 1

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
        
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
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))
        
        ####### Added by Yuxing Wang, Modular observation    
        if self.mode == "modular":
            obs = self.get_modular_obs()
            obs['stage'] = [1.0]
            if self.nca_setting is not None:
                obs['design'] = self.make_design_batch(self.act_stage_design)
        #################################################

        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        #error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check termination conditions
        com_pos = np.mean(robot_pos_final, axis=1)
        if com_pos[0] > (59)*self.VOXEL_SIZE:
            done = True
            reward += 2.0

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
        robot_ort = self.object_orientation_at_time(self.get_time(), "robot")
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))

        return obs
    ####### Added by Yuxing Wang, reload the world
    def update(self,body,connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Traverser-v0.json'))
        self.world.add_from_array('robot', body, 1, 9, connections=connections)

        self.blocks = [
            [20, 1, 3],
            [24, 1, 3],
            [27, 1, 2],
            [31, 1, 3],
            [34, 1, 3],
            [38, 1, 2],
            [41, 1, 3],
            [45, 1, 3],
            [21, 4, 3],
            [24, 4, 2],
            [26, 4, 1],
            [29, 4, 2],
            [31, 4, 3],
            [34, 4, 2],
            [36, 4, 2],
            [40, 4, 3],
            [43, 4, 2],
            [46, 4, 3],
        ]
        for i in range(20, 45, 3):
            self.blocks.append([i, 7, 2])

        count = 0
        self.terrain_list = ["ground"]
        for x, y, size in self.blocks:
            temp_obj = WorldObject.from_json(os.path.join(self.DATA_PATH, f'rigid_{size}x{size}.json'))
            temp_obj.set_pos(x, y)
            temp_obj.rename(f'block_{count}')
            self.world.add_object(temp_obj)
            self.terrain_list.append(f'block_{count}')
            count += 1

        # init sim
        StairsBase.__init__(self, self.world)

        EvoGymBase.reset(self)
        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.body = body
        self.voxel_num = self.body.size   
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)  
        return self.get_modular_obs() 