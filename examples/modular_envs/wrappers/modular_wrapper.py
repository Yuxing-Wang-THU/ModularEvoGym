import numpy as np
import copy
from gym import spaces
from evogym import *
from evogym.envs import BenchmarkBase
import os

# ModularEvoGym Wrapper
class modular_env(BenchmarkBase):
    def __init__(self, env, body=None, mode="modular", nca_setting=None, init_nca_design=None) -> None:
        self.env = env
        self.env_id = self.env.spec.id
        self.init_position = (self.env.world.objects['robot'].pos.x, self.env.world.objects['robot'].pos.y)
        self.set_modular_attributes(body=body,mode=mode,nca_setting=nca_setting,init_nca_design=init_nca_design)
    
    # Step
    def step(self, action):
        if self.stage == 'design':
            ob, reward, done, info = self.design_step(action)
            return ob, reward, done, info
        if self.mode == "modular":
            action = action[~self.act_mask.astype(bool)] 
            _, reward, done, _ = self.env.step(action)
            obs = self.modular_obs()
        return obs, reward, done, {'design_success': True, 'stage': 'act'}
    
    # Reset
    def reset(self):
        if self.nca_setting is not None:
            return self.nca_reset()
        if self.mode == "modular":
            obs = self.modular_reset()
        return obs

    # Close
    def close(self):
        self.env.close()

    # Set attributes
    def set_modular_attributes(self, body=None, mode="modular", nca_setting=None, init_nca_design=None):
        self.mode = mode
        self.stage = 'act'
        self.nca_setting = nca_setting
        if self.mode == "modular":
            # NCA part
            if nca_setting is not None:
                self.env.reset()
                self.nca_setting = nca_setting
                self.design_steps = self.nca_setting.design_iterations 
                self.stage = 'design'
                self.cur_t = 0
                self.init_nca_design = copy.deepcopy(init_nca_design)
                self.cur_nca_design = copy.deepcopy(init_nca_design)
            else:
                self.nca_setting = None
                self.design_steps = None
                self.stage = 'act'

            self.body = body
            self.voxel_num = self.body.size
            self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
            self.act_mask = self.get_act_mask(self.body)
            self.obs_mask = self.get_obs_mask(self.body)
            self.init_modular_obs = self.get_modular_obs()
            self.observation_space = self.convert_obs_to_space(self.init_modular_obs)
            self.modular_state_dim = self.observation_space['modular'].shape[0] // self.voxel_num
            self.modular_action_dim = 1
            self.other_dim = self.observation_space['other'].shape[0]

    # Functions for creating modular state-action space
    def get_modular_obs(self):
        obs = {}
        origin_ob=self.env.get_relative_pos_obs_nof("robot")
        obs["modular"]= self.modular_ob_wrapper_padding(origin_ob=origin_ob,index_array=self.obs_index_array,body=self.body)
        obs["other"] = self.get_other_obs()
        obs["act_mask"] = self.act_mask
        obs["obs_mask"] = self.obs_mask
        obs['stage'] = np.array([1.0])
        return obs

    def transform_to_modular_obs(self, body):
        # This function is inspired by Mark Horton:
        # https://github.com/EvolutionGym/evogym/issues/6
        loc_idx = 0
        # index in obs vector where each corner will be stored. -1 if no value
        index_by_corner = np.zeros(tuple(body.shape) + (2, 2), dtype=int) - 1
        for y in range(body.shape[0]):
            for x in range(body.shape[1]):
                if body[y,x] != 0:
                    has_upper_neighbor = ((y-1) >= 0) and (body[y-1,x] != 0)
                    has_left_neighbor = ((x-1) >= 0) and (body[y,x-1] != 0)
                    has_right_neighbor = ((x+1) < body.shape[1]) and (body[y,x+1] != 0)
                    has_upper_right_neighbor = ((x+1) < body.shape[1]) and ((y-1) >= 0) and (body[y-1,x+1] != 0)

                    if has_upper_neighbor:
                        index_by_corner[y, x, 0, :] = index_by_corner[y - 1, x, 1, :]
                    if has_left_neighbor:
                        index_by_corner[y, x, :, 0] = index_by_corner[y, x - 1, :, 1]

                    if not (has_upper_neighbor or has_left_neighbor):
                        index_by_corner[y, x, 0, 0] = loc_idx
                        loc_idx += 1
                    if not has_upper_neighbor:
                        if has_right_neighbor and has_upper_right_neighbor:
                            index_by_corner[y, x, 0, 1] = index_by_corner[y-1, x+1, 1, 0]
                        else:
                            index_by_corner[y, x, 0, 1] = loc_idx
                            loc_idx += 1
                    if not has_left_neighbor:
                        index_by_corner[y, x, 1, 0] = loc_idx
                        loc_idx += 1

                    index_by_corner[y, x, 1, 1] = loc_idx
                    loc_idx += 1
        # Get index array
        index_array = self.modular_observation_array(index_by_corner)
        return index_array, loc_idx

    def modular_observation_array(self, index_by_corner):
        modular_ob_index = []
        for row in range(index_by_corner.shape[0]):
            for col in range(index_by_corner.shape[1]):
                block = index_by_corner[row][col]
                modular_ob_index.append(block.flatten())
        return np.array(modular_ob_index)

    def modular_ob_wrapper(self, origin_ob, index_array):
        modular_obs = []
        for index in index_array:
            if np.sum(index) == -4:
                modular_obs.append(np.zeros(8).astype(float))
            else:
                modular_obs.append(np.concatenate((origin_ob[0][index],origin_ob[1][index]),axis=0))
        return np.array(modular_obs).flatten()
    
    def modular_ob_wrapper_padding(self, origin_ob, index_array, body):
        modular_obs_tmp = []
        modular_obs = []
        obs_padding = []
        body = list(body.flatten())
        for index in index_array:
            if np.sum(index) == -4:
                modular_obs_tmp.append(np.zeros(8).astype(float))
            else:
                modular_obs_tmp.append(np.concatenate((origin_ob[0][index],origin_ob[1][index]),axis=0))
        
        # Add material information
        # ps: If you do not want to add this information, just comment out this for loop 
        for i in range(len(modular_obs_tmp)):
            modular_obs_tmp[i]=np.append(modular_obs_tmp[i],body[i])
            if body[i] == 0:
                obs_padding.append(modular_obs_tmp[i])
            else:
                modular_obs.append(modular_obs_tmp[i])

        for pad in obs_padding:
            modular_obs.append(pad)

        del modular_obs_tmp
        return np.array(modular_obs).flatten()
    
    def convert_obs_to_space(self, observation):
        from collections import OrderedDict
        import numpy as np
        from gym import spaces
        if isinstance(observation, dict):
            space = spaces.Dict(
                OrderedDict(
                    [
                        (key, self.convert_obs_to_space(value))
                        for key, value in observation.items()
                    ]
                )
            )
        elif isinstance(observation, np.ndarray):
            low = np.full(observation.shape, -100.0, dtype=np.float32)
            high = np.full(observation.shape, 100.0, dtype=np.float32)
            space = spaces.Box(low, high, dtype=observation.dtype)
        else:
            raise NotImplementedError(type(observation), observation)
        return space
    
    # Functions for creating state-action masks
    def get_act_mask(self, body):
        am = copy.deepcopy(body)
        actuator_masks = am.flatten()
        for i in range(len(actuator_masks)):
            if actuator_masks[i].astype(int) == 4 or actuator_masks[i].astype(int) == 3:
                actuator_masks[i]=0
            else:
                actuator_masks[i]=1
        return actuator_masks.astype(int)

    def get_obs_mask(self, body):
        bd = copy.deepcopy(body)
        bd_materials = bd.flatten()
        obs = []
        obs_padding = []
        for i in range(len(bd_materials)):
            if bd_materials[i].astype(int) == 0:
                obs_padding.append(1)
            else:
                obs.append(0)
        return np.append(obs, np.array(obs_padding))
   
    # NCA design functions
    def apply_design_action(self, output, robot_seed=None):
        robot_seed_tmp = copy.deepcopy(robot_seed)
        # Map out to cell state(0,1,2,3,4)
        counter = 0 
        for i in range(1,self.nca_setting.im_size-1):
            for e in range(1,self.nca_setting.im_size-1):
                # neighbor
                if self.nca_setting.number_neighbors == 9:
                    sum_cells = np.sum(robot_seed[0, i-1:i+2, e-1:e+2])
                elif self.nca_setting.number_neighbors == 5:
                    sum_cells = robot_seed[0, i-1, e]+robot_seed[0, i+1, e]+robot_seed[0, i, e+1]+robot_seed[0, i, e-1]+robot_seed[0,i,e]
                if sum_cells > 0: # If any of the surrounded cells or the cell itself is activate, modify the that position
                    idx = output[counter]
                    # Next cell states
                    robot_seed_tmp[0, i, e] =int(idx)
                counter += 1

        robot_seed = copy.deepcopy(robot_seed_tmp)

        real_robot = robot_seed[0,1:self.nca_setting.im_size-1,1:self.nca_setting.im_size-1]
        
        if is_connected(real_robot) and has_actuator(real_robot):
            success = True
        else:
            success = False
        return robot_seed, success

    def make_design_batch(self, state=None):
        robot_seed = state
        # Make batch
        batch_inputs = np.zeros((self.voxel_num, self.nca_setting.number_neighbors))
        counter = 0
        for i in range(1,self.nca_setting.im_size-1):
            for e in range(1,self.nca_setting.im_size-1):
                if self.nca_setting.number_neighbors == 9:
                    cell_input = robot_seed[0, i-1:i+2, e-1:e+2].flatten()
                elif self.nca_setting.number_neighbors == 5:
                    cell_input = np.concatenate([robot_seed[0, i-1, e],robot_seed[0, i+1, e],robot_seed[0, i, e+1],robot_seed[0, i, e-1],robot_seed[0,i,e] ])
                batch_inputs[counter] = cell_input
                counter += 1
        return batch_inputs
   
    def update(self, position, env_id, body, connections):
        # make world
        if "mizer" in self.env_id:
            self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ShapeChange.json'))
        else:
            self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, env_id +'.json'))
        self.world.add_from_array('robot', body, position[0], position[1], connections=connections)
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
    
    def design_step(self, action):
        
        # fake modular obs
        ob = self.init_modular_obs
        self.cur_t = self.cur_t + 1
        self.cur_nca_design, success = self.apply_design_action(output=action, robot_seed=self.cur_nca_design)
        ob['design'] = self.make_design_batch(self.cur_nca_design)
        ob['stage'] = [0.0]

        if not success:
            return ob, 0.0, True, {'design_success': False, 'stage': 'design'}

        if self.cur_t == self.design_steps:
            # if not success:
            #     return ob, 0.0, True, {'design_success': False, 'stage': 'design'} 
            self.stage = 'act'
            self.act_stage_design = self.cur_nca_design
            self.real_design = self.cur_nca_design[0,1:self.nca_setting.im_size-1,1:self.nca_setting.im_size-1]
            act_ob = self.update(body=self.real_design,position=self.init_position, json_name=self.json_name, connections=get_full_connectivity(self.real_design))
            act_ob['design'] = self.make_design_batch(self.act_stage_design)
            act_ob['stage'] = [1.0]
            return act_ob, 0.0, False, {'design_success': 'Done', 'stage': 'design','real_design':self.real_design}
        reward = 0.0
        done = False
        return ob, reward, done, {'design_success': True, 'stage': 'design'}
    
    def nca_reset(self):
        self.stage = 'design'
        self.cur_t = 0
        self.cur_nca_design = self.init_nca_design
        ob = self.init_modular_obs
        ob['design'] = self.make_design_batch(self.cur_nca_design)
        ob['stage'] = [0.0]
        return ob 
    
    # Obs functions
    def get_obs_catch(self, robot_pos_final, package_pos_final):
        
        robot_com_pos = np.mean(robot_pos_final, axis=1)
        package_com_pos = np.mean(package_pos_final, axis=1)

        obs = np.array([
            package_com_pos[0]-robot_com_pos[0], package_com_pos[1]-robot_com_pos[1],
        ])
        return obs

    def get_obs_mani(self, robot_pos_final, robot_vel_final, package_pos_final, package_vel_final):
        
        robot_com_pos = np.mean(robot_pos_final, axis=1)
        robot_com_vel = np.mean(robot_vel_final, axis=1)
        box_com_pos = np.mean(package_pos_final, axis=1)
        box_com_vel = np.mean(package_vel_final, axis=1)

        obs = np.array([
            robot_com_vel[0], robot_com_vel[1],
            box_com_pos[0]-robot_com_pos[0], box_com_pos[1]-robot_com_pos[1],
            box_com_vel[0], box_com_vel[1]
        ])
        return obs

    def get_obs_topple(self, robot_pos_final, beam_pos_final):

        beam_com_pos_final = np.mean(beam_pos_final, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        diff = beam_com_pos_final - robot_com_pos_final
        return np.array([diff[0], diff[1]])

    def get_other_obs(self):
        # Get task-related state
        if self.env_id in ['Walker-v0', 'Climber-v0','Climber-v1']:
            return self.env.get_vel_com_obs("robot")
        elif self.env_id in ['BridgeWalker-v0']: 
            return np.concatenate((self.env.get_vel_com_obs("robot"), self.env.get_ort_obs("robot"))) 
        elif self.env_id in ['CaveCrawler-v0']:
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_floor_obs("robot", ["terrain"], self.env.sight_dist),
            self.env.get_ceil_obs("robot", ["terrain"], self.env.sight_dist),
            ))
        elif self.env_id in ['Balancer-v0','Balancer-v1',"Flipper-v0"]:
            return np.array([self.env.object_orientation_at_time(self.env.get_time(), "robot")]) 
        elif "mizer" in self.env_id:
            return np.array([0.0, 0.0])
        elif self.env_id in ['Climber-v2']:
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_ort_obs("robot"),
            self.env.get_ceil_obs("robot", ["pipe"], self.env.sight_dist),
            ))
        elif self.env_id in ['Jumper-v0']:
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_floor_obs("robot", ["ground"], self.env.sight_dist),
            ))
        elif self.env_id in ['Carrier-v0','Carrier-v1','Pusher-v0','Pusher-v1','Thrower-v0','Lifter-v0']:
            # collect post step information
            robot_pos_final = self.env.object_pos_at_time(self.env.get_time(), "robot")
            robot_vel_final = self.env.object_vel_at_time(self.env.get_time(), "robot")
            package_pos_final = self.env.object_pos_at_time(self.env.get_time(), "package")
            package_vel_final = self.env.object_vel_at_time(self.env.get_time(), "package")
            # observation
            obs = self.get_obs_mani(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
            
            if self.env_id == 'Lifter-v0':
                obs = np.concatenate((obs,self.env.get_ort_obs("package"),))
            return obs  
        elif self.env_id in ['Catcher-v0']:
            # collect post step information
            robot_pos_final = self.env.object_pos_at_time(self.env.get_time(), "robot")
            package_pos_final = self.env.object_pos_at_time(self.env.get_time(), "package")

            # observation
            obs = self.get_obs_catch(robot_pos_final, package_pos_final)
            obs = np.concatenate((
                obs,
                self.env.get_vel_com_obs("robot"),
                self.env.get_vel_com_obs("package"),
                self.env.get_ort_obs("package"),
            ))
            return obs       
        elif self.env_id in ['BeamToppler-v0', 'BeamSlider-v0']:
            # collect post step information
            robot_pos_final = self.env.object_pos_at_time(self.env.get_time(), "robot")
            beam_pos_final = self.env.object_pos_at_time(self.env.get_time(), "beam")

            # observation
            obs = self.get_obs_topple(robot_pos_final, beam_pos_final)
            obs = np.concatenate((
                obs,
                self.env.get_vel_com_obs("robot"),
                self.env.get_vel_com_obs("beam"),
                self.env.get_ort_obs("beam"),
            ))
            return obs
        elif self.env_id in ['UpStepper-v0','DownStepper-v0','ObstacleTraverser-v0','ObstacleTraverser-v1']:
            robot_ort_final = self.env.object_orientation_at_time(self.env.get_time(), "robot")
            # observation
            obs = np.concatenate((
                self.env.get_vel_com_obs("robot"),
                np.array([robot_ort_final]),
                self.env.get_floor_obs("robot", ["ground"], self.env.sight_dist),
                ))
            return obs     
        elif self.env_id in ['Hurdler-v0']:
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_ort_obs("robot"),
            self.env.get_floor_obs("robot", ["ground"], self.env.sight_dist),
            ))
        elif self.env_id in ['PlatformJumper-v0']:
            robot_ort = self.env.object_orientation_at_time(self.env.get_time(), "robot")
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.env.get_floor_obs("robot", self.env.terrain_list, self.env.sight_dist),
            ))

        elif self.env_id in ['GapJumper-v0','Traverser-v0']:    
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_ort_obs("robot"),
            self.env.get_floor_obs("robot", self.env.terrain_list, self.env.sight_dist),
            ))

        else:
            raise ValueError("Env Error")  
               
    def modular_obs(self):
        obs = {}
        origin_ob=self.env.get_relative_pos_obs_nof("robot")
        obs["modular"]= self.modular_ob_wrapper_padding(origin_ob=origin_ob,index_array=self.obs_index_array,body=self.body)
        obs["other"] = self.get_other_obs()
        obs["act_mask"] = self.act_mask
        obs["obs_mask"] = self.obs_mask
        obs['stage'] = [1.0]
        if self.nca_setting is not None:
            obs['design'] = self.make_design_batch(self.act_stage_design)
        return obs

    def modular_reset(self):
        self.stage = 'act'
        self.env.reset()
        obs = {}
        origin_ob=self.env.get_relative_pos_obs_nof("robot")
        obs["modular"]= self.modular_ob_wrapper_padding(origin_ob=origin_ob,index_array=self.obs_index_array,body=self.body)
        obs["other"] = self.get_other_obs()
        obs["act_mask"] = self.act_mask
        obs["obs_mask"] = self.obs_mask
        obs['stage'] = [1.0]
        return obs
