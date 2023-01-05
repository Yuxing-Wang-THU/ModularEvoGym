
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from typing import Dict, Optional, List
from modularevogym import *

import random
import math
import pkg_resources
import numpy as np
import os

import copy 

class EvoGymBase(gym.Env):
    """
    Base class for all Evolution Gym environments.

    Args:
        world (EvoWorld): object specifying the voxel layout of the environment.
    """
    def __init__(self, world: EvoWorld) -> None:

        # sim
        self._sim = EvoSim(self.world)
        self._default_viewer = EvoViewer(self._sim)

    def step(self, action: Dict[str, np.ndarray]) -> bool:
        """
        Step the environment by running physcis computations.

        Args:
            action (Dict[str, np.ndarray]): dictionary mapping robot names to actions. Actions are `(n,)` arrays, where `n` is the number of actuators in the target robot.
        
        Returns:
            bool: whether or not the simulation has reached an unstable state and cannot be recovered (`True` = unstable).
        """
        #step
        for robot_name, a in action.items():
            a = np.clip(a, 0.6, 1.6)
            a[abs(a) < 1e-8] = 0
            self._sim.set_action(robot_name, a)
        done = self._sim.step()

        return done

    def reset(self,) -> None:
        """
        Reset the simulation to the initial state.
        """
        self._sim.reset()

    @property
    def sim(self,) -> EvoSim:
        """
        Returns the environment's simulation.

        Returns:
            EvoSim: simulation object to return.
        """
        return self._sim

    @property
    def default_viewer(self,) -> EvoViewer:
        """
        Returns the environment's default viewer.

        Returns:
            EvoSim: viewer object to return.
        """
        return self._default_viewer
    
    def render(self,
               mode: str ='screen',
               verbose: bool = False,
               hide_background: bool = False,
               hide_grid: bool = False,
               hide_edges: bool = False,
               hide_voxels: bool = False) -> Optional[np.ndarray]:
        """
        Render the simulation.

        Args:
            mode (str): values of 'screen' and 'human' will render to a debug window. If set to 'img' will return an image array.
            verbose (bool): whether or not to print the rendering speed (rps) every second.
            hide_background (bool): whether or not to render the cream-colored background. If shut off background will be white.
            hide_grid (bool): whether or not to render the grid.
            hide_edges (bool): whether or not to render edges around all objects.
            hide_voxels (bool): whether or not to render voxels.

        Returns:
            Optional[np.ndarray]: if `mode` is set to `img`, will return an image array.
        """
        return self.default_viewer.render(mode, verbose, hide_background, hide_grid, hide_edges, hide_voxels)

    def close(self) -> None:
        """
        Close the simulation.
        """
        self.default_viewer.hide_debug_window() 

    def get_actuator_indices(self, robot_name: str) -> np.ndarray:
        """
        Returns the voxel indices a target robot's actuators in the environment's simulation.

        Args:
            robot_name (str): name of robot.
        
        Returns:
            np.ndarray: `(n,)` array of actuator indices, where `n` is the number of actuators.
        """
        return self._sim.get_actuator_indices(robot_name)

    def get_dim_action_space(self, robot_name: str) -> int:
        """
        Returns the number of actuators for a target robot in the environment's simulation.

        Args:
            robot_name (str): name of robot.
        
        Returns:
            int: number of actuators.
        """
        return self._sim.get_dim_action_space(robot_name)

    def get_time(self, ) -> int:
        """
        Returns the current time as defined in the environment's simulator. Time starts at `0` and is incremented each time the environment steps. Time resets to `0` when the environment is reset.

        Returns:
            int: the current time.
        """
        return self._sim.get_time()

    def pos_at_time(self, time: int) -> np.ndarray:
        """
        Returns positions of all point-masses in the environment's simulation at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of points in the environment's simulation.
        """
        return self._sim.pos_at_time(time)

    def vel_at_time(self, time: int) -> np.ndarray:
        """
        Returns velocities of all point-masses in the environment's simulation at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of points in the environment's simulation.
        """
        return self._sim.vel_at_time(time)
        
    def object_pos_at_time(self, time: int, object_name: str) -> np.ndarray:
        """
        Returns positions of all point-masses in a target object at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of point-masses in the target object.
        """
        return self._sim.object_pos_at_time(time, object_name)

    def object_vel_at_time(self, time: int, object_name: str) -> np.ndarray:
        """
        Returns velocities of all point-masses in a target object at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of point-masses in the target object.
        """
        return self._sim.object_vel_at_time(time, object_name)

    def object_orientation_at_time(self, time: int, object_name: str) -> float:
        """
        Returns an estimate of the orientation of an object at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurement.
            object_name (str): name of object
        
        Returns:
            float: orientation with respect to x-axis in radians (increasing counter-clockwise) from the range [0, 2π].
        """
        return self._sim.object_orientation_at_time(time, object_name)  

    def get_pos_com_obs(self, object_name: str) -> np.ndarray:
        """
        Observation helper-function. Computes the position of the center of mass of a target object by averaging the positions of the object's point masses.

        Args:
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2,)` array of the position of the center of mass.
        """
        object_points_pos = self._sim.object_pos_at_time(self.get_time(), object_name)
        object_pos_com = np.mean(object_points_pos, axis=1)
        return np.array([object_pos_com[0], object_pos_com[1]])

    def get_vel_com_obs(self, object_name: str) -> np.ndarray:
        """
        Observation helper-function. Computes the velocity of the center of mass of a target object by averaging the velocities of the object's point masses.

        Args:
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2,)` array of the velocity of the center of mass.
        """
        object_points_vel = self._sim.object_vel_at_time(self.get_time(), object_name)
        object_vel_com = np.mean(object_points_vel, axis=1)
        return np.array([object_vel_com[0], object_vel_com[1]])

    def get_relative_pos_obs(self, object_name: str):
        """
        Observation helper-function. Computes the positions of a target object's point masses relative to their center of mass.

        Args:
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2n,)` array of positions, where `n` is the number of point masses.
        """
        object_points_pos = self._sim.object_pos_at_time(self.get_time(), object_name)
        object_pos_com = np.mean(object_points_pos, axis=1)
        # return (object_points_pos-np.array([object_pos_com]).T).flatten()
        return (object_points_pos-np.array([object_pos_com]).T)

    def get_ort_obs(self, object_name: str):
        """
        Observation helper-function. Returns the orientation of a target object.

        Args:
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(1,)` array of the object's orientation.
        """
        return np.array([self.object_orientation_at_time(self.get_time(), object_name)])

    def get_floor_obs(
        self, 
        object_name: str, 
        terrain_list: List[str], 
        sight_dist: int, 
        sight_range: float = 5) -> np.ndarray:
        """
        Observation helper-function. Computes an observation describing the shape of the terrain below the target object. Specifically, for each voxel to the left and right of the target object's center of mass (along with the voxel containing the center of mass), the following observation is computed: min(y-distance in voxels to the nearest terrain object below the target object's center of mass, `sight_range`). Results are returned in a 1D numpy array.

        Args:
            object_name (str): name of target object.
            terrain_list (List[str]): names of objects to be considered terrain in the computation.
            sight_dist (int): number of voxels to the left and right of the target object's center of mass for which an observation should be returned.
            sight_range (float): the max number of voxels below the object that can be seen. (default = 5)
        
        Returns:
            np.ndarray: `(2 * sight_range + 1, )` array of distance observations.
        """
        
        object_points_pos = self._sim.object_pos_at_time(self.get_time(), object_name)
        object_pos_com = np.mean(object_points_pos, axis=1)
        
        if len(terrain_list) == 0:
            return None

        terrain_pos = self._sim.object_pos_at_time(self.get_time(), terrain_list[0])
        for i in range(1, len(terrain_list)):
            terrain_pos = np.concatenate((terrain_pos, self._sim.object_pos_at_time(self.get_time(), terrain_list[i])), axis = 1)

        right_mask = terrain_pos[0, :] > (object_pos_com[0] - (sight_dist+0.5))
        terrain_pos = terrain_pos[:, right_mask]

        left_mask = terrain_pos[0, :] < (object_pos_com[0] + (sight_dist+0.5))
        terrain_pos = terrain_pos[:, left_mask]

        bot_mask = terrain_pos[1, :] < (object_pos_com[1])
        terrain_pos = terrain_pos[:, bot_mask]

        elevations = np.zeros((sight_dist*2+1)) - sight_range + object_pos_com[1]
        for i in range(-sight_dist, sight_dist+1):
            less_than_mask = terrain_pos[0, :] > (object_pos_com[0] + (i-0.5))
            greater_than_mask = terrain_pos[0, :] < (object_pos_com[0] + (i+0.5))
            try:
                max_elevation = np.max(terrain_pos[1, (less_than_mask & greater_than_mask)])
                elevations[i+sight_dist] = max_elevation
            except:
                pass

        elevations = object_pos_com[1] - elevations
        elevations = np.clip(elevations, 0, sight_range)

        return elevations

    def get_ceil_obs(
        self, 
        object_name: str, 
        terrain_list: List[str], 
        sight_dist: int, 
        sight_range: float = 5) -> np.ndarray:
        """
        Observation helper-function. Computes an observation describing the shape of the terrain above the target object. Specifically, for each voxel to the left and right of the target object's center of mass (along with the voxel containing the center of mass), the following observation is computed: min(y-distance in voxels to the nearest terrain object above the target object's center of mass, `sight_range`). Results are returned in a 1D numpy array.

        Args:
            object_name (str): name of target object.
            terrain_list (List[str]): names of objects to be considered terrain in the computation.
            sight_dist (int): number of voxels to the left and right of the target object's center of mass for which an observation should be returned.
            sight_range (float): the max number of voxels above the object that can be seen. (default = 5)
        
        Returns:
            np.ndarray: `(2 * sight_range + 1, )` array of distance observations.
        """
        object_points_pos = self._sim.object_pos_at_time(self.get_time(), object_name)
        object_pos_com = np.mean(object_points_pos, axis=1)
        
        if len(terrain_list) == 0:
            return None

        terrain_pos = self._sim.object_pos_at_time(self.get_time(), terrain_list[0])
        for i in range(1, len(terrain_list)):
            terrain_pos = np.concatenate((terrain_pos, self._sim.object_pos_at_time(self.get_time(), terrain_list[i])), axis = 1)

        right_mask = terrain_pos[0, :] > (object_pos_com[0] - (sight_dist+0.5))
        terrain_pos = terrain_pos[:, right_mask]

        left_mask = terrain_pos[0, :] < (object_pos_com[0] + (sight_dist+0.5))
        terrain_pos = terrain_pos[:, left_mask]

        bot_mask = terrain_pos[1, :] > (object_pos_com[1])
        terrain_pos = terrain_pos[:, bot_mask]

        elevations = np.zeros((sight_dist*2+1)) + sight_range + object_pos_com[1]
        for i in range(-sight_dist, sight_dist+1):
            less_than_mask = terrain_pos[0, :] > (object_pos_com[0] + (i-0.5))
            greater_than_mask = terrain_pos[0, :] < (object_pos_com[0] + (i+0.5))
            try:
                max_elevation = np.min(terrain_pos[1, (less_than_mask & greater_than_mask)])
                elevations[i+sight_dist] = max_elevation
            except:
                pass

        elevations =  elevations - object_pos_com[1]
        elevations = np.clip(elevations, 0, sight_range)

        return elevations

class BenchmarkBase(EvoGymBase):

    DATA_PATH = pkg_resources.resource_filename('modularevogym.envs', os.path.join('sim_files'))
    VOXEL_SIZE = 0.1

    def __init__(self, world):

        EvoGymBase.__init__(self, world)
        self.default_viewer.track_objects('robot')

    def step(self, action):

        action_copy = {}

        for robot_name, a in action.items():
            action_copy[robot_name] = a + 1

        return super().step(action_copy)
    
    def pos_at_time(self, time):
        return super().pos_at_time(time)*self.VOXEL_SIZE

    def vel_at_time(self, time):
        return super().vel_at_time(time)*self.VOXEL_SIZE
        
    def object_pos_at_time(self, time, object_name):
        return super().object_pos_at_time(time, object_name)*self.VOXEL_SIZE

    def object_vel_at_time(self, time, object_name):
        return super().object_vel_at_time(time, object_name)*self.VOXEL_SIZE

    def get_pos_com_obs(self, object_name):
        return super().get_pos_com_obs(object_name)*self.VOXEL_SIZE

    def get_vel_com_obs(self, object_name):
        temp = super().get_vel_com_obs(object_name)*self.VOXEL_SIZE
        # print(f'child says super vel obs: {super().get_vel_com_obs(object_name)}\n')
        # print(f'vel obs: {temp}\n\n')
        return temp

    def get_relative_pos_obs(self, object_name):
        return super().get_relative_pos_obs(object_name)*self.VOXEL_SIZE

    def get_floor_obs(self, object_name, terrain_list, sight_dist, sight_range = 5):
        return super().get_floor_obs(object_name, terrain_list, sight_dist, sight_range)*self.VOXEL_SIZE

    def get_ceil_obs(self, object_name, terrain_list, sight_dist, sight_range = 5):
        return super().get_ceil_obs(object_name, terrain_list, sight_dist, sight_range)*self.VOXEL_SIZE

    # Functions for ModularEvogym 
    # Author: Yuxing Wang Date:01/01/2023
    # ----------------------------------------------------------------------------------------------- #
    def set_modular_attributes(self, body=None, mode="normal", nca_setting=None, init_nca_design=None):
        self.mode = mode
        self.stage = 'act'
        self.nca_setting = nca_setting
        if self.mode == "modular":
            # NCA
            if nca_setting is not None:
                super().reset()
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

    def get_modular_obs(self):
        obs = {}
        origin_ob=self.get_relative_pos_obs("robot")
        obs["modular"]= self.modular_ob_wrapper_padding(origin_ob=origin_ob,index_array=self.obs_index_array,body=self.body)
        obs["other"] = self.get_other_obs()
        obs["act_mask"] = self.act_mask
        obs["obs_mask"] = self.obs_mask
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
   
    # design functions
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
   
    def update(self, position, json_name, body, connections):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, json_name))
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
        if self.json_name in ['Walker-v0.json', 'Climber-v0.json','Climber-v1.json']:
            return self.get_vel_com_obs("robot")
        elif self.json_name in ['BridgeWalker-v0.json']: 
            return np.concatenate((self.get_vel_com_obs("robot"), self.get_ort_obs("robot"))) 
        elif self.json_name in ['CaveCrawler-v0.json']:
            return np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_floor_obs("robot", ["terrain"], self.sight_dist),
            self.get_ceil_obs("robot", ["terrain"], self.sight_dist),
            ))
        elif self.json_name in ['Balancer-v0.json','Balancer-v1.json',"Flipper-v0.json"]:
            return np.array([self.object_orientation_at_time(self.get_time(), "robot")])
        elif self.json_name in ['ShapeChange.json']:
            return np.array([0.0, 0.0])
        elif self.json_name in ['Climber-v2.json']:
            return np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_ceil_obs("robot", ["pipe"], self.sight_dist),
            ))
        elif self.json_name in ['Jumper-v0.json']:
            return np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        elif self.json_name in ['Carrier-v0.json','Carrier-v1.json','Pusher-v0.json',
                                'Pusher-v1.json','Thrower-v0.json','Lifter-v0.json']:
            # collect post step information
            robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
            robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
            package_pos_final = self.object_pos_at_time(self.get_time(), "package")
            package_vel_final = self.object_vel_at_time(self.get_time(), "package")
            # observation
            obs = self.get_obs_mani(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)

            if self.json_name == 'Lifter-v0.json':
                obs = np.concatenate((obs,self.get_ort_obs("package"),))

            return obs
        
        elif self.json_name in ['Catcher-v0.json']:
            # collect post step information
            robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
            package_pos_final = self.object_pos_at_time(self.get_time(), "package")

            # observation
            obs = self.get_obs_catch(robot_pos_final, package_pos_final)
            obs = np.concatenate((
                obs,
                self.get_vel_com_obs("robot"),
                self.get_vel_com_obs("package"),
                self.get_ort_obs("package"),
            ))
            return obs
        
        elif self.json_name in ['BeamToppler-v0.json', 'BeamSlider-v0.json']:
            # collect post step information
            robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
            beam_pos_final = self.object_pos_at_time(self.get_time(), "beam")

            # observation
            obs = self.get_obs_topple(robot_pos_final, beam_pos_final)
            obs = np.concatenate((
                obs,
                self.get_vel_com_obs("robot"),
                self.get_vel_com_obs("beam"),
                self.get_ort_obs("beam"),
            ))
            return obs

        elif self.json_name in ['UpStepper-v0.json','DownStepper-v0.json','ObstacleTraverser-v0.json',
                                'ObstacleTraverser-v1.json']:
            robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")
            # observation
            obs = np.concatenate((
                self.get_vel_com_obs("robot"),
                np.array([robot_ort_final]),
                self.get_floor_obs("robot", ["ground"], self.sight_dist),
                ))
            return obs
        
        elif self.json_name in ['Hurdler-v0.json']:
            return np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        elif self.json_name in ['PlatformJumper-v0.json']:
            robot_ort = self.object_orientation_at_time(self.get_time(), "robot")
            return np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))
        elif self.json_name in ['GapJumper-v0.json','Traverser-v0.json']:    
            obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))
        else:
            raise ValueError("Env Error")  
    
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
            # print("design done, new structure: ", self.real_design)
            return act_ob, 0.0, False, {'design_success': 'Done', 'stage': 'design','real_design':self.real_design}
        
        reward = 0.0
        done = False
        return ob, reward, done, {'design_success': True, 'stage': 'design'}
    
    def modular_obs(self):
        obs = {}
        origin_ob=self.get_relative_pos_obs("robot")
        obs["modular"]= self.modular_ob_wrapper_padding(origin_ob=origin_ob,index_array=self.obs_index_array,body=self.body)
        obs["other"] = self.get_ort_obs("robot")
        obs["act_mask"] = self.act_mask
        obs["obs_mask"] = self.obs_mask

        if self.nca_setting is not None:
            obs['design'] = self.make_design_batch(self.act_stage_design)
            obs['stage'] = [1.0]
        return obs
    def nca_reset(self):
        self.stage = 'design'
        self.cur_t = 0
        self.cur_nca_design = self.init_nca_design
        ob = self.init_modular_obs
        ob['design'] = self.make_design_batch(self.cur_nca_design)
        ob['stage'] = [0.0]
        return ob 
    def modular_reset(self):
        super().reset()
        obs = {}
        origin_ob=self.get_relative_pos_obs("robot")
        # print(f"{self.rank} origin obs: ", origin_ob)
        obs["modular"]= self.modular_ob_wrapper_padding(origin_ob=origin_ob,index_array=self.obs_index_array,body=self.body)
        # print(f"{self.rank} modified obs: ", obs["modular"])
        obs["other"] = self.get_other_obs()
        obs["act_mask"] = self.act_mask
        obs["obs_mask"] = self.obs_mask
        return obs
    # ----------------------------------------------------------------------------------------------- #
