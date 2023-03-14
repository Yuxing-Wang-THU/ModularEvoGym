import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
import gym
import evogym.envs
from modular_envs.wrappers.dummy_vec_env import DummyVecEnv
from modular_envs.wrappers.subproc_vec_env import SubprocVecEnv
from modular_envs.wrappers.vec_env import VecEnvWrapper
import time 
from collections import deque,defaultdict

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

def make_env(env_id, robot_structure, seed, rank,nca_setting=None, init_nca_design=None):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id, mode='modular', body = robot_structure.body, connections = robot_structure.connections,
                            nca_setting=nca_setting, init_nca_design=init_nca_design,env_id=env_id)

        env.seed(seed + rank)
        # Identify envs
        env.rank = rank 

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
        
        # Store the un-normalized rewards
        env = RecordEpisodeStatistics(env)

        return env

    return _thunk
    
# make parallel envs for different structures
def make_vec_envs(env_name,
                  robot_structures,
                  seed,
                  gamma,
                  device,
                  ret=True,
                  ob=True,
                  nca_setting=None, 
                  init_nca_design=None,
                  obs_to_norm={"modular","other"}
                  ):
    envs = [
        make_env(env_name, robot_structures[i], seed, i, nca_setting=nca_setting, init_nca_design=init_nca_design)
        for i in range(len(robot_structures))
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
        
    if gamma is None:
        envs = VecNormalize(envs, ret=False)
    else:
        envs = VecNormalize(envs, gamma=gamma, ret=ret, ob=ob, obs_to_norm=obs_to_norm)

    envs = VecPyTorch(envs, device)
    
    return envs

# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation

class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)

class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = self._obs_np2torch(obs)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self._obs_np2torch(obs)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
    
    def _obs_np2torch(self, obs):
        if isinstance(obs, dict):
            for ot, ov in obs.items():
                obs[ot] = torch.from_numpy(obs[ot]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
    
    def update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self,
        venv,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
        training=True,
        obs_to_norm={"modular","other"} 
    ):
        VecEnvWrapper.__init__(self, venv)

        self.ob_rms = self._init_ob_rms(ob, obs_to_norm)
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.obs_to_norm = obs_to_norm

    def _init_ob_rms(self, ob, obs_to_norm):
        if not ob:
            return None

        obs_space = self.observation_space
        ob_rms = {}

        if isinstance(obs_space, gym.spaces.Dict):
            for obs_type in obs_to_norm:
                shape = obs_space[obs_type].shape
                ob_rms[obs_type] = RunningMeanStd(shape=shape)
        else:
            shape = obs_space.shape
            ob_rms = RunningMeanStd(shape=shape)
        return ob_rms

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        self.ret[news] = 0.0
        return obs, rews, news, infos
    

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            for obs_type in self.ob_rms.keys():
                obs = self._obfilt_helper(obs, obs_type)
            return obs
        else:
            return obs

    ### handle NCA
    def _obfilt_helper(self, obs, obs_type, update=True):
        if isinstance(obs, dict):
            obs_p = obs[obs_type]
        else:
            obs_p = obs

        if self.training and update:
            obs_for_update = self.get_update_obs(obs)
            if obs_for_update is not None:
                self.ob_rms[obs_type].update(obs_for_update[obs_type])

        obs_p = np.clip(
            (obs_p - self.ob_rms[obs_type].mean)
            / np.sqrt(self.ob_rms[obs_type].var + self.epsilon),
            -self.clipob,
            self.clipob,
        )
        if isinstance(obs, dict):
            obs[obs_type] = obs_p
        else:
            obs = obs_p
        return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_update_obs(self, obs):
        obs_for_update = {}
        index = obs['stage']
        ac_idx = np.argwhere(index>0)
        batch_size = ac_idx.shape[0]
        if batch_size>0:
            for obs_type in self.ob_rms.keys():
                obs_for_update[obs_type] = obs[obs_type][ac_idx[:,0]]
            return obs_for_update
        else:
            return None


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.t0 = (
            time.time()
        )  # TODO: use perf_counter when gym removes Python 2 support
        self.episode_return = 0.0
        # Stores individual components of the return. For e.g. return might
        # have separate reward for speed and standing.
        self.episode_return_components = defaultdict(int)
        self.episode_length = 0
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        return observation

    def step(self, action):
        observation, reward, done, info = super(
            RecordEpisodeStatistics, self
        ).step(action)
        self.episode_return += reward
        self.episode_length += 1
        for key, value in info.items():
            if "__reward__" in key:
                self.episode_return_components[key] += value

        if done:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.episode_length,
                "t": round(time.time() - self.t0, 6),
            }
            for key, value in self.episode_return_components.items():
                info["episode"][key] = value
                self.episode_return_components[key] = 0

            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            self.episode_return = 0.0
            self.episode_length = 0
            
        return observation, reward, done, info
