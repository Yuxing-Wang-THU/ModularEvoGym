import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class Buffer(object):
    def __init__(self, obs_sample, act_shape, num_envs=None,cfg=None):
        self.cfg=cfg
        self.num_envs = num_envs
        T, P = self.cfg.TIMESTEPS, num_envs
        self.obs = {}
        for obs_type, obs_value in obs_sample.items():
            if obs_type == "stage":
                continue
            self.obs[obs_type] = torch.zeros(T, P, obs_value.shape[0])
        
        # self.obs['design'] = torch.zeros(T, P, self.cfg.DESIGN_SIZE, self.cfg.CELLDIM*self.cfg.NUM_NEIGHBOR)
        self.obs["stage"] = torch.ones(T, P, 1)
        
        self.act = torch.zeros(T, P, act_shape)
        self.val = torch.zeros(T, P, 1)
        self.rew = torch.zeros(T, P, 1)
        self.ret = torch.zeros(T, P, 1)
        self.logp = torch.zeros(T, P, 1)
        self.masks = torch.ones(T, P, 1)
        self.timeout = torch.ones(T, P, 1)
        self.step = 0

    def to(self, device):
        if isinstance(self.obs, dict):
            for obs_type, obs_space in self.obs.items():
                self.obs[obs_type] = self.obs[obs_type].to(device)
        else:
            self.obs = self.obs.to(device)
        self.act = self.act.to(device)
        self.val = self.val.to(device)
        self.rew = self.rew.to(device)
        self.ret = self.ret.to(device)
        self.logp = self.logp.to(device)
        self.masks = self.masks.to(device)
        self.timeout = self.timeout.to(device)
    
    def insert(self, obs, act, logp, val, rew, masks, timeouts):
        if isinstance(obs, dict):
            for obs_type, obs_val in obs.items():
                self.obs[obs_type][self.step] = obs_val
        else:
            self.obs[self.step] = obs
        self.act[self.step] = act
        self.val[self.step] = val
        self.rew[self.step] = rew
        self.logp[self.step] = logp
        self.masks[self.step] = masks
        self.timeout[self.step] = timeouts
        self.step = (self.step + 1) % self.cfg.TIMESTEPS

    def compute_returns(self, next_value):
        """
        We use ret as approximate gt for value function for training. When step
        is terminal state we need to handle two cases:
        1. Agent Died: timeout[step] = 1 and mask[step] = 0. This ensures
           gae is reset to 0 and self.ret[step] = 0.
        2. Agent Alive but done true due to timeout: timeout[step] = 0
           mask[step] = 0. This ensures gae = 0 and self.ret[step] = val[step].
        """
        gamma, gae_lambda = self.cfg.GAMMA, self.cfg.GAE_LAMBDA
        # val: (T+1, P, 1), self.val: (T, P, 1) next_value: (P, 1)
        # if self.val.shape[1] == 1:
        #     val = torch.cat((self.val.squeeze(2), next_value.t())).unsqueeze(2)
        # else:
        val = torch.cat((self.val.squeeze(), next_value.t())).unsqueeze(2)
        gae = 0
        for step in reversed(range(self.cfg.TIMESTEPS)):
            delta = (
                self.rew[step]
                + gamma * val[step + 1] * self.masks[step]
                - val[step]
            ) * self.timeout[step]
            gae = delta + gamma * gae_lambda * self.masks[step] * gae
            self.ret[step] = gae + val[step]

    def get_sampler(self, adv):
        dset_size = self.cfg.TIMESTEPS * self.num_envs
        assert dset_size >= self.cfg.NUM_MINI_BATCH_SIZE
        mini_batch_size = dset_size // self.cfg.NUM_MINI_BATCH_SIZE

        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            mini_batch_size,
            drop_last=True,
        )   

        for idxs in sampler:
            batch = {}
            batch["ret"] = self.ret.view(-1, 1)[idxs]
            if isinstance(self.obs, dict):
                batch["obs"] = {}
                for ot, ov in self.obs.items():
                    if ot == 'stage':
                        pass
                    else:
                        batch["obs"][ot] = ov.view(-1, *ov.size()[2:])[idxs]
                batch["obs"]['stage'] = self.obs['stage'].view(-1, 1)[idxs]
            else:
                batch["obs"] = self.obs.view(-1, *self.obs.size()[2:])[idxs]
            
            batch["val"] = self.val.view(-1, 1)[idxs]
            batch["act"] = self.act.view(-1, self.act.size(-1))[idxs]
            batch["adv"] = adv.view(-1, 1)[idxs]
            batch["logp_old"] = self.logp.view(-1, 1)[idxs]
            yield batch
