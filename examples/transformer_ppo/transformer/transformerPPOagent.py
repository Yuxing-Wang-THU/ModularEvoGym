from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from .transformermodel import TransformerModel
from .distributions import DiagGaussian,FixedDiagGaussian

class TransformerPPOAC(nn.Module):

    def __init__(
        self,
        modular_state_dim,
        modular_action_dim,
        sequence_size,
        other_feature_size,
        ppo_args=None,
        trans_args=None,
        ac_type=None,
        device=None
    ):
        super(TransformerPPOAC, self).__init__()
        self.sequence_size = sequence_size
        self.input_state = [None] * self.sequence_size
        self.other_feature_size = other_feature_size
        self.state_dim = modular_state_dim
        self.action_dim = modular_action_dim
        self.ppo_args = ppo_args
        self.trans_args = trans_args
        self.ac_type=ac_type
        self.device = device

        self.v_net = TransformerModel(
            feature_size=self.state_dim,
            output_size=self.action_dim,
            sequence_size=self.sequence_size,
            other_feature_size=self.other_feature_size,
            ninp=trans_args.attention_embedding_size,
            nhead=trans_args.attention_heads,
            nhid= trans_args.attention_hidden_size,
            nlayers=trans_args.attention_layers,
            dropout=trans_args.dropout_rate,
            args=trans_args,
            use_transformer=self.ac_type,
            is_actor = False
        )

        self.mu_net = TransformerModel(
            feature_size=self.state_dim,
            output_size=self.action_dim,
            sequence_size=self.sequence_size,
            other_feature_size=self.other_feature_size,
            ninp=trans_args.attention_embedding_size,
            nhead=trans_args.attention_heads,
            nhid= trans_args.attention_hidden_size,
            nlayers=trans_args.attention_layers,
            dropout=trans_args.dropout_rate,
            args=trans_args,
            use_transformer=self.ac_type,
            is_actor = True
        )

        self.num_actions = self.sequence_size

        if self.ppo_args.ACTION_STD_FIXED:
            self.act_dist = FixedDiagGaussian(num_outputs=self.num_actions, std=ppo_args.ACTION_STD)
        else:
            self.act_dist = DiagGaussian(num_outputs=self.num_actions)

    def forward(self, state, act=None, return_attention=False):

        # unzip observations
        modular_state, other_state,act_mask,obs_padding = (
            state["modular"],
            state["other"],
            state["act_mask"],
            state["obs_mask"],
        )
        
        batch_size = modular_state.shape[0]
        act_mask = act_mask.bool()
        obs_padding = obs_padding.bool()
        self.input_state = modular_state.reshape(batch_size, self.sequence_size, -1).permute(
            1, 0, 2
        )
        # module_vals shape: (batch_size, num_modular * J)
        module_vals, v_attention_maps = self.v_net(
            self.input_state, other_state, obs_padding, return_attention=return_attention
        )
        # val shape: (batch_size, 1)
        # Zero out mask values
        module_vals = module_vals * (1 - obs_padding.int())
        num_limbs = self.sequence_size - torch.sum(obs_padding.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(module_vals, dim=1, keepdim=True), num_limbs)

        # mu shape: (batch_size, num_modular * J)
        mu, mu_attention_maps = self.mu_net(
            self.input_state,other_state, obs_padding, return_attention=return_attention
        )

        pi = self.act_dist(mu)

        # In case next step is training
        if act is not None:
            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            logp = logp.sum(-1, keepdim=True)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.sum(-1, keepdim=True)
            # entropy = entropy.mean()
            return val, pi, logp, entropy
        else:
            if return_attention:
                return val, pi, v_attention_maps, mu_attention_maps
            else:
                return val, pi, None, None
    
    def reset_seq_size(self,seq_size):
        self.sequence_size = seq_size
        self.input_state = [None] * self.sequence_size
        self.num_actions = seq_size
        self.mu_net.reset_seq_size(seq_size)
        self.v_net.reset_seq_size(seq_size)
        if self.use_nca_vnet:
            self.nca_v_net.reset_seq_size(seq_size) 
        
        if self.ppo_args.ACTION_STD_FIXED:
            self.act_dist = FixedDiagGaussian(num_outputs=self.num_actions, std=self.ppo_args.ACTION_STD)
        else:
            self.act_dist = DiagGaussian(num_outputs=self.num_actions)

class Agent(nn.Module):
    
    def __init__(self, actor_critic):
        super(Agent, self).__init__()
        self.ac = actor_critic

    def forward(self, obs, act):
        index = obs['stage']
        batch_size = index.shape[0]
        ac_index = np.argwhere(index.cpu().numpy()>0)

        val = torch.zeros(batch_size,1).to(self.ac.device)
        logp = torch.zeros(batch_size,1).to(self.ac.device)
        ent = torch.zeros(batch_size,1).to(self.ac.device)

        ### ac batch
        if ac_index.shape[0]>0:
            if isinstance(obs, dict):
                ac_obs_batch ={}
                for ot, ov in obs.items():
                    if ot == 'stage'or ot == 'design':
                        pass
                    else:
                        ac_obs_batch[ot] = ov.view(-1, *ov.size()[1:])[ac_index[:,0]]

            ac_act_batch = act[ac_index[:,0]]
            ac_val, _, ac_logp, ac_ent = self.ac(ac_obs_batch, ac_act_batch)
            
            val[ac_index[:,0]] = ac_val
            logp[ac_index[:,0]] = ac_logp
            ent[ac_index[:,0]] = ac_ent

        ent = ent.mean()
        return val, logp, ent

    @torch.no_grad()
    def uni_act(self, obs, mean_action=False):
        index = obs['stage']
        batch_size = index.shape[0]
        ac_idx = np.argwhere(index.cpu().numpy()>0)
        
        val = torch.zeros(batch_size,1).to(self.ac.device)
        logp = torch.zeros(batch_size,1).to(self.ac.device)
        act = torch.zeros(batch_size,self.ac.sequence_size).to(self.ac.device)

        ### ac batch
        if ac_idx.shape[0]>0:
            if isinstance(obs, dict):
                ac_obs_batch ={}
                for ot, ov in obs.items():
                    if ot == 'stage' or ot == 'design':
                        pass
                    else:
                        ac_obs_batch[ot] = ov[ac_idx[:,0]]

            ac_val, ac_act, ac_logp = self.act(ac_obs_batch,mean_action=mean_action)

            val[ac_idx[:,0]] = ac_val
            logp[ac_idx[:,0]] = ac_logp

            for j in range(ac_idx.shape[0]):
                act[ac_idx[:,0][j]] = ac_act[j]
        return val, act, logp

    @torch.no_grad()
    def act(self, obs, mean_action=False):
        val, pi, _, _ = self.ac(obs)
        if mean_action:
            act = pi.mode()
        else:
            act = pi.sample()
        logp = pi.log_prob(act)
        act_mask = obs["act_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        del pi
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        val, act, logp = self.uni_act(obs)
        return val
