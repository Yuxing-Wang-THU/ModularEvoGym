import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .transformer.distributions import Categorical

# FC Net of NCA
class MorphFCNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, number_state=9, device=None):
        super(MorphFCNet, self).__init__()
        
        self.input_dim = input_dim
        self.device = device

        self.linear1 = nn.Linear(input_dim, hidden_dim) 
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) 
        self.linear3 = nn.Linear(hidden_dim, number_state)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.FloatTensor(x)/3.0
        
        x = x.to(self.device)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x= self.linear3(x)
        material_dist = Categorical(logits=x)  
        return material_dist

    def design(self, x=None, mean_action=False):

        material_dist = self.forward(x)
        action = material_dist.mode() if mean_action else material_dist.sample()
        action_log_prob = material_dist.log_prob(action)

        if len(action_log_prob.shape) == 2:
            action_log_prob = action_log_prob.sum()
        else:
            action_log_prob = action_log_prob.sum(-1, keepdim=True)
        
        entropy = material_dist.entropy()

        return action, action_log_prob, entropy, material_dist

    def get_log_prob(self, x, action):
        material_dist, _ = self.forward(x)
        # skeleton transform log prob
        action_log_prob = material_dist.log_prob(action)
        if len(action_log_prob.shape) == 2:
            action_log_prob = action_log_prob.sum()
        else:
            action_log_prob = action_log_prob.sum(-1, keepdim=True)

        return action_log_prob

# Neural Cellular Automata
class NCA(nn.Module):
    def __init__(self, settings, obs_sample=None, device=None) -> None:
        super(NCA, self).__init__()
        self.settings = settings
        self.im_size = self.settings.im_size
        self.robot_size = self.im_size - 2
        self.structure_shape = (self.robot_size, self.robot_size)
        self.num_classes = self.settings.num_classes
        self.input_dim = self.settings.number_neighbors
        self.device = device
        self.DESIGN_SIZE = int(self.robot_size*self.robot_size)
        self.modular_obs_sample = obs_sample
    
        print(f"Input dimension: {self.input_dim} and classes: {self.num_classes}")
        
        self.morphogens = np.zeros(shape=(1, self.im_size, self.im_size))
        self.morphogens[0, int((self.im_size-1)/2), int((self.im_size-1)/2)] = 1

        # Simple Designer
        self.designer = MorphFCNet(input_dim=self.input_dim, number_state=self.num_classes,device=self.device)
       
        print("Morphgenes shape: ", self.morphogens.shape)
        
        self.iterations = self.settings.design_iterations

        print("Init morph: ", self.morphogens)

    def get_init_design(self):
        a = self.morphogens
        return copy.deepcopy(a)
        
    def design_act(self, state, act=None, value_net=None,mean_action=False,return_attention=False):
        if len(state.shape) == 2:
            batch_size =1
        else:
            batch_size = state.shape[0]
         
        output, action_log_prob, entropy, material_dist = self.designer.design(x=state, mean_action=mean_action)
        
        if act is not None:
            logp = material_dist.log_prob(act)
            logp = logp.sum(-1, keepdim=True)
            act_entropy = material_dist.entropy()
            act_entropy = act_entropy.sum(-1, keepdim=True)

        if value_net is not None:
            # unzip observations
            if isinstance(state, torch.Tensor):
                modular_state=state.to(self.device) / 3
            else:
                modular_state = torch.FloatTensor(state).to(self.device) / 3
            
            other_state = torch.zeros_like(torch.FloatTensor(self.modular_obs_sample['other']),device=self.device).repeat(batch_size,1)
            
            input_state = modular_state.reshape(batch_size, self.DESIGN_SIZE, -1).permute(
                        1, 0, 2)

            module_vals, v_attention_maps = value_net(
                    input_state, other_state, obs_mask=None, return_attention=return_attention
                )
            val = torch.divide(torch.sum(module_vals, dim=1, keepdim=True), self.DESIGN_SIZE)
            
            if act is not None:
                return val, logp, act_entropy
            else:
                return output, action_log_prob, val
        else:
            return output, action_log_prob
            
    def get_log_prob(self, x, action):
        return self.designer.get_log_prob(x, action)
