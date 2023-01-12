import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical as TorchCategorical
import numpy as np

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class AddFixedBias(nn.Module):
    def __init__(self, bias):
        super(AddFixedBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1), requires_grad=False)

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy()

    def mode(self):
        return self.mean

class DiagGaussian(nn.Module):
    def __init__(self, num_outputs):
        super(DiagGaussian, self).__init__()
        
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = x
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class FixedDiagGaussian(nn.Module):
    def __init__(self, num_outputs, std):
        super(FixedDiagGaussian, self).__init__()

        log_std = np.log(std)
        self.logstd = AddFixedBias(torch.ones(num_outputs)*log_std)
       
    def forward(self, x):
        action_mean = x
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class DiagGaussian2(Normal):

    def __init__(self, loc, scale):
        super().__init__(loc, scale)

    def kl(self):
        loc1 = self.loc
        scale1 = self.scale
        log_scale1 = self.scale.log()
        loc0 = self.loc.detach()
        scale0 = self.scale.detach()
        log_scale0 = log_scale1.detach()
        kl = log_scale1 - log_scale0 + (scale0.pow(2) + (loc0 - loc1).pow(2)) / (2.0 * scale1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_prob(self, value):
        return super().log_prob(value)
        
    def mean_sample(self):
        return self.loc

class Categorical(TorchCategorical):

    def __init__(self, probs=None, logits=None, uniform_prob=0.0):
        super().__init__(probs, logits)
        self.uniform_prob = uniform_prob
        if uniform_prob > 0.0:
            self.uniform = TorchCategorical(logits=torch.zeros_like(self.logits))

    # def sample(self):
    #     return super().sample().unsqueeze(-1)
    def sample(self):
        if self.uniform_prob == 0.0:
            return super().sample()
        else:
            if torch.bernoulli(torch.tensor(self.uniform_prob)).bool():
                # print('unif')
                return self.uniform.sample()
            else:
                # print('original')
                return super().sample()

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True).squeeze()

    def log_prob(self, value):
        if self.uniform_prob == 0.0:
            return super().log_prob(value).unsqueeze(1)
        else:
            return super().log_prob(value).unsqueeze(1) * (1 - self.uniform_prob) + self.uniform.log_prob(value).unsqueeze(1) * self.uniform_prob

class GaussianCategorical:

    def __init__(self, logits, scale, gaussian_dim):
        self.gaussian_dim = gaussian_dim
        self.logits = logits
        self.loc = loc = logits[:, :gaussian_dim]
        self.scale = scale = scale[:, :gaussian_dim]
        self.gaussian = DiagGaussian(loc, scale)
        self.discrete = Categorical(logits=logits[:, gaussian_dim:])

    def log_prob(self, value):
        gaussian_log_prob = self.gaussian.log_prob(value[:, :self.gaussian_dim])
        discrete_log_prob = self.discrete.log_prob(value[:, -1])
        return gaussian_log_prob + discrete_log_prob

    def mean_sample(self):
        gaussian_samp = self.gaussian.mean_sample()
        discrete_samp = self.discrete.mean_sample().unsqueeze(1).float()
        return torch.cat([gaussian_samp, discrete_samp], dim=-1)

    def sample(self):
        gaussian_samp = self.gaussian.sample()
        discrete_samp = self.discrete.sample().unsqueeze(1).float()
        return torch.cat([gaussian_samp, discrete_samp], dim=-1)
