import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

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
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian5(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian5, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.num_outputs = num_outputs
        self.fc_mean1 = nn.Sequential(init_(nn.Linear(num_inputs, num_inputs)), nn.Tanh(), init_(nn.Linear(num_inputs, 1)))
        self.fc_mean2 = nn.Sequential(init_(nn.Linear(num_inputs, num_inputs)), nn.Tanh(), init_(nn.Linear(num_inputs, 1)))
        self.fc_mean3 = nn.Sequential(init_(nn.Linear(num_inputs, num_inputs)), nn.Tanh(), init_(nn.Linear(num_inputs, 1)))
        self.fc_mean4 = nn.Sequential(init_(nn.Linear(num_inputs, num_inputs)), nn.Tanh(), init_(nn.Linear(num_inputs, 1)))
        self.fc_mean5 = nn.Sequential(init_(nn.Linear(num_inputs, num_inputs)), nn.Tanh(), init_(nn.Linear(num_inputs, 1)))
        
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean1 = self.fc_mean1(x)
        action_mean2 = self.fc_mean2(x)
        action_mean3 = self.fc_mean3(x)
        action_mean4 = self.fc_mean4(x)
        action_mean5 = self.fc_mean5(x)
        action_mean = torch.cat([action_mean1,action_mean2,action_mean3,action_mean4,action_mean5], 1)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp()), action_mean

class DiagGaussian1(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian1, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp()), action_mean

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
