from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, output_dim: int, hidden_layers=(256, 128, 64)):
    layers = []
    last = input_dim
    for width in hidden_layers:
        layers.extend((nn.Linear(last, int(width)), nn.ReLU()))
        last = int(width)
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64)):
        super().__init__()
        self.backbone = mlp(state_dim, 2 * action_dim, hidden_layers)
        self.action_dim = int(action_dim)

    def forward(self, state):
        mean, log_std = self.backbone(state).chunk(2, dim=-1)
        return mean, log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def sample(self, state, deterministic=False):
        mean, log_std = self(state)
        if deterministic:
            pre_tanh = mean
            log_prob = None
        else:
            std = log_std.exp()
            distribution = Normal(mean, std)
            pre_tanh = distribution.rsample()
            log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        action = torch.tanh(pre_tanh)
        if log_prob is not None:
            # Change-of-variables correction for tanh squashing.
            log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob


class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64)):
        super().__init__()
        self.net = mlp(state_dim, action_dim, hidden_layers)

    def forward(self, state):
        return torch.tanh(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64)):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1, hidden_layers)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1))
