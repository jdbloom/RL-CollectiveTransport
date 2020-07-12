#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:02:16 2020

@author: aaaaambition
"""
import torch.nn as nn
import torch.nn.functional as f


class DQN(nn.Module):
    """Initialize a deep Q-learning network"""
    def __init__(self, observation_size, num_actions, num_ops_per_action):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, observation_size*2),
            nn.ReLU(),
            nn.Linear(observation_size*2, observation_size*2),
            nn.ReLU(),
            nn.Linear(observation_size*2, observation_size*2),
            nn.ReLU(),
            nn.Linear(observation_size*2, num_ops_per_action**num_actions)
        )

    def forward(self, x):
        return self.net(x)
