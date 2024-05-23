from .Actor import Actor

import numpy as np
import math
from collections import namedtuple
import statistics
from itertools import product

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

Loss = nn.MSELoss()

class Agent(Actor):
    def __init__(self, n_agents, n_obs, n_actions, options_per_action, id, learning_scheme,
                 n_chars=4, intention_look_back = 2, min_max_action = 1, use_intention=False, 
                 use_recurrent=False, attention=False, gnn=False, intention_neighbors=False,
                 meta_param_size = 1, seq_len=5, edge_index = None, prox_filter_angle = 45):

        args = {'id':id, 'n_obs':n_obs, 'n_actions':n_actions, 'options_per_action':options_per_action, 'n_agents':n_agents,
                'n_chars':n_chars, 'meta_param_size':meta_param_size, 'intention':use_intention, 'recurrent_intention':use_recurrent,
                'attention':attention, 'gnn':gnn, 'intention_neighbors':intention_neighbors, 'intention_look_back':intention_look_back,
                'seq_len':seq_len}

        super().__init__(**args)
        self.n_agents = n_agents
        self.learning_scheme = learning_scheme
        self.decimals = 2
        min_max = 1.25
        bins = 8

        self.actions = list(product([-0.1,0,0.1], repeat=2))
        
        self.build_networks(learning_scheme)

        if self.intention:
            self.build_intention_network('DDPG')
            if self.intention_neighbors:
                self.build_neighbors()

    def make_agent_state(self, env_obs, heading_intention=None, global_knowledge=None):
        return env_obs   

    def angle_difference(self, a1, a2):
        diff = a1 - a2
        while diff < -180.0:
            diff += 360.0
        while diff > 180.0:
            diff -= 360
        return diff

    def reset_obj_stats(self):
        self.object_stats = []        

    def choose_agent_action(self, observation, test=False):
        if self.learning_scheme == 'None':
            return [0, 0]

        if self.networks['learning_scheme'] == 'DDPG' or self.networks['learning_scheme'] == 'TD3':
            actions = self.choose_action(observation, self.networks, test)
            if math.isnan(actions[0]):
                actions = [0,0]

        return actions