from .Actor import Actor

import numpy as np
import math
from collections import namedtuple
import statistics

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

Loss = nn.MSELoss()

class Agent(Actor):
    def __init__(self, n_agents, n_obs, n_actions, options_per_action, id, learning_scheme,
                 n_chars=4, intention_look_back = 2, min_max_action = 1, use_intention=False, 
                 use_recurrent=False, attention=False, meta_param_size = 1, seq_len=5):


        args = {'id':id, 'n_obs':n_obs, 'n_actions':n_actions, 'options_per_action':options_per_action, 'n_agents':n_agents,
                'n_chars':n_chars, 'meta_param_size':meta_param_size, 'intention':use_intention, 'recurrent_intention':use_recurrent,
                'attention':attention, 'intention_look_back':intention_look_back, 'seq_len':seq_len}

        super().__init__(**args)

        self.learning_scheme = learning_scheme

        self.object_stats = []
        self.min_obj_stats = np.zeros(4) # vel, accel, ang_vel, ang_accel
        self.max_obj_stats = np.zeros(4)
        self.decimals = 2
        min_max = 1.25
        bins = 8
        self.angle_bins = np.arange(-180, 180, 360/bins)
        self.acceleration_bins = np.around(np.arange(-min_max, min_max, (min_max*2)/bins), self.decimals)
        self.binned_angle = None
        self.binned_acceleration = None
        self.obj_state = None
        
        
        self.build_networks(learning_scheme)

        if self.intention:
            self.build_intention_network('DDPG')

    def make_agent_state(self, env_obs, heading_intention):
        if self.intention:
            env_obs = np.concatenate((env_obs, np.array([heading_intention]))) 
        return env_obs   

    def angle_difference(self, a1, a2):
        diff = a1 - a2
        while diff < -180.0:
            diff += 360.0
        while diff > 180.0:
            diff -= 360
        return diff

    def store_object_stats(self, obj_stats, calculate):
        # len = 3 is so that we can calculate velocity and acceleration
        # calculate is a flag that tracks based on episode time steps.
        # we dont want to do any calculations within the first 3 time
        # steps of an episode.
        if not calculate:
            self.object_stats.append(obj_stats)
        else:
            # get rid of the oldest stat
            self.object_stats.pop(0)
            self.object_stats.append(obj_stats)
            velocity_t0 = math.sqrt((self.object_stats[2][0] - self.object_stats[1][0])**2 + (self.object_stats[2][1] - self.object_stats[1][1])**2)/0.1
            velocity_t1 = math.sqrt((self.object_stats[1][0] - self.object_stats[0][0])**2 + (self.object_stats[1][1] - self.object_stats[0][1])**2)/0.1
            acceleration = (velocity_t0 - velocity_t1)/0.1

            angular_velocity_t0 = self.angle_difference(self.object_stats[2][5], self.object_stats[1][5])/0.1

            angular_velocity_t1 = self.angle_difference(self.object_stats[1][5], self.object_stats[0][5])/0.1
            angular_acceleration = (angular_velocity_t0 - angular_velocity_t1)/0.1

            self.binned_acceleration = int(min(self.acceleration_bins, key=lambda x:abs(x-acceleration))*(10**self.decimals))
            self.binned_angle = int(min(self.angle_bins, key=lambda x:abs(x-self.object_stats[2][5])))
            self.obj_state = [self.binned_acceleration, self.binned_angle]


            # This is for finding bin limits only
            if velocity_t0 > self.max_obj_stats[0]: self.max_obj_stats[0] = velocity_t0
            if velocity_t0 < self.min_obj_stats[0]: self.min_obj_stats[0] = velocity_t0
            if acceleration > self.max_obj_stats[1]: self.max_obj_stats[1] = acceleration
            if acceleration < self.min_obj_stats[1]: self.min_obj_stats[1] = acceleration
            if angular_velocity_t0 > self.max_obj_stats[2]: self.max_obj_stats[2] = angular_velocity_t0
            if angular_velocity_t0 < self.min_obj_stats[2]: self.min_obj_stats[2] = angular_velocity_t0
            if angular_acceleration > self.max_obj_stats[3]: self.max_obj_stats[3] = angular_acceleration
            if angular_acceleration < self.min_obj_stats[3]: self.min_obj_stats[3] = angular_acceleration

    def reset_obj_stats(self):
        self.object_stats = []        

    def choose_agent_action(self, observation, failures, test=False):
        if self.learning_scheme == 'None':
            # Not sure what to do here for no learning
            return [0, 0, 0], 0

        if failures:
            self.failed = True
            return self.failure_action, self.failure_action_code

        self.failed = False
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            action_num = self.choose_action(observation, self.networks, test)
            actions = self.parse_action(action_num)

        if self.networks['learning_scheme'] == 'DDPG' or self.networks['learning_scheme'] == 'TD3':
            actions = self.choose_action(observation, self.networks, test)
            actions = np.pad(actions, (0, 1))
            action_num = None

        return actions, action_num

    
    def parse_action(self, action_num):
        '''
        This function will parse the number action to
        a set of wheel actions:

        0 - (- 1,-1)
        1 - (-1, 0)
        2 - (-1, 1)
        3 - (0, -1)
        4 - (0, 0)
        5 - (0, 1)
        6 - (1, -1)
        7 - (1, 0)
        8 - (1, 1)
        '''
        if action_num < 0 or action_num >=self.options_per_action**self.n_actions:
            raise Exception('Action Number Out of Range:'+str(action_num))
        l_wheel = round((math.floor(action_num/self.options_per_action) - 1)/10.0, 1)
        r_wheel = round((action_num%self.options_per_action - 1)/10.0, 1)
        # Trailing zero is hardcoded control for gripper
        return np.array([l_wheel, r_wheel, 0])
        

    def choose_object_intention(self, positions, agent_prox_flags, test = False):
        observation = np.append(np.array(positions), np.array(agent_prox_flags))
        return self.choose_action(observation, self.intention_networks, test)        

