#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import math
from collections import deque, namedtuple
import os
import json

import torch
import torch.nn.functional as f
import torch.optim as optim

from .dqn_model import DQN
from matplotlib import pyplot as plt

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN():

    def __init__(self, num_agents, num_obs, num_actions, num_action_options, id):
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("Using GPU!!!!")
        else:
            'cpu'
            print("WARNING")
            print("WARNING")
            print("Using CPU")

        # State is:
        # Vector from robot to goal
        # Left wheel speed
        # Right wheel speed
        # Distance from cylinder to goal
        # Vector from robot to cylinder


        # Action is:
        # int representing joint left and right
        # wheel speed changes
        # See make_action() for parsing

        #TODO: Clean up variables
        self.id = id
        self.action_size = num_actions
        self.options_per_action = num_action_options
        self.num_agents = num_agents
        self.size_obs = num_obs
        self.memory = deque(maxlen=100000)
        self.last_action = 0.0
        # Can probably change these to lists... I think they were just used for plotting CHANGEME
        self.last_N_rewards = deque(maxlen=10)
        self.minimum_train_eps = 4
        self.episode_num = 0
        self.time_ticks = 0
        # Discount Factor
        self.gamma = 0.99
        # Exploration Rate: at the beginning do 100% exploration
        self.epsilon = 1.0
        # Set floor for how low epsilon can go
        self.epsilon_min = 0.01
        # Set the learning rate
        self.learning_rate = 0.00015
        # batch_size
        self.batch_size = 16
        # Decay epsilon so we can shift from exploration to exploitation
        self.epsilon_decay_step = 1.0/500000

        # Size_obs -2 to account for Done and reward not being known predictors
        self.policy_net = DQN(self.size_obs - 1,
                              self.action_size, self.options_per_action).to(self.device)
        self.target_net = DQN(self.size_obs - 1,
                              self.action_size, self.options_per_action).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.loss = 0
        self.running_reward = 0
        self.rewards_N_ep = []

        self.update_target_freqency = 100
        self.save_model_frequency = 50

        self.model_file_path = 'python_code/trained_models/test1/Q_Network_Parameters_'
        self.data_file_path = 'python_code/Data/test1/Model_Info/Model_Info_'+str(id)+'.json'
        self.loss_file_path = 'python_code/Data/test1/Loss_Info/Loss_Info_'+str(id)+'.json'
        self.running_model_info = {}
        self.loss_info = []

    def train(self):
        self.last_N_rewards.append(self.running_reward)
        if self.episode_num > self.minimum_train_eps:
            self.learn()
        else:
            self.loss_info.append(0) # TODO this is wrong... what should it be?

    def learn(self):
        # Randomly sample replay buffer for size = batch_size
        sampled_batch = self.replay_buffer(self.batch_size)
        # Unpack batch into states, actions, rewards, next states, and dones
        states, actions, rewards, next_states, dones = list(zip(*sampled_batch))
        # Transform numpy to tensor
        states = torch.from_numpy(np.stack(states)).to(self.device)
        actions = torch.from_numpy(np.stack(actions)).to(self.device)
        rewards = torch.from_numpy(np.stack(rewards)).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device)
        dones = torch.from_numpy(np.stack(dones)).to(self.device)
        # produce q function based on the states
        qfun = self.policy_net(states)
        # get state-action values based on the actions taken
        state_action_values = qfun.gather(1, actions.unsqueeze(-1).long()).squeeze()
        # Get predicted values from the target net based on next states
        next_state_values = self.target_net(next_states).max(1).values.detach()
        # Calculate the TD error
        TD_error = rewards + self.gamma*next_state_values*(1-dones)
        # TODO: Loss is noise at the moment, we need to revisit how this is calculated and if it makes sense for what we are doing.
        self.loss = f.smooth_l1_loss(state_action_values, TD_error)
        # Clear the gradiants
        self.optimizer.zero_grad()
        # Back propogate the loss
        self.loss.backward()
        # TODO: What does this do?
        self.optimizer.step()
        # Add store loss
        self.loss_info.append(self.loss.item())



    def make_action(self, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """

        observation = self.memory[-1]
        observation = torch.tensor(observation[0], dtype=torch.float32).to(self.device)
        observation = observation.unsqueeze(0)

        if not test:
            self.time_ticks += 1
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.options_per_action**self.action_size) #TODO: Is this correct?
            else:
                with torch.no_grad():
                    action = self.target_net(observation).detach().cpu().clone().numpy() #WHY TARGET AND NOT POLICY???
                    action = np.argmax(action, axis=1)[0]

            if self.epsilon > self.epsilon_min:
                self.epsilon = max(0, self.epsilon - self.epsilon_decay_step)
        else:
            with torch.no_grad():
                action = torch.argmax(self.policy_net(observation)).item()
        self.last_action = action
        return self.parse_action(action)

    def parse_action(self, selection_number):
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
        l_wheel = (math.floor(selection_number/self.options_per_action) - 1)/10.0
        r_wheel = (selection_number%self.options_per_action - 1)/10.0
        return np.array([l_wheel, r_wheel], dtype=np.float32)

    def receive_observation(self, state, done):
        reward = state[-1] # Reward is always the last element of state
        state = state[:-1]
        if len(self.memory)==0:
            last_state = state
        else: last_state = self.memory[-1][-2]
        self.push(last_state, self.last_action, reward, state, done)
        self.running_reward += self.gamma*reward
        self.train()

    def push(self, state, action, reward, next_state, done):
        """
        Push new data to buffer and remove the old one if the buffer is full.
        """
        action = np.array(action, dtype=np.uint8)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def replay_buffer(self, batch_size):
        """
        Select batch from buffer.
        """
        return random.sample(self.memory, batch_size)

    def unpack(self, results):
        result = results[0]
        state, reward, done, info = result.asTuple()
        return state.as1xnArray(), reward, done, info

    def init_game_setting(self):
        pass
    
    def finish_episode(self):
        print('Episode Number:', self.episode_num)
        print('Epsilon:', self.epsilon)
        print('Reward:', self.running_reward)
        if self.episode_num > self.minimum_train_eps:
            self.running_model_info[self.episode_num] = [self.epsilon,
                                self.running_reward]
            with open(self.data_file_path, 'w') as fp:
                json.dump(self.running_model_info, fp)
            with open(self.loss_file_path, 'w') as lp:
                json.dump(self.loss_info, lp)
        else:
            self.running_model_info[self.episode_num] = [self.epsilon,
                                self.running_reward]
            with open(self.data_file_path, 'w') as fp:
                json.dump(self.running_model_info, fp)
            with open(self.loss_file_path, 'w') as lp:
                json.dump(self.loss_info, lp)

        self.episode_num += 1
        self.running_reward = 0
        if self.episode_num > self.minimum_train_eps:
            if self.episode_num % self.save_model_frequency == 0:
                avg_last_10 = sum(self.last_N_rewards)/10
                print("Average of last 10 rewards:\t",avg_last_10)
                self.rewards_N_ep.append(avg_last_10)
                print('------------ Saving Model -------------')
                torch.save(self.policy_net.state_dict(), self.model_file_path+str(self.episode_num)+'.pth')

            if self.episode_num % self.update_target_freqency == 0:
                print('------------ UPDATING TARGET -------------')
                self.target_net.load_state_dict(self.policy_net.state_dict())
    
