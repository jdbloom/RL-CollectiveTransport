import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, num_actions, observation_size, num_ops_per_action):
        super(DeepQNetwork, self).__init__()
        print('DQN network observation_size = ', observation_size)

        output_dims = (num_ops_per_action**num_actions)
        
        self.fc1 = nn.Linear(observation_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dims)

        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x))
        actions = self.fc3(x1)
        return actions

    def save_model(self, file_path):
        print('... saving model ...')
        T.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        print('... loading model ...')
        self.load_state_dict(T.load(file_path))
