import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin = None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)

class ActorNetwork(nn.Module):
    def __init__(self, num_actions, observation_size, num_ops_per_action, min_max_action = 1):
        super(ActorNetwork, self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        output_dims = (num_ops_per_action**num_actions)

        # allows for us to change the range of the action from (-min_max_action, min_max_action)
        self.min_max_action = min_max_action

        self.fc1_dims = 400
        self.fc2_dims = 300

        self.fc1 = nn.Linear(observation_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, output_dims) # For Continuous Action, Make this num_actions instead of output dims

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.init_weights(3e-3) #find where this comes from and maybe find the purpose of this line???

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.mu.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        prob = self.fc1(x)
        prob = self.relu(prob)
        prob = self.fc2(prob)
        prob = self.relu(prob)
        mu = self.mu(prob)
        # out = self.min_max_action*self.tanh(out) # This is needed for continuous action space
        return mu


class CriticNetwork(nn.Module):
    def __init__(self, num_actions, observation_size):
        super(CriticNetwork, self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.fc1_dims = 400
        self.fc2_dims = 300

        self.fc1 = nn.Linear(observation_size+num_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.relu = nn.ReLU()
        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.q.weight.data.uniform_(-init_w, init_w)

    def forward(self, X):
        state, action = X
        action_value = self.fc1(T.cat([state, action], 1))
        action_value = self.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = self.relu(action_value)
        action_value = self.q(action_value)
        return action_value
