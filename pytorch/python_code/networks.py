import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def fanin_init(size, fanin = None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)

############################################################################
# Action Network for DQN
############################################################################
class DQN(nn.Module):
    def __init__(self, lr, num_actions, observation_size, num_ops_per_action,
                 fc1_dims = 64, fc2_dims = 128):
        super().__init__()

        output_dims = num_ops_per_action**num_actions

        self.fc1 = nn.Linear(observation_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

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
        print('... saving DeepQNetwork model ...')
        T.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        print('... loading DeepQNetwork model ...')
        self.load_state_dict(T.load(file_path))

############################################################################
# Communication Network for DDQN
############################################################################
class DDQNComms(nn.Module):
    def __init__(self, *, lr=None, observation_size=None, alphabet_size=None,
                 fc1_dims = 64, fc2_dims = 128):
        super().__init__()
        output_dims = alphabet_size

        #print('Comms Network has input size = ', observation_size, 'and output size = ', output_dims)

        self.fc1 = nn.Linear(observation_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x))
        raw_message = self.fc3(x1)
        return raw_message

    def save_model(self, file_path):
        print('... saving DoubleDeepQComms model ...')
        T.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        print('... loading DoubleDeepQComms model ...')
        self.load_state_dict(T.load(file_path))

############################################################################
# Action Network for DDQN
############################################################################
class DDQN(nn.Module):
    def __init__(self, *, lr = None, num_actions = None, observation_size = None,
                 num_ops_per_action = None, fc1_dims = 64, fc2_dims = 128):
        super().__init__()

        output_dims = (num_ops_per_action**num_actions)
        #print('DQN network observation_size = ', observation_size, 'and output size = ', output_dims)

        self.fc1 = nn.Linear(observation_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

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
        print('... saving DoubleDeepQNetwork model ...')
        T.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        print('... loading DoubleDeepQNetwork model ...')
        self.load_state_dict(T.load(file_path))

############################################################################
# Actor Network for DDPG
############################################################################
class DDPGActorNetwork(nn.Module):
    def __init__(self, num_actions, observation_size, num_ops_per_action, name, min_max_action = 1):
        super().__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        #output_dims = (num_ops_per_action**num_actions) # For discrete action space
        output_dims = num_actions

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

        self.name = name


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
        mu = self.min_max_action*self.tanh(mu) # This is needed for continuous action space
        return mu

    def save_checkpoint(self, path):
        print('... saving', self.name,'chekpoint ...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, 'checkpoint ...', path)
        self.load_state_dict(T.load(path + '_' + self.name))

############################################################################
# Critic Network for DDPG
############################################################################
class DDPGCriticNetwork(nn.Module):
    def __init__(self, num_actions, observation_size, name):
        super().__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.fc1_dims = 400
        self.fc2_dims = 300
        self.fc1 = nn.Linear(observation_size+num_actions, self.fc1_dims) #! 1 represents the action number. Change this to num_actions when we move to continuous
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.relu = nn.ReLU()
        self.init_weights(3e-3)

        self.name = name

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

    def save_checkpoint(self, path):
        print('... saving', self.name,'chekpoint ...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, 'checkpoint ...', path)
        self.load_state_dict(T.load(path + '_' + self. name))

############################################################################
# Actor Network for TD3
############################################################################
class TD3ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, min_max_action = 1):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.min_max_action = min_max_action
        self.name = name + '_TD3'


        self.fc1 = nn.Linear(input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.min_max_action * T.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self, path):
        print('... saving', self.name,'chekpoint ...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, 'checkpoint ...')
        T.load_state_dict(T.load(path + '_' + self.name))

############################################################################
# Critic Network for TD3
############################################################################
class TD3CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name + '_TD3'



        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = F.relu(self.fc1(T.cat([state, action], dim = 1)))
        q1_action_value = F.relu(self.fc2(q1_action_value))
        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self, path):
        print('... saving', self.name,'chekpoint ...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, 'checkpoint ...')
        T.load_state_dict(T.load(path + '_' + self.name))
