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
    def __init__(self, id, lr, num_actions, observation_size, num_ops_per_action,
                 fc1_dims = 64, fc2_dims = 128, name = 'DQN'):
        super().__init__()

        self.name = name

        output_dims = num_ops_per_action**num_actions

        self.fc1 = nn.Linear(observation_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x))
        actions = self.fc3(x1)

        return actions

    def save_model(self, file_path):
        print('... saving',self.name,'...')
        T.save(self.state_dict(), file_path+'_'+self.name)

    def load_model(self, file_path):
        print('... loading', self.name, '...')
        self.load_state_dict(T.load(file_path+'_'+self.name))

############################################################################
# Communication Network for DDQN
############################################################################
class DDQNComms(nn.Module):
    def __init__(self, *, id = None, lr=None, observation_size=None, alphabet_size=None,
                 fc1_dims = 64, fc2_dims = 128, name = 'DDQN_COMMS_'):
        super().__init__()

        self.name = name+str(id)

        output_dims = alphabet_size

        #print('Comms Network has input size = ', observation_size, 'and output size = ', output_dims)

        self.fc1 = nn.Linear(observation_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x))
        raw_message = self.fc3(x1)
        return raw_message

    def save_model(self, file_path):
        print('... saving',self.name ,'...')
        T.save(self.state_dict(), file_path+'_'+self.name)

    def load_model(self, file_path):
        print('... loading',self.name,'...')
        self.load_state_dict(T.load(file_path+'_'+self.name))

############################################################################
# Action Network for DDQN
############################################################################
class DDQN(nn.Module):
    def __init__(self, *,id = None, lr = None, num_actions = None, observation_size = None,
                 num_ops_per_action = None, fc1_dims = 64, fc2_dims = 128, name = 'DDQN'):
        super().__init__()

        self.name = name

        output_dims = (num_ops_per_action**num_actions)
        #print('DQN network observation_size = ', observation_size, 'and output size = ', output_dims)

        self.fc1 = nn.Linear(observation_size, fc1_dims)
        #self.bn1 = nn.BatchNorm1d(fc1_dims)
        #self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        #self.bn2 = nn.BatchNorm1d(fc2_dims)
        #self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        #l1=self.fc1(state)
        #act1=l1*(T.tanh(F.softplus(l1))) #mish
        #bn1=self.bn1(act1)
        #dp1=self.dp1(bn1)
        x1 = F.relu(self.fc2(x))
        #l2=self.fc2(act1)
        #act2=l2*(T.tanh(F.softplus(l2))) #mish
        #bn2=self.bn2(act2)
        #dp2=self.dp2(bn2)
        #actions = self.fc3(act2)
        actions = self.fc3(x1)
        return actions

    def save_model(self, file_path):
        print('... saving', self.name, '...')
        T.save(self.state_dict(), file_path+'_'+self.name)

    def load_model(self, file_path):
        print('... loading', self.name, ' ...')
        self.load_state_dict(T.load(file_path))

############################################################################
# Actor Network for DDPG
############################################################################
class DDPGActorNetwork(nn.Module):
    def __init__(self, id, num_actions, observation_size, lr, name, min_max_action = 1):
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

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.name = name+'_'+str(id)+'_DDPG'

        self.to(self.device)


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
        print('... saving', self.name,'...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, '...', path)
        self.load_state_dict(T.load(path + '_' + self.name))

############################################################################
# Recurrent Layer for Environment Encoder
############################################################################
class EnvironmentEncoder(nn.Module):
    def __init__(self, observation_size, hidden_size, meta_param_size, batch_size, num_layers, lr):
        super(EnvironmentEncoder, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.observation = observation_size
        self.hidden_size = hidden_size
        self.meta_param_size = meta_param_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.ee = nn.LSTM(observation_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.meta_layer = nn.Linear(hidden_size, meta_param_size)

        self.ee_optimizer = optim.Adam(self.ee.parameters(), lr=lr, weight_decay= 1e-4)

        self.to(self.device)

    def forward(self, observation, choose_action = False):
        hidden0 = (T.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device), T.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))
        lstm_out , (h_out, _) = self.ee(observation,hidden0)
        lstm_out = h_out.view(-1,self.hidden_size)
        meta_parameters = self.meta_layer(lstm_out)
        meta_parameters = T.relu(meta_parameters)
        if choose_action:
            meta_parameters = meta_parameters[-meta_param_size:]
        return meta_parameters

    def save_checkpoint(self, path):
        print('... saving', self.name,'...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, '...', path)
        self.load_state_dict(T.load(path + '_' + self. name))
        
############################################################################
# Critic Network for DDPG
############################################################################
class DDPGCriticNetwork(nn.Module):
    def __init__(self, id, num_actions, observation_size, lr, name):
        super().__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.fc1_dims = 400
        self.fc2_dims = 300
        self.fc1 = nn.Linear(observation_size+num_actions, self.fc1_dims) #! 1 represents the action number. Change this to num_actions when we move to continuous
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.relu = nn.ReLU()
        self.init_weights(3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.name = name+'_'+str(id)+'_DDPG'
        self.to(self.device)

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
        print('... saving', self.name,'...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, '...', path)
        self.load_state_dict(T.load(path + '_' + self. name))

############################################################################
# Actor Network for TD3
############################################################################
class TD3ActorNetwork(nn.Module):
    def __init__(self, id, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, min_max_action = 1):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.min_max_action = min_max_action
        self.name = name +'_'+str(id)+'_TD3'


        self.fc1 = nn.Linear(input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha, weight_decay = 1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.min_max_action * T.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self, path):
        print('... saving', self.name,'...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, '...')
        self.load_state_dict(T.load(path + '_' + self.name))

############################################################################
# Critic Network for TD3
############################################################################
class TD3CriticNetwork(nn.Module):
    def __init__(self, id, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name +'_'+str(id)+'_TD3'



        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta, weight_decay = 1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = F.relu(self.fc1(T.cat([state, action[:,:2]], dim = 1))) #Remove [:,:2] from actions if grippers action needed as input
        q1_action_value = F.relu(self.fc2(q1_action_value))
        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self, path):
        print('... saving', self.name,'...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, '...')
        self.load_state_dict(T.load(path + '_' + self.name))
