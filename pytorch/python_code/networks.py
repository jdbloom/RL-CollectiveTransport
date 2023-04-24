import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
import random
from collections import deque
import math

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

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))

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

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))

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

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))

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

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))

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
        self.name = "Enviroment_Encoder"
        self.to(self.device)

    def forward(self, observation, choose_action = False):
        hidden0 = (T.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device), T.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))
        lstm_out , (h_out, _) = self.ee(observation,hidden0)
        lstm_out = h_out.view(-1,self.hidden_size)
        meta_parameters = self.meta_layer(lstm_out)
        meta_parameters = T.relu(meta_parameters)
        if choose_action:
            meta_parameters = meta_parameters[-1]
        return meta_parameters

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))
        
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

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))

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

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))

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

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(self.heads*self.head_dim, embed_size)

        self.softmax = nn.Softmax(dim = 3)

    def forward(self, values, keys, query, mask=None):
        # mask is only for the decoder
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = T.einsum("nqhd, nkhd->nhqk", [queries, keys])
        # query shape = N, query_len, heads, heads_dim
        # key shape = N, key_len, heads, heads_dim
        # energy shape = N, heads, query_len, key_len
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = self.softmax(energy / (self.embed_size**(1/2)))

        out = T.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape = N, heads, querry_len, key_len
        # values shape = N, value_len, heads, heads_dim
        # out = N, query_len, heads, heads_dim, then flatten last two dims

        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        # mask is only for the decoder
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query)) # skip connection in encoder block
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x)) # skip connection in encoder block
        return out

class AttentionEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, max_length, src_vocab_size=None):
        super().__init__()
        # masked_length is the max length of a sequence, so for us it is however long we want our sequences to be when training 
        self.embed_size = embed_size
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.word_embedding = nn.Sequential(nn.Linear(6, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 1))
        self.position_embedding = nn.Embedding(max_length, embed_size) # We need this to propagate the causality
        self.fc_out = nn.Linear(embed_size*max_length, 1) # Transform to angle
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion,)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr = 0.0001  , weight_decay = 1e-4)
        self.to(self.device)

        self.name = 'Attention_Encoder'

    def forward(self, x, mask = None):
        N, seq_len, obs_size = x.shape
        positions = T.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.word_embedding(x) + self.position_embedding(positions) 
        
        for layer in self.layers:
            mp = layer(out, out, out, mask)
        out = self.fc_out(mp.view(N,-1))
        out = self.tanh(out) # converts to single number in range (-1, 1) to represent angle
        return out
    
    def save_checkpoint(self, path):
        print('... saving', self.name,'...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path):
        print('... loading', self.name, '...')
        self.load_state_dict(T.load(path + '_' + self.name))


############################################################################
# Graph Attention
############################################################################
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

class GAT_QNetwork(nn.Module):
    def __init__(self,in_channels,hidden_channels,num_actions,num_heads,num_robots):
        super(GAT_QNetwork,self).__init__()
        self.gat1 = GATConv(in_channels,hidden_channels,heads=num_heads)
        self.gat2 = GATConv(hidden_channels*num_heads,hidden_channels)
        self.fc = nn.Linear(hidden_channels,num_actions)

    def forward(self,x ,edge_index):
        x = x.view(-1, x.shape[-1])  # Flatten the first two dimensions
        x = F.relu(self.gat1(x,edge_index))
        x = F.dropout(x,p=0.6,training=self.training)
        x= F.relu(self.gat2(x,edge_index))
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.fc(x)
        #print("xxx shapelast:", x.shape)
        return x

class SharedGATDQNAgent:
    def __init__(self,in_channels,hidden_channels,num_actions,num_heads,num_robots,learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=32):
        self.qnetwork_local = GAT_QNetwork(in_channels,hidden_channels,num_actions,num_heads,num_robots).to(device)
        #print("self.qnetwork_local",self.qnetwork_local.summary())
        from torchsummary import summary
        self.qnetwork_target= GAT_QNetwork(in_channels,hidden_channels,num_actions,num_heads,num_robots).to(device)
        self.optimizer = T.optim.Adam(self.qnetwork_local.parameters(),lr=learning_rate)
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_robots = num_robots
        self.num_actions = num_actions
        #summary(self.qnetwork_local,(32,4,18))





    def step(self,joint_state,actions,rewards,next_joint_state,dones,edge_index):
        #print("actions in step",actions.shape)
        #actions = actions.view(-1, 1)
        self.memory.append((joint_state,actions,rewards,next_joint_state,dones,edge_index))
        if len(self.memory)>= self.batch_size:
            experiences = random.sample(self.memory,k=self.batch_size)
            loss = self.learn(experiences)
            return loss

    def parse_actions(self,action_num):
        options_per_action = 3
        n_actions = 2
        if action_num < 0 or action_num > options_per_action ** n_actions-1:
            raise Exception('Action Number Out of Range:' + str(action_num))
        l_wheel = round((math.floor(action_num / options_per_action) - 1) / 10.0, 1)
        r_wheel = round((action_num % options_per_action - 1) / 10.0, 1)
        return np.array([l_wheel, r_wheel, 0])

    def build_reward(self,cyl_dist_goal, prev_cyl_dist_goal, robot_positions, obstacle_positions, time_step,obj_pos,object_radius):
        threshold = 0.2
        object_threshold = 0.2
        progress_reward = prev_cyl_dist_goal - cyl_dist_goal

        if obstacle_positions != 0:
            obstacle_penalty = 0
            for position in robot_positions:
                for obstacle_position in obstacle_positions:
                    distance_to_obstacle = np.linalg.norm(position - obstacle_position)
                    if distance_to_obstacle < threshold:
                        obstacle_penalty -= (threshold - distance_to_obstacle)

            for obstacle_position in obstacle_positions:
                distance_to_obstacle = np.linalg.norm(obj_pos - obstacle_position) - object_radius
                if distance_to_obstacle < object_threshold:
                    obstacle_penalty -= (object_threshold - distance_to_obstacle)

        time_penalty = -1

        if obstacle_positions == 0:
            total_reward = progress_reward + time_penalty
        else:
            total_reward = progress_reward + obstacle_penalty + time_penalty


        #total_reward = progress_reward + obstacle_penalty + time_penalty
        return total_reward

    def act(self, joint_state, edge_index, eps=0.):
        #print("joint_state shape:", joint_state.shape)
        if random.random() > eps:
            joint_state = T.tensor(joint_state, dtype=T.float).to(device)
            self.qnetwork_local.eval()
            with T.no_grad():
                action_values = self.qnetwork_local(joint_state, edge_index)
            self.qnetwork_local.train()
            return action_values.cpu().data.numpy()
        else:
            random_action_numbers = np.random.randint(0, self.num_actions, size=self.num_robots)

            # Convert random action numbers to one-hot encoded action_values
            random_action_values = np.eye(self.num_actions)[random_action_numbers]
            return random_action_values

    

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, edge_indices = zip(*experiences)
        
        #print("States tuple:", np.shape(states))

        # for idx, state in enumerate(states):
        #     print(f"State {idx} shape: {state.shape}, type: {state.dtype}")

        # for idx, action in enumerate(actions):
        #     print(f"Action {idx} shape: {action.shape}, type: {action.dtype}")

        states = T.stack([T.tensor(state, dtype=T.float).to(device) for state in states], dim=0)

        #print("actionsbef", np.shape(actions))
        actions = T.tensor(actions, dtype=T.long).view(-1, self.num_actions).to(device)


        rewards = T.tensor(rewards, dtype=T.float).to(device)
        next_states = T.stack([T.tensor(next_state, dtype=T.float).to(device) for next_state in next_states], dim=0)
        dones = T.tensor(dones, dtype=T.float).to(device)
        edge_indices = T.stack(edge_indices, dim=0).view(2, -1).to(device)


        #print("edge_indicesss", edge_indices.shape)
        #print("statesss", states.shape)
        #print("actions", actions.shape)

        # Get max predicted Q values (for next states) from target model

        Q_targets_next = self.qnetwork_target(next_states, edge_indices).detach().max(1)[0].unsqueeze(1)

        print("Q_targets_next",Q_targets_next.shape)

        # Compute Q targets for current states
        rewards = rewards.view(-1, 1).repeat(self.num_robots, 1)
        #print("rewards",rewards.shape)

        dones = dones.view(-1, 1).repeat(self.num_robots, 1) 

        # print("rewards",rewards.shape)

        # print("dones",dones.shape)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        #print("Q_targets",Q_targets.shape)

        Q_expected = self.qnetwork_local(states, edge_indices).gather(1, actions)

        #print("Q_expected",Q_expected.shape)


        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)


    def soft_update(self,local_model,target_model,tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)











    
