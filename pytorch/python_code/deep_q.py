import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .replay_buffer import ReplayBuffer

class GenericNetwork(nn.Module):
    def __init__(self, *,learning_rate=None, input_size=None, output_dims=None):
        super().__init__()
                
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dims)

        self.optimizer = optim.RMSprop(self.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x))
        results = self.fc3(x1)
        return results

    def save_model(self, file_path):
        print('... saving model ...')
        T.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        print('... loading model ...')
        self.load_state_dict(T.load(file_path))

class DeepQ:
    def __init__(self, *,learning_rate=None, input_size=None, output_dims=None, batch_size=None,
                 buffer_size=100000, epsilon=None, gamma=None):
        nn_args = {'learning_rate':learning_rate, 'input_size':input_size, 'output_dims'output_dims)}
        self.eval_net = GenericNetwork(**nn_args)
        self.target_net = GenericNetwork(**nn_args)
        self.memory = ReplayBuffer(buffer_size, input_size)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0: #!!! Import
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.eval_net.device)
        actions = T.tensor(action).to(self.eval_net.device)
        rewards = T.tensor(reward).to(self.eval_net.device)
        states_ = T.tensor(new_state).to(self.eval_net.device)
        dones = T.tensor(done).to(self.eval_net.device)

        return states, actions, rewards, states_, dones

    def doubleQLearn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        self.eval_net.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.eval_net.forward(states)[indices, actions]

        target_net = self.target_net.forward(states_)
        eval_net = self.eval_net.forward(states_)

        max_actions = T.argmax(eval_net, dim=1)

        target_net[dones] = 0.0

        q_target = rewards + self.gamma*target_net[indices, max_actions]

        loss = self.eval_net.loss(q_target, q_pred).to(self.eval_net.device)
        loss.backward()
        self.eval_net.optimizer.step()
        self.learn_step_counter+=1

        self.decrement_epsilon()

    def save_model(self, path):
        self.eval_net.save_model(path)

    def load_model(self, path):
        self.eval_net.load_model(path)
