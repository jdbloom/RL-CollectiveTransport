from .deep_q_network import DeepQNetwork
from .replay_buffer import ReplayBuffer

from collections import namedtuple
import numpy as np
import torch as T
from .mailbox import Mailbox
import math


class Agent_DQN():
    def __init__(self, num_agents, num_observations, num_actions,
                 num_ops_per_action, id, comm_scheme="None"):

        self.id = id

        self.num_agents = num_agents
        self.num_actions = num_actions
        self.num_ops_per_action = num_ops_per_action
        self.num_observations = num_observations

        self.alphabet_size = 8
        self.alphabet_space = [i for i in range(self.alphabet_size)]
        self.dead_channel_code = self.alphabet_size
        if comm_scheme == 'None':
            self.left_contacts = {i:[] for i in range(num_agents)}
            self.right_contacts = self.left_contacts
        elif comm_scheme == 'left':
            self.left_contacts = {i:[i+1] for i in range(num_agents)}
            self.left_contacts[num_agents - 1] = [0]
            self.right_contacts = {i:[] for i in range(num_agents)}
        elif comm_scheme == 'right':
            self.right_contacts = {i:[i-1] for i in range(num_agents)}
            self.right_contacts[0] = [num_agents - 1]
            self.left_contacts = {i:[] for i in range(num_agents)}
        elif comm_scheme == 'neighbors':
            self.left_contacts = {i:[i+1] for i in range(num_agents)}
            self.left_contacts[num_agents - 1] = [0]
            self.right_contacts = {i:[i-1] for i in range(num_agents)}
            self.right_contacts[0] = [num_agents - 1]
        else:
            raise Exception('Unknown comm_scheme ' + comm_scheme)
            
        # An iterable describing who may contact who. Entries in format {sender: [receivers]}
        # Merge left and right contacts into a master dictionary
        self.contacts = {key:val+self.right_contacts[key] for (key,val) in self.left_contacts.items()}
        
        self.mailbox = Mailbox(self.contacts)
        
        self.gamma = 0.99997
        self.lr = 0.0001
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-6

        self.batch_size = 32

        self.replace_target_cnt = 1000

        self.action_space = [i for i in range(self.num_ops_per_action**self.num_actions)]

        # If the robot is failed, it will always perform it's "failure_action"
        self.failed = False
        self.failure_action = [0, 0, 1] #failure action (wheel increases dont matter, failure code is in buzz)

        self.learn_step_counter = 0

        self.memory = ReplayBuffer(100000, num_observations) # accounting for reward in msg

        nn_args = [self.lr, self.num_actions, self.num_observations, self.num_ops_per_action, self.alphabet_size]
        self.q_eval = DeepQNetwork(*nn_args)
        self.q_next = DeepQNetwork(*nn_args)

    def choose_action(self, observation, test):
        # The last observation indicates whether the robot has failed or not
        if observation[-1] != 0:
            self.failed = True
            return self.failure_action, 9, self.dead_channel_code
        else: self.failed = False

        if test or np.random.random() > self.epsilon:
            state = T.tensor([observation[:-1]], dtype = T.float).to(self.q_eval.device) #Need the [:-1] to strip the failure flag
            actions = self.q_eval.forward(state)
            action = T.argmax(actions[0][:10]).item()
            outgoing_message = T.argmax(actions[0][10:]).item()
        else:
            action = np.random.choice(self.action_space)
            outgoing_message = np.random.choice(self.alphabet_space)

        actions = self.parse_action(action)

        return actions, action, outgoing_message

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
        l_wheel = (math.floor(action_num/self.num_ops_per_action) - 1)/10.0
        r_wheel = (action_num%self.num_ops_per_action - 1)/10.0
        # Trailing zero is hardcoded control for gripper
        return np.array([l_wheel, r_wheel, 0], dtype=np.float32)

    def get_agent_incoming_communications(self, agent_id):
        '''
        Returns the specified agent's communications as a namedtuple
        (left_comm, right_comm)
        '''
        messages = self.mailbox.inbox[agent_id]
        incoming_comms = namedtuple('incoming_comms', 'left_comm right_comm')
        left_comm = self.dead_channel_code
        right_comm = self.dead_channel_code
        for message in messages:
            if agent_id in self.left_contacts[message.sender]:
                left_comm = message.contents
            elif agent_id in self.right_contacts[message.sender]:
                right_comm = message.contents
        return incoming_comms(left_comm, right_comm)

    def clear_agent_inbox(self, agent_id):
        self.mailbox.clear_inbox(agent_id)

    def schedule_message_to_all_contacts(self, sender, contents):
        '''
        The sender sends a message containing contents to everyone it can contact
        '''
        for receiver in self.mailbox.contacts[sender]:
            self.mailbox.schedule_message(sender, receiver, contents)

    def carry_mail(self):
        self.mailbox.carry_mail()

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = T.LongTensor(np.arange(self.batch_size))
        q_pred = self.q_eval.forward(states)[indices, actions]

        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def doubleQLearn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]

        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter+=1

        self.decrement_epsilon()
        #return loss.item()

    def save_model(self, path):
        self.q_eval.save_model(path)

    def load_model(self, path):
        self.q_eval.load_model(path)
