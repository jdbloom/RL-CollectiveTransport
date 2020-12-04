import numpy as np
import torch as T
import torch.nn as nn
from torch.optim import Adam
from collections import namedtuple
import math

from .networks_torch import ActorNetwork, CriticNetwork
from .deep_q_comms import DeepQComms
from .replay_buffer import ReplayBuffer
from .mailbox import Mailbox


Loss = nn.MSELoss()


class Agent_DDPG:
    def __init__(self, num_agents, num_observation, num_actions,
                 num_ops_per_action, id, comm_scheme="None",
                 alphabet_size=4, min_max_action=1, alpha=0.001,
                 beta=0.002, lr=0.0001, gamma=0.99, max_size=1000000,
                 tau=0.005, batch_size=64, noise=0.1):

        self.id = id

        self.num_agents = num_agents
        self.num_ops_per_action = num_ops_per_action
        self.alphabet_size = alphabet_size
        self.init_comm_scheme(comm_scheme, self.num_agents)

        # An iterable describing who may contact who. Entries in format {sender: [receivers]}
        # Merge left and right contacts into a master dictionary
        self.contacts = {key: val+self.right_contacts[key]
                         for (key, val) in self.left_contacts.items()}

        self.mailbox = Mailbox(self.contacts, self.dead_channel_code)

        self.gamma = gamma
        self.tau = tau
        # ! We will need to add in the number of actions when we switch to continuous
        self.memory = ReplayBuffer(max_size, num_observation)
        self.comms_memory = ReplayBuffer(max_size, num_observation + 2*alphabet_size)
        self.batch_size = batch_size
        self.noise = noise
        # sets the +/- bounds for the action space: EX. 1 -> (-1, 1)
        self.min_max_action = min_max_action

        self.init_networks(num_observation, num_actions, num_ops_per_action, alpha, beta, lr)

        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-6

        self.action_space = [i for i in range(num_ops_per_action**num_actions)]

        # If the robot is failed, it will always perform it's "failure_action"
        self.failed = False
        # failure action (wheel increases dont matter, failure code is in buzz)
        self.failure_action = [0, 0, 1]
        self.failure_action_code = len(self.action_space)

        self.learn_step_counter = 0
        self.replace_target_cnt = 1000



    def init_comm_scheme(self, comm_scheme, num_agents):
        self.alphabet_space = [np.zeros(self.alphabet_size) for i in range(self.alphabet_size+1)]
        for i in range(self.alphabet_size):
            self.alphabet_space[i+1][i] = 1

        self.dead_channel_code = self.alphabet_space[0]
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

    def init_networks(self, num_observation, num_actions, num_ops_per_action, alpha, beta, lr):
        self.actor = ActorNetwork(num_actions = num_actions, observation_size = num_observation, num_ops_per_action = num_ops_per_action)
        self.target_actor = self.actor = ActorNetwork(num_actions = num_actions, observation_size = num_observation, num_ops_per_action = num_ops_per_action)
        self.actor_optimizer = Adam(self.actor.parameters(),lr = alpha)

        self.critic = CriticNetwork(num_actions = num_actions, observation_size = num_observation)
        self.target_critic = CriticNetwork(num_actions = num_actions, observation_size = num_observation)
        self.critic_optimizer = Adam(self.critic.parameters(),lr = beta)

        self.update_network_parameters(tau = 1)

        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

        comms_nn_args = {'lr':lr, 'observation_size':num_observation, 'alphabet_size':self.alphabet_size}
        self.q_comms_eval = DeepQComms(**comms_nn_args)
        self.q_comms_next = DeepQComms(**comms_nn_args)

    def update_network_parameters(self, tau = None):
        # If tau = 1 -> hard update (should only be done during init)
        if tau is None:
            tau = self.tau
        # Update Actor Network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        # Update Critic Network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def DDPGlearn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        '''
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.actor.device)
        actions = T.tensor(action).to(self.actor.device)
        rewards = T.tensor(reward).to(self.actor.device)
        states_ = T.tensor(new_state).to(self.actor.device)
        dones = T.tensor(done).to(self.actor.device)
        '''
        self.learn_step_counter += 1

        states, actions, rewards, states_, dones = self.sample_memory()
        target_actions = T.unsqueeze(T.argmax(self.target_actor(states_), 1), 1) #! remove argmax for continuous actions
        q_value_ = self.target_critic([states_, target_actions])
        q_value_[dones] = 0.0
        target = T.unsqueeze(rewards, 1) + self.gamma*q_value_

        # Critic Update
        self.critic.zero_grad()
        actions = T.unsqueeze(actions, 1)
        q_value = self.critic([states, actions])
        value_loss = Loss(q_value, target)
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        self.actor.zero_grad()
        new_policy_actions = T.unsqueeze(T.argmax(self.actor(states), 1), 1)
        actor_loss = -self.critic([states, new_policy_actions])
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()

        self.learn_step_counter += 1

    def doubleQLearnComms(self):
        if self.comms_memory.mem_ctr < self.batch_size:
            return
        self.q_comms_eval.optimizer.zero_grad()

        self.replace_target_comms_network()

        states, message, rewards, states_, dones = self.sample_comms_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.q_comms_eval.forward(states)[indices, message]

        q_next = self.q_comms_next.forward(states_)
        q_comms_eval = self.q_comms_eval.forward(states_)

        max_actions = T.argmax(q_comms_eval, dim=1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_comms_eval.loss(q_target, q_pred).to(self.q_comms_eval.device)
        loss.backward()
        self.q_comms_eval.optimizer.step()
        #self.learn_step_counter+=1

        self.decrement_epsilon()
        #return loss.item()

    def choose_action(self, observation, failure, evaluate = False):
        if failure:
            self.failed = True
            return self.failure_action, self.failure_action_code
        else: self.failed = False

        if evaluate or np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype = T.float).to(self.actor.device)
            actions = self.actor(state)
            action = T.argmax(actions[0]).item()
        else:
            action = np.random.choice(self.action_space)

        actions = self.parse_action(action)


        # CONTINUOUS ACTION SPACE!!!
        '''
        state = T.tensor([observation], dtype = T.float).to(self.q_eval.device)
        actions = self.actor(state)
        if not evaluate:
            # need to add noise to allow for exploration
            actions += [T.normal(mean = 0.0, stddev=self.noise) for n in self.num_actions]
        actions = np.clip(actions, -self.min_max_action, self.min_max_action)

        return actions[0]
        '''

        return actions, action

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

    def choose_message(self, observation, failure, test):
        if failure:
            self.failed = True
            return self.dead_channel_code
        else:
            self.failed = False
        if test or np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype = T.float).to(self.actor.device)
            messages = self.q_comms_eval.forward(state)
            outgoing_message_code = T.argmax(messages[0]).item()
            outgoing_message = self.alphabet_space[outgoing_message_code + 1]
            # Alphabet_space[0] is the dead_channel_code
        else:
            outgoing_message_code = np.random.choice(self.alphabet_size) + 1
            outgoing_message = self.alphabet_space[outgoing_message_code]

        return outgoing_message, outgoing_message_code

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

    def store_comms_transition(self, state, message, reward, state_, done):
        self.comms_memory.store_transition(state, message, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def replace_target_comms_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_comms_next.load_state_dict(self.q_comms_eval.state_dict())

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.actor.device)
        actions = T.tensor(action).to(self.actor.device)
        rewards = T.tensor(reward).to(self.actor.device)
        states_ = T.tensor(new_state).to(self.actor.device)
        dones = T.tensor(done).to(self.actor.device)

        return states, actions, rewards, states_, dones

    def sample_comms_memory(self):
        state, action, reward, new_state, done = self.comms_memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_comms_eval.device)
        actions = T.tensor(action).to(self.q_comms_eval.device)
        rewards = T.tensor(reward).to(self.q_comms_eval.device)
        states_ = T.tensor(new_state).to(self.q_comms_eval.device)
        dones = T.tensor(done).to(self.q_comms_eval.device)

        return states, actions, rewards, states_, dones

    def load_weights(self, output):
        if output is None: return
        self.actor.load_state_dict(T.load('actor.pkl'.format(output)))
        self.critic.load_state_dict(T.load('critic.pkl'.format(output)))


    def save_model(self, output):
        T.save(self.actor.state_dict(),output+'_actor.pkl')
        T.save(self.critic.state_dict(), output+'_critic.pkl')
