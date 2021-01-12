from .networks import DDQN, DDQNComms, DQN, DDPGActorNetwork, DDPGCriticNetwork, TD3ActorNetwork, TD3CriticNetwork
from .replay_buffer import ReplayBuffer
from .communications import Mailbox

import numpy as np
import math
from collections import namedtuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

Loss = nn.MSELoss()

class Agent():
    def __init__(self, num_agents, num_observations, num_actions,
                 num_ops_per_action, id, learning_scheme, comms_scheme = "None",
                 alphabet_size=4, min_max_action = 1):
        self.id = id
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.num_ops_per_action = num_ops_per_action
        self.num_observations = num_observations
        self.alphabet_size = alphabet_size
        self.comms_scheme = comms_scheme
        self.learning_scheme = learning_scheme
        self.min_max_action = min_max_action

        self.make_comms_scheme()
        self.make_learning_scheme()


    def make_comms_scheme(self):
        print('######## COMMUNICATION SCHEME ########')
        print('SCHEME:', self.comms_scheme)
        self.alphabet_space = [np.zeros(self.alphabet_size) for i in range(self.alphabet_size + 1)]
        for i in range(self.alphabet_size):
            self.alphabet_space[i+1][i] = 1
        self.dead_channel_message = self.alphabet_space[0]
        self.dead_channel_code = -1
        obs_size = self.num_observations+2*self.alphabet_size

        if self.comms_scheme == 'None':
            self.left_contacts = {i:[] for i in range(self.num_agents)}
            self.right_contacts = self.left_contacts
        elif self.comms_scheme == 'Left':
            self.left_contacts = {i:[i+1] for i in range(self.num_agents)}
            self.left_contacts[self.num_agents-1] = [0]
            self.right_contacts = {i:[] for i in range(self.num_agents)}
        elif self.comms_scheme == 'Right':
            self.left_contacts = {i:[] for i in range(self.num_agents)}
            self.right_contacts = {i:[i-1] for i in range(self.num_agents)}
            self.right_contacts[0] = [self.num_agents - 1]
        elif self.comms_scheme == 'Neighbors':
            self.left_contacts = {i:[i+1] for i in range(self.num_agents)}
            self.left_contacts[self.num_agents-1] = [0]
            self.right_contacts = {i:[i-1] for i in range(self.num_agents)}
            self.right_contacts[0] = [self.num_agents - 1]
        else:
            raise Exception('Unknown Communication Scheme ' + self.comms_scheme)

        if self.comms_scheme != 'None':
            print('MODEL SPECIFICS:')
            comms_nn_args = {'lr':self.lr, 'observation_size':obs_size, 'alphabet_size':self.alphabet_size}
            self.q_comms_eval = DDQNComms(**comms_nn_args)
            self.q_comms_next = DDQNComms(**comms_nn_args)
            print('Communication Network:')
            print(self.q_comms_eval)
            print(self.q_comms_next)
            self.comms_memory = ReplayBuffer(100000, self.num_observations + 2*self.alphabet_size, 1, 'Discrete')

        self.contacts = {key:val+self.right_contacts[key] for (key,val) in self.left_contacts.items()}
        self.mailbox = Mailbox(self.contacts, self.dead_channel_code)

    def get_agent_incoming_communications(self, agent_id):
        messages = self.mailbox.inbox[agent_id]
        incoming_comms = namedtuple('incoming_comms', 'left_msg right_msg')
        left_msg = self.dead_channel_message
        right_msg = self.dead_channel_message
        for message in messages:
            if agent_id in self.left_contacts[message.sender]:
                if message.contents != -1:
                    left_msg = self.alphabet_space[message.contents]
            elif agent_id in self.right_contacts[message.sender]:
                if message.contents != -1:
                    right_msg = self.alphabet_space[message.contents]

        msg = incoming_comms(left_msg, right_msg)
        return msg

    def make_agent_state(self, env_obs, agent_id):
        if self.comms_scheme == 'None':
            return np.concatenate((env_obs, self.dead_channel_message, self.dead_channel_message))
        msg = self.get_agent_incoming_communications(agent_id)
        #import ipdb; ipdb.set_trace()
        agent_state = np.concatenate((env_obs, msg.left_msg, msg.right_msg))
        return agent_state

    def clear_agent_inbox(self, agent_id):
        self.mailbox.clear_inbox(agent_id)

    def schedule_message_to_all_contacts(self, sender, contents):
        for receiver in self.mailbox.contacts[sender]:
            self.mailbox.schedule_message(sender, receiver, contents)

    def carry_mail(self):
        self.mailbox.carry_mail()


    def make_learning_scheme(self):
        # These are the hyper parameters relating to the neural networks
        self.action_space = [i for i in range(self.num_ops_per_action**self.num_actions)]
        self.gamma = 0.99997
        self.tau = 0.005
        self.alpha = 0.001
        self.beta = 0.002
        self.lr = 0.0001
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-6
        self.batch_size = 64
        self.replace_target_cnt = 1000
        self.failed = False
        self.failure_action = [0, 0, 1]
        self.failure_action_code = len(self.action_space)
        self.learn_step_counter = 0
        self.noise = 0.1
        self.update_actor_iter = 2
        self.warmup = 1000
        self.time_step = 0
        print('######## LEARNING SCHEME ########')
        print('SCHEME:', self.learning_scheme)
        print('MODEL SPECIFICS:')
        if self.learning_scheme == 'None':
            # define logic for no learning?
            return

        if self.learning_scheme == 'DQN':
            self.make_DQN()

        elif self.learning_scheme == 'DDQN':
            self.make_DDQN()

        elif self.learning_scheme == 'DDPG':
            self.make_DDPG()

        elif self.learning_scheme == 'TD3':
            self.make_TD3()

        else:
            raise Exception('Unknown Learning Scheme' + self.learning_scheme)

    def make_DQN(self):
        #define networks for DQN
        self.min_max_action = 0.1
        obs_size = self.num_observations + 2*self.alphabet_size
        actions_nn_args = {'lr':self.lr, 'num_actions':self.num_actions, 'observation_size':obs_size,
                   'num_ops_per_action':self.num_ops_per_action}
        comms_nn_args = {'lr':self.lr, 'observation_size':obs_size, 'alphabet_size':self.alphabet_size}

        self.q_eval = DQN(**actions_nn_args)
        self.q_next = DQN(**actions_nn_args)
        print(self.q_eval)
        print(self.q_next)

        self.memory = ReplayBuffer(100000, obs_size, 1, 'Discrete')

    def make_DDQN(self):
        self.min_max_action = 0.1
        obs_size = self.num_observations + 2*self.alphabet_size
        actions_nn_args = {'lr':self.lr, 'num_actions':self.num_actions, 'observation_size':obs_size,
                   'num_ops_per_action':self.num_ops_per_action}

        self.q_eval = DDQN(**actions_nn_args)
        self.q_next = DDQN(**actions_nn_args)
        print(self.q_eval)
        print(self.q_next)

        self.memory = ReplayBuffer(100000, self.num_observations + 2*self.alphabet_size, 1, 'Discrete')

    def make_DDPG(self):
        #define networks for DDPG
        self.min_max_action = 1
        obs_size = self.num_observations + 2*self.alphabet_size
        actor_nn_args = {'num_actions':self.num_actions, 'observation_size':obs_size,
                         'num_ops_per_action':self.num_ops_per_action, 'min_max_action':self.min_max_action}

        self.actor = DDPGActorNetwork(**actor_nn_args)
        self.target_actor = DDPGActorNetwork(**actor_nn_args)
        self.actor_optimizer = Adam(self.actor.parameters(), lr = self.lr)
        print(self.actor)
        print(self.target_actor)
        self.critic = DDPGCriticNetwork(**critic_nn_args)
        self.target_critic = DDPGCriticNetwork(**critic_nn_args)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = self.lr)
        print(self.critic)
        print(self.target_critic)

        self.update_network_parameters(tau = 1)

        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

        self.memory = ReplayBuffer(100000, self.num_observations + 2*self.alphabet_size, self.num_actions, 'Continuous')

    def make_TD3(self):
        #difine networks for TD3
        self.min_max_action = 1
        obs_size = self.num_observations + 2*self.alphabet_size
        self.actor = TD3ActorNetwork(self.alpha, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'actor')
        self.target_actor = TD3ActorNetwork(self.alpha, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'target_actor')
        print(self.actor)
        print(self.target_actor)
        self.critic_1 = TD3CriticNetwork(self.beta, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'critic_1')
        self.target_critic_1 = TD3CriticNetwork(self.beta, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'target_critic_1')
        print(self.critic_1)
        print(self.target_critic_1)
        self.critic_2 = TD3CriticNetwork(self.beta, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'critic_2')
        self.target_critic_2 = TD3CriticNetwork(self.beta, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'target_critic_2')
        print(self.critic_2)
        print(self.target_critic_2)

        self.update_network_parameters(tau = 1)

        self.memory = ReplayBuffer(100000, self.num_observations + 2*self.alphabet_size, self.num_actions, 'Continuous')

    def update_network_parameters(self, tau = None):
        # If tau = 1 -> hard update (should only be done during init)
        if self.learning_scheme == 'DDPG':
            if tau is None:
                tau = self.tau
            # Update Actor Network
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            # Update Critic Network
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        elif self.learning_scheme == 'TD3':
            if tau is None:
                tau = self.tau

            actor_params = self.actor.named_parameters()
            critic_1_params = self.critic_1.named_parameters()
            critic_2_params = self.critic_2.named_parameters()
            target_actor_params = self.target_actor.named_parameters()
            target_critic_1_params = self.target_critic_1.named_parameters()
            target_critic_2_params = self.target_critic_2.named_parameters()

            critic_1 = dict(critic_1_params)
            critic_2 = dict(critic_2_params)
            actor = dict(actor_params)
            target_actor = dict(target_actor_params)
            target_critic_1 = dict(target_critic_1_params)
            target_critic_2 = dict(target_critic_2_params)

            for name in critic_1:
                critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()

            for name in critic_2:
                critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()

            for name in actor:
                actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()

            self.target_critic_1.load_state_dict(critic_1)
            self.target_critic_2.load_state_dict(critic_2)
            self.target_actor.load_state_dict(actor)
        else:
            print('UNKNOWN NETWORK UPDATE RULE')
            return

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def replace_target_comms_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_comms_next.load_state_dict(self.q_comms_eval.state_dict())

    def choose_action(self, observation, failure, test = False):
        if failure:
            self.failed = True
            return self.failure_action, self.failure_action_code
        else:
            self.failed = False

        if self.learning_scheme == 'None':
            # Not sure what to do here for no learning
            return self.failure_action, self.failure_action_code

        if self.learning_scheme == 'DQN':
            return self.DQN_choose_action(observation, test)

        elif self.learning_scheme == 'DDQN':
            return self.DDQN_choose_action(observation, test)

        elif self.learning_scheme == 'DDPG':
            return self.DDPG_choose_action(observation, test)

        elif self.learning_scheme == 'TD3':
            return self.TD3_choose_action(observation, test)

    def DQN_choose_action(self, observation, test = False):
        if test or np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype = T.float).to(self.q_eval.device)
            action_values = self.q_eval.forward(state)
            action = T.argmax(action_values[0]).item()
        else:
            action = np.random.choice(self.action_space)
        actions = self.parse_action(action)
        return (actions, action)

    def DDQN_choose_action(self, observation, test = False):
        if test or np.random.random() > self.epsilon:
            state = T.tensor([observatoin], dtype = T.float).to(self.q_eval.device)
            action_values = self.q_eval.forward(state)
            action = T.argmax(actions[0]).item()
        else:
            action = np.random.choice(self.action_space)
        actions = self.parse_action(action)
        return (actions, action)

    def DDPG_choose_action(self, observation, test = False):
        state = T.tensor([observation], dtype = T.float).to(self.actor.device).unsqueeze(0)
        actions = self.actor(state)
        if not test:
            actions += T.normal(0.0, self.noise, size = (1, self.num_actions)).to(self.actor.device)
        actions = T.clip(actions, -self.min_max_action, self.min_max_action)
        gripper = np.zeros((1, 1))
        actions = np.append(actions[0].cpu().detach().numpy(), gripper)
        return (actions, None)

    def TD3_choose_action(self,observation, test = False):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale = self.noise, size = (self.num_actions,))).to(self.actor.device    )
        else:
            state = T.tensor([observation], dtype = T.float).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale = self.noise), dtype = T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, -self.min_max_action, self.min_max_action)
        self.time_step += 1
        gripper = np.zeros((1, 1))
        actions = np.append(mu_prime.cpu().detach().numpy(), gripper)
        return (actions, None)

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
        if action_num < 0 or action_num >=self.num_ops_per_action**self.num_actions:
            raise Exception('Action Number Out of Range:'+str(action_num))
        l_wheel = round((math.floor(action_num/self.num_ops_per_action) - 1)/10.0, 1)
        r_wheel = round((action_num%self.num_ops_per_action - 1)/10.0, 1)
        # Trailing zero is hardcoded control for gripper
        return np.array([l_wheel, r_wheel, 0])

    def choose_message(self, state, failure, test = False):
        if failure:
            self.failed = True
            return self.dead_channel_message, self.dead_channel_code
        else:
            self.failed = False
        if test or np.random.random() > self.epsilon:
            state = T.tensor([state], dtype = T.float).to(self.q_comms_eval.device)
            messages = self.q_comms_eval.forward(state)
            outgoing_message_code = T.argmax(messages[0]).item()
            outgoing_message = self.alphabet_space[outgoing_message_code + 1]
        else:
            outgoing_message_code = np.random.choice(self.alphabet_size)
            outgoing_message = self.alphabet_space[outgoing_message_code + 1]
        return outgoing_message, outgoing_message_code

    def learn(self):
        if self.learning_scheme == 'None':
            return
        elif self.learning_scheme == 'DQN':
            self.DQN_learn()
        elif self.learning_scheme == 'DDQN':
            self.DDQN_learn()
        elif self.learning_scheme == 'DDPG':
            self.DDPG_learn()
        elif self.learning_scheme == 'TD3':
            self.TD3_learn()

    def DQN_learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory(network = self.q_eval)

        indices = T.LongTensor(np.arange(self.batch_size).astype(np.long))

        q_pred = self.q_eval(states)[indices, actions]

        q_next = self.q_next(states_).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def DDQN_learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory(network = self.q_eval)

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval(states)[indices, actions]

        q_next = self.q_next(states_)
        q_eval = self.q_eval(states_)

        max_actions = T.argmax(q_eval, dim = 1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def DDPG_learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.sample_memory(network = self.actor)
        target_actions = self.target_actor(states_)
        q_value_ = self.target_critic([states_, target_actions])
        q_value_[dones] = 0.0
        target = T.unsqueeze(rewards, 1) + self.gamma*q_value_

        #Critic Update
        self.critic.zero_grad()
        q_value = self.critic([states, actions])
        value_loss = Loss(q_value, target)
        value_loss.backward()
        self.critic_optimizer.step()

        #Actor Update
        self.actor.zero_grad()
        new_policy_actions = self.actor(states)
        actor_loss = -self.critic([states, new_policy_actions])
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()

        self.learn_step_counter += 1

    def TD3_learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.sample_memory(network = self.target_actor)

        target_actions = self.target_actor.forward(states_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale = 0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, -self.min_max_action, self.min_max_action)

        q1_ = self.target_critic_1(states_, target_actions)
        q2_ = self.target_critic_2(states_, target_actions)

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        q1_[dones] = 0.0
        q2_[dones] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value = T.min(q1_, q2_)

        target = rewards + self.gamma*critic_value
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

        self.learn_step_counter += 1

    def learn_comms(self):
        if self.comms_memory.mem_ctr < self.batch_size:
            return
        self.q_comms_eval.optimizer.zero_grad()

        self.replace_target_comms_network()

        states, message, rewards, states_, dones = self.sample_comms_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_comms_eval.forward(states)[indices, message]

        q_next = self.q_comms_next.forward(states_)
        q_eval = self.q_comms_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim = 1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_comms_eval.loss(q_target, q_pred).to(self.q_comms_eval.device)
        loss.backward()
        self.q_comms_eval.optimizer.step()

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def store_comms_transition(self, state, action, reward, state_, done):
        self.comms_memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def sample_memory(self, network):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(network.device)
        actions = T.tensor(action).to(network.device)
        rewards = T.tensor(reward).to(network.device)
        states_ = T.tensor(new_state).to(network.device)
        dones = T.tensor(done).to(network.device)

        return states, actions, rewards, states_, dones

    def sample_comms_memory(self):
        state, action, reward, new_state, done = self.comms_memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_comms_eval.device)
        actions = T.tensor(action).to(self.q_comms_eval.device)
        rewards = T.tensor(reward).to(self.q_comms_eval.device)
        states_ = T.tensor(new_state).to(self.q_comms_eval.device)
        dones = T.tensor(done).to(self.q_comms_eval.device)

        return states, actions, rewards, states_, dones

    def save_model(self, path):
        if self.learning_scheme == 'DQN' or self.learning_scheme == 'DDQN':
            self.q_eval.save_model(path)

    def load_model(self, path):
        if self.learning_scheme == 'DQN' or self.learning_scheme == 'DDQN':
            self.q_eval.load_model(path)
