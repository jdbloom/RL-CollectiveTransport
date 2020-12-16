import numpy as np
import torch as T
import torch.nn as nn
from torch.optim import Adam
from collections import namedtuple
import math

from .TD3_networks import ActorNetwork, CriticNetwork
from .deep_q_comms import DeepQComms
from .replay_buffer import ReplayBuffer
from .mailbox import Mailbox


Loss = nn.MSELoss()


class Agent_DDPG:
    def __init__(self, num_agents, num_observation, num_actions,
                 num_ops_per_action, id, comm_scheme="None",
                 alphabet_size=4, min_max_action=1, alpha=0.001,
                 beta=0.002, lr=0.0001, gamma=0.99, max_size=1000000,
                 tau=0.005, batch_size=64, noise=0.1, update_actor_iter=2, warmup = 1000):

        self.id = id

        self.num_agents = num_agents
        self.n_actions = num_actions
        self.num_ops_per_action = num_ops_per_action
        self.alphabet_size = alphabet_size
        self.init_comm_scheme(comm_scheme, self.num_agents)

        self.warmup = warmup
        # An iterable describing who may contact who. Entries in format {sender: [receivers]}
        # Merge left and right contacts into a master dictionary
        self.contacts = {key: val+self.right_contacts[key]
                         for (key, val) in self.left_contacts.items()}

        self.mailbox = Mailbox(self.contacts, self.dead_channel_code)

        self.gamma = gamma
        self.tau = tau
        self.state_size = num_observation+2*alphabet_size
        # ! We will need to add in the number of actions when we switch to continuous
        self.memory = ReplayBuffer(max_size, self.state_size)
        self.comms_memory = ReplayBuffer(max_size, self.state_size)
        self.batch_size = batch_size
        self.noise = noise
        # sets the +/- bounds for the action space: EX. 1 -> (-1, 1)
        self.min_max_action = min_max_action

        self.init_networks(self.state_size, num_actions, num_ops_per_action, alpha, beta, lr)

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
        self.update_actor_iter = update_actor_iter
        self.time_step = 0


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
        self.actor = ActorNetwork(alpha, input_dims = num_observations, fc1_dims = 400, fc2_dims = 300, n_actions = num_actions, name = 'actor')
        self.critic_1 = CriticNetwork(beta, input_dims = num_observations, fc1_dims = 400, fc2_dims = 300, n_actions = num_actions, name = 'critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims = num_observations, fc1_dims = 400, fc2_dims = 300, n_actions = num_actions, name = 'critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims = num_observations, fc1_dims = 400, fc2_dims = 300, n_actions = num_actions, name = 'target_actor')
        self.target_critic_1 = CriticNetwork(beta, input_dims = num_observations, fc1_dims = 400, fc2_dims = 300, n_actions = num_actions, name = 'target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims = num_observations, fc1_dims = 400, fc2_dims = 300, n_actions = num_actions, name = 'target_critic_2')


        self.critic = CriticNetwork(num_actions = num_actions, observation_size = num_observation)
        self.target_critic = CriticNetwork(num_actions = num_actions, observation_size = num_observation)
        self.critic_optimizer = Adam(self.critic.parameters(),lr = beta)

        self.update_network_parameters(tau = 1)

        comms_nn_args = {'lr':lr, 'observation_size':num_observation, 'alphabet_size':self.alphabet_size}
        self.q_comms_eval = DeepQComms(**comms_nn_args)
        self.q_comms_next = DeepQComms(**comms_nn_args)

    def update_network_parameters(self, tau = None):
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
    def TD3_learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return

        state, action, reward, state_, done = self.sample_memory()

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale = 0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, -self.min_max_action, self.min_max_action)

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[dones] = 0.0
        q2_[dones] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_.loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def choose_action(self, observation, failure, evaluate = False):
        if failure:
            self.failed = True
            return self.failure_action, self.failure_action_code
        else: self.failed = False

        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale = self.noise, size = (self.n_actions,)))

        else:
            state = T.tensor(observation, dtype = T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale = self.noise), dtype = T.float).to(self.actor.device)
        # NEED TO DECIDE ABOUT COMMUNICATION HERE!!!
        mu_prime = T.clamp(mu_prime, -self.min_max_action, self.min_max_action)
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def choose_message(self, observation, failure, test, i):
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
        m = incoming_comms(left_comm, right_comm)
        #print('Incoming Comms for agent ', agent_id, m.left_comm, m.right_comm)
        return m

    def make_agent_state(self, env_obs, agent_id):
        communications = self.get_agent_incoming_communications(agent_id)
        agent_state = np.concatenate((env_obs, communications.left_comm, communications.right_comm))
        return agent_state

    def clear_agent_inbox(self, agent_id):
        self.mailbox.clear_inbox(agent_id)

    def schedule_message_to_all_contacts(self, sender, contents):
        '''
        The sender sends a message containing contents to everyone it can contact
        '''
        for receiver in self.mailbox.contacts[sender]:
            self.mailbox.schedule_message(sender, receiver, contents)

    def carry_mail(self):
        self.mailbox.clear_inbox()
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


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_2.load_checkpoint()
