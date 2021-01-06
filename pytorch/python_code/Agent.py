from .networks import DDQN, DDQNComms, DQN, DDPGActorNetwork, DDPGCriticNetwork, TD3ActorNetwork, TD3CriticNetwork
from .replay_buffer import ReplayBuffer

from collections import namedtuple
from torch.optim import Adam
import numpy as np
import torch as T
#from .communications import Mailbox

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
        self.alphabet_space = [np.zeros(self.alphabet_size) for i in range(self.alphabet_size + 1)]
        for i in range(self.alphabet_size):
            self.alphabet_space[i+1][i] = 1
        self.dead_channel_code = self.alphabet_space[0]

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
        self.failure_action_code = [0, 0, 1]
        self.failure_action_code = len(self.action_space)
        self.learn_step_counter = 0
        self.noise = 0.1
        self.update_actor_iter = 2
        self.warmup = 1000

        self.memory = ReplayBuffer(100000, self.num_observations + 2*self.alphabet_size)
        self.comms_memory = ReplayBuffer(100000, self.num_observations + 2*self.alphabet_size)

        if self.learning_scheme == 'None':
            # define logic for no learning?
            pass

        if self.learning_scheme == 'DQN':
            #define networks for DQN
            actions_nn_args = {'lr':self.lr, 'num_actions':self.num_actions, 'observation_size':self.num_observations,
                       'num_ops_per_action':self.num_ops_per_action}
            self.q_eval = DQN(**actions_nn_args)
            self.q_next = DQN(**actions_nn_args)

        elif self.learning_scheme == 'DDQN':
            obs_size = self.num_observations + 2*self.alphabet_size
            actions_nn_args = {'lr':self.lr, 'num_actions':self.num_actions, 'observation_size':obs_size,
                       'num_ops_per_action':self.num_ops_per_action}
            comms_nn_args = {'lr':self.lr, 'observation_size':obs_size, 'alphabet_size':self.alphabet_size}

            self.q_eval = DDQN(**actions_nn_args)
            self.q_next = DDQN(**actions_nn_args)
            self.q_comms_eval = DDQNComms(**comms_nn_args)
            self.q_comms_next = DDQNComms(**comms_nn_args)

        elif self.learning_scheme == 'DDPG':
            #define networks for DDPG
            obs_size = self.num_observations + 2*self.alphabet_size
            actor_nn_args = {'num_actions':self.num_actions, 'observation_size':obs_size,
                             'num_ops_per_action':self.num_ops_per_action, 'min_max_action':self.min_max_action}
            critic_nn_args = {'num_actions':self.num_actions, 'observation_size':obs_size}

            self.actor = DDPGActorNetwork(**actor_nn_args)
            self.target_actor = DDPGActorNetwork(**actor_nn_args)
            self.actor_optimizer = Adam(self.actor.parameters(), lr = self.lr)

            self.critic = DDPGCriticNetwork(**critic_nn_args)
            self.target_critic = DDPGCriticNetwork(**critic_nn_args)
            self.critic_optimizer = Adam(self.critic.parameters(), lr = self.lr)

            self.update_network_parameters(tau = 1)

            self.actor.cuda()
            self.target_actor.cuda()
            self.critic.cuda()
            self.target_critic.cuda()

            comms_nn_args = {'lr':self.lr, 'observation_size':obs_size, 'alphabet_size':self.alphabet_size}

            self.q_comms_eval = DDQNComms(**comms_nn_args)
            self.q_comms_next = DDQNComms(**comms_nn_args)

        elif self.learning_scheme == 'TD3':
            #difine networks for TD3
            obs_size = self.num_observations + 2*self.alphabet_size
            self.actor = TD3ActorNetwork(self.alpha, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'actor')
            self.target_actor = TD3ActorNetwork(self.alpha, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'target_actor')

            self.critic_1 = TD3CriticNetwork(self.beta, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'critic_1')
            self.target_critic_1 = TD3CriticNetwork(self.beta, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'target_critic_1')

            self.critic_2 = TD3CriticNetwork(self.beta, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'critic_2')
            self.target_critic_2 = TD3CriticNetwork(self.beta, input_dims = obs_size, fc1_dims = 400, fc2_dims = 300, n_actions = self.num_actions, name = 'target_critic_2')

            self.update_network_parameters(tau = 1)

            comms_nn_args = {'lr':self.lr, 'observation_size':obs_size, 'alphabet_size':self.alphabet_size}

            self.q_comms_eval = DDQNComms(**comms_nn_args)
            self.q_comms_next = DDQNComms(**comms_nn_args)
        else:
            raise Exception('Unknown Learning Scheme' + self.learning_scheme)

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
