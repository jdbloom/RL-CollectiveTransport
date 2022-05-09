from LearningAids import Hyperparameters, NetworkAids 
from replay_buffer import ReplayBuffer

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam



class Actor(Hyperparameters):
    '''
    This class will be the foundation class for Agent and will hold all specific functions
    '''
    def __init__(self, id, n_obs, n_actions, options_per_action, n_agents, n_chars, meta_param_size, 
                 intention = False, recurrent_intention = False, intention_look_back = 2):

        super().__init__()

        self.NetworkBuilder = NetworkAids()

        self.id = id
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.options_per_action = options_per_action

        self.n_chars = n_chars
        self.n_agents = n_agents
        self.meta_param_size = meta_param_size

        self.action_space = [i for i in range(self.options_per_action**self.n_actions)]
        self.failure_action_code = len(self.action_space)

        self.intention = intention
        self.recurrent_intention = recurrent_intention
        self.intention_look_back = intention_look_back

        self.network_input_size = self.n_obs + self.n_agents*self.n_chars
        if self.intention and not self.recurrent_intention:
            self.network_input_size += 1
        self.intention_network_input = self.intention_look_back*2+self.n_agents*self.n_chars
        self.recurrent_intention_input = self.n_obs + self.n_agents*self.n_chars+self.meta_param_size

    def build_DQN(self):
        nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                   'num_ops_per_action':self.options_per_action}
        return self.NetworkBuilder.make_DQN_networks(nn_args)
    
    def build_DDQN(self):
        nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                   'num_ops_per_action':self.options_per_action}
        return self.NetworkBuilder.make_DDQN_networks(nn_args)
    
    def build_DDPG(self):
        actor_nn_args = {'id':self.id, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':self.n_actions, 'lr': self.lr, 'observation_size':self.network_input_size}

        return self.NetworkBuilder.make_DDPG_networks(actor_nn_args, critic_nn_args)


    def build_TD3(self):
        actor_nn_args = {'id':self.id, 'alpha':self.alpha, 'input_dims': self.network_input_size, 'fc1_dims':400,
                         'fc2_dims':300, 'n_actions':self.n_actions}
        critic_nn_args = {'id':self.id, 'beta':self.beta, 'input_dims':self.network_input_size, 'fc1_dims':400,
                          'fc2_dims':300, 'n_actions':self.n_actions}

        return self.NetworkBuilder.make_TD3_networks(actor_nn_args, critic_nn_args)

    def build_networks(self, learning_scheme):
        if learning_scheme == 'DQN':
            self.networks = self.build_DQN()
        elif learning_scheme == 'DDQN':
            self.networks = self.build_DDQN()
        elif learning_scheme == 'DDPG':
            self.networks = self.build_DDPG()
        elif learning_scheme == 'TD3':
            self.networks = self.build_TD3()
        else:
            raise Exception('[ERROR] Learning scheme is not recognised: '+learning_scheme)


    def build_intention_network(self, learning_scheme):
        if learning_scheme == 'DDPG':
            if self.recurrent_intention:
                self.intention_networks = self.build_RDDPG_intention()
            else:
                self.intention_networks = self.build_DDPG_intention()
        elif learning_scheme == 'TD3':
            if self.recurrent_intention:
                self.intention_networks = self.build_RTD3_intention()
            else:
                self.intention_networks = self.build_TD3_intention()
        else:
            raise Exception('[Error] Intention learning scheme is not recognised: '+learning_scheme)

    def build_DDPG_intention(self):
        actor_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':self.intention_network_input,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':1, 'lr': self.lr, 'observation_size':self.intention_network_input}

        return self.NetworkBuilder.make_DDPG_networks(actor_nn_args, critic_nn_args)
    
    def build_RDDPG_intention(self):
        actor_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':self.intention_network_input,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':1, 'lr': self.lr, 'observation_size':self.intention_network_input}
        ee_nn_args = {'observation_size': self.network_input_size, 'hidden_size':self.meta_param_size, 'meta_param_size': self.meta_param_size, 'batch_size': self.batch_size, 'num_layers':1, 'lr': self.ee_lr}
        Networks = self.NetworkBuilder.make_DDPG_networks(actor_nn_args, critic_nn_args)
        Networks['ee'] = self.NetworkBuilder.make_EE_networks(ee_nn_args)

        return Networks
        


    #def update_network_parameters(self, tau = None, learning_scheme = None):



if __name__=='__main__':
    agent = Actor(1, 32, 2, 3, 1, 2, 2, intention=False, recurrent_intention = False)
    print('[TESTING] DQN')
    agent.build_networks('DQN')
    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])

    print('[TESTING] DDQN')
    agent.build_networks('DDQN')
    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])
    
    print('[TESTING] DDPG')
    agent.build_networks('DDPG')
    print(agent.networks['actor'])
    print(agent.networks['critic'])
    
    print('[TESTING] TD3')
    agent.build_networks('TD3')
    print(agent.networks['actor'])
    print(agent.networks['critic_1'])
    print(agent.networks['critic_2'])

    print('[TESTING] Intention DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, intention=True, recurrent_intention = False)
    agent.build_intention_network('DDPG')
    print(agent.intention_networks['actor'])
    print(agent.intention_networks['critic'])

    print('[TESTING] Recurrent Intention DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, intention=True, recurrent_intention = True)
    agent.build_intention_network('DDPG')
    print(agent.intention_networks['actor'])
    print(agent.intention_networks['critic'])
    print(agent.intention_networks['ee'])