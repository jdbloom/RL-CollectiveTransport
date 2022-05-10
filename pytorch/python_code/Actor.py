from LearningAids import NetworkAids 
from replay_buffer import ReplayBuffer

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam

import numpy as np



class Actor(NetworkAids):
    '''
    This class will be the foundation class for Agent and will hold all specific functions
    '''
    def __init__(self, id, n_obs, n_actions, options_per_action, n_agents, n_chars, meta_param_size, 
                 intention = False, recurrent_intention = False, intention_look_back = 2):

        super().__init__()

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

    def build_networks(self, learning_scheme):
        if learning_scheme == 'DQN':
            self.networks = self.build_DQN()
            self.networks['learning_scheme'] = 'DQN'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete')
        elif learning_scheme == 'DDQN':
            self.networks = self.build_DDQN()
            self.networks['learning_scheme'] = 'DDQN'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete')
        elif learning_scheme == 'DDPG':
            self.networks = self.build_DDPG()
            self.networks['learning_scheme'] = 'DDPG'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.n_actions, 'Continuous')
        elif learning_scheme == 'TD3':
            self.networks = self.build_TD3()
            self.networks['learning_scheme'] = 'TD3'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.n_actions, 'Continuous')
        else:
            raise Exception('[ERROR] Learning scheme is not recognised: '+learning_scheme)


    def build_DQN(self):
        nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                   'num_ops_per_action':self.options_per_action}
        return self.make_DQN_networks(nn_args)
    
    def build_DDQN(self):
        nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                   'num_ops_per_action':self.options_per_action}
        return self.make_DDQN_networks(nn_args)
    
    def build_DDPG(self):
        actor_nn_args = {'id':self.id, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':self.n_actions, 'lr': self.lr, 'observation_size':self.network_input_size}

        return self.make_DDPG_networks(actor_nn_args, critic_nn_args)


    def build_TD3(self):
        actor_nn_args = {'id':self.id, 'alpha':self.alpha, 'input_dims': self.network_input_size, 'fc1_dims':400,
                         'fc2_dims':300, 'n_actions':self.n_actions}
        critic_nn_args = {'id':self.id, 'beta':self.beta, 'input_dims':self.network_input_size, 'fc1_dims':400,
                          'fc2_dims':300, 'n_actions':self.n_actions}

        return self.make_TD3_networks(actor_nn_args, critic_nn_args)

    def build_intention_network(self, learning_scheme):
        if learning_scheme == 'DDPG':
            if self.recurrent_intention:
                self.intention_networks = self.build_RDDPG_intention()
                self.intention_networks['learning_scheme'] = 'RDDPG'
            else:
                self.intention_networks = self.build_DDPG_intention()
                self.intention_networks['learning_scheme'] = 'DDPG'
        elif learning_scheme == 'TD3':
            if self.recurrent_intention:
                self.intention_networks = self.build_RTD3_intention()
                self.intention_networks['learning_scheme'] = 'RTD3'
            else:
                self.intention_networks = self.build_TD3_intention()
                self.intention_networks['learning_scheme'] = 'TD3'
        else:
            raise Exception('[Error] Intention learning scheme is not recognised: '+learning_scheme)

    def build_DDPG_intention(self):
        actor_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':self.intention_network_input,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':1, 'lr': self.lr, 'observation_size':self.intention_network_input}

        return self.make_DDPG_networks(actor_nn_args, critic_nn_args)
    
    def build_RDDPG_intention(self):
        actor_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':self.intention_network_input,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':1, 'lr': self.lr, 'observation_size':self.intention_network_input}
        ee_nn_args = {'observation_size': self.network_input_size, 'hidden_size':self.meta_param_size, 'meta_param_size': self.meta_param_size, 'batch_size': self.batch_size, 'num_layers':1, 'lr': self.ee_lr}
        Networks = self.make_DDPG_networks(actor_nn_args, critic_nn_args)
        Networks['ee'] = self.make_EE_networks(ee_nn_args)

        return Networks
        

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        # Update Intention Networks 
        if self.intention:
            if self.intention_networks['learning_scheme'] == 'DDPG' or self.intention_networks['learning_scheme'] == 'RDDPG':
                self.intention_networks = self.update_DDPG_network_parameters(tau, self.intention_networks)
            elif self.intention_networks['learning_scheme'] == 'TD3' or self.intention_networks['learning_scheme'] == 'RTD3':
                self.intention_networks = self.update_TD3_network_parameters(tau, self.intention_networks)
        # Update Action Selection Networks
        if self.networks['learning_scheme'] == 'DDPG':
            self.networks = self.update_DDPG_network_parameters(tau, self.networks)
        elif self.networks['learning_scheme'] == 'TD3':
            self.networks = self.update_TD3_network_parameters(tau, self.networks)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_ctr==0:
            self.networks['q_next'].load_state_dict(self.networks['q_eval'].state_dict())
 
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

    def choose_action(self, observation, test=False):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            if test or np.random.random()>self.epsilon:
                actions = self.DQN_DDQN_choose_action(observation, self.networks)
            else:
                actions = np.random.choice(self.action_space)
            return actions

        elif self.networks['learning_scheme'] =='DDPG':
            actions = self.DDPG_choose_action(observation, self.networks)
            if not test:
                actions+=T.normal(0.0, self.noise, size = (1, self.n_actions)).to(self.networks['actor'].device)
            actions = T.clamp(actions, -self.min_max_action, self.min_max_action)
            return actions[0].cpu().detach().numpy()

        elif self.networks['learning_scheme'] == 'TD3':
            return self.TD3_choose_action(observation, self.networks, self.n_actions)

        else:
            raise Exception('[ERROR]: Learning scheme not recognised for action selection'+self.networks['learning_scheme'])
    
    def learn(self):
        if self.networks['replay'].mem_ctr < (self.n_agents*self.batch_size + self.batch_size):
                return

        if self.intention:
            self.learn_intention()

        if self.networks['learning_scheme'] == 'DQN':
            self.networks['q_eval'].optimizer.zero_grad()
            self.replace_target_network()
            return self.learn_DQN(self.networks)

        elif self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].optimizer.zero_grad()
            self.replace_target_network()
            return self.learn_DDQN(self.networks)

        elif self.networks['learning_scheme'] == 'DDPG':
            self.update_network_parameters()
            return self.learn_DDPG(self.networks)

        elif self.networks['learning_scheme'] == 'TD3':
            self.update_network_parameters()
            return self.learn_TD3(self.networks)

    def save_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].save_model(path)

        elif self.networks['learning_scheme'] == 'DDPG':
            self.networks['actor'].save_checkpoint(path)
            self.networks['target_actor'].save_checkpoint(path)
            self.networks['critic'].save_checkpoint(path)
            self.networks['target_critic'].save_checkpoint(path)

        elif self.networks['learning_scheme'] == 'TD3':
            self.networks['actor'].save_checkpoint(path)
            self.networks['target_actor'].save_checkpoint(path)
            self.networks['critic_1'].save_checkpoint(path)
            self.networks['target_critic_1'].save_checkpoint(path)
            self.networks['critic_2'].save_checkpoint(path)
            self.networks['target_critic_2'].save_checkpoint(path)

    def load_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].load_model(path)
            #print('-------------------- Weights ------------------')
            #for param in self.q_eval.parameters():
            #    print(param.data)
        elif self.networks['learning_scheme'] == 'DDPG':
            self.networks['actor'].load_checkpoint(path)
            self.networks['target_actor'].load_checkpoint(path)
            self.networks['critic'].load_checkpoint(path)
            self.networks['target_critic'].load_checkpoint(path)

        elif self.networks['learning_scheme'] == 'TD3':
            self.networks['actor'].load_checkpoint(path)
            self.networks['target_actor'].load_checkpoint(path)
            self.networks['critic_1'].load_checkpoint(path)
            self.networks['target_critic_1'].load_checkpoint(path)
            self.networks['critic_2'].load_checkpoint(path)
            self.networks['target_critic_2'].load_checkpoint(path)
            
    
if __name__=='__main__':
    agent_args = {'id':1, 'n_obs':32, 'n_actions':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'intention':False, 'recurrent_intention':False, 'intention_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])

    
    print('[TESTING] DQN')
    agent.build_networks('DQN')
    agent.learn_step_counter = agent.replace_target_ctr
    agent.replace_target_network()
    observation = np.random.random(size = agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())

    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])

    print('[TESTING] DDQN')
    agent.build_networks('DDQN')
    agent.replace_target_network()
    observation = np.random.random(size = agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())
    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])
    
    print('[TESTING] DDPG and param update')
    agent.build_networks('DDPG')
    agent.update_network_parameters()
    observation = np.random.random(size = agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())

    '''
    print('ACTOR')
    for name, param in agent.networks['actor'].named_parameters():
        if param.requires_grad:
            print(name, param.data)
    print('TARGET ACTOR')
    for name, param in agent.networks['target_actor'].named_parameters():
        if param.requires_grad:
            print (name, param.data)
    agent.update_network_parameters(tau = 1)
    print('TARGET ACTOR AFTER UPDATE')
    for name, param in agent.networks['target_actor'].named_parameters():
        if param.requires_grad:
            print(name, param.data)
    print(agent.networks['actor'])
    print(agent.networks['critic'])
    '''

    print('[TESTING] TD3')
    agent.build_networks('TD3')
    agent.update_network_parameters()
    observation = np.random.random(size = agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['n_obs']+agent_args['n_agents']*agent_args['n_chars'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())
    print(agent.networks['actor'])
    print(agent.networks['critic_1'])
    print(agent.networks['critic_2'])
    

    print('[TESTING] Intention DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, intention=True, recurrent_intention = False)
    agent.build_networks('DDPG')
    agent.build_intention_network('DDPG')
    agent.update_network_parameters()
    print(agent.intention_networks['actor'])
    print(agent.intention_networks['critic'])
    

    print('[TESTING] Recurrent Intention DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, intention=True, recurrent_intention = True)
    agent.build_networks('DDPG')
    agent.build_intention_network('DDPG')
    agent.update_network_parameters()
    print(agent.intention_networks['actor'])
    print(agent.intention_networks['critic'])
    print(agent.intention_networks['ee'])
