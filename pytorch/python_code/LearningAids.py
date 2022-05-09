from networks import DQN, DDQN, DDPGActorNetwork, DDPGCriticNetwork, TD3ActorNetwork, TD3CriticNetwork, EnvironmentEncoder

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam
class Hyperparameters:
    def __init__(self):
        self.gamma = 0.99997
        self.tau = 0.005
        self.alpha = 0.001
        self.beta = 0.002
        self.lr = 0.0001
        self.ee_lr = 0.01
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-5

        self.intn_epsilon = 1.0
        self.intn_eps_min = 0.01
        self.intn_eps_dec = 1e-5

        self.batch_size = 100
        self.replace_target_cnt = 1000
        self.failed = False
        self.failure_action = [0, 0, 1]
        
        self.learn_step_counter = 0
        self.intention_learn_step_counter = 0
        self.noise = 0.1
        self.update_actor_iter = 2
        self.warmup = 1000
        self.time_step = 0

        self.min_max_action = 1

class NetworkAids:
    def __init__(self):
        pass
    def make_DQN_networks(self, nn_args):
        return {'q_eval':DQN(**nn_args), 'q_next':DQN(**nn_args)}
    
    def make_DDQN_networks(self, nn_args):
        return {'q_eval':DDQN(**nn_args), 'q_next':DDQN(**nn_args)}
    
    def make_DDPG_networks(self, actor_nn_args, critic_nn_args):
        DDPG_netowkrs = {
                        'actor': DDPGActorNetwork(**actor_nn_args, name = 'actor'),
                        'target_actor': DDPGActorNetwork(**actor_nn_args, name = 'target_actor'),
                        'critic': DDPGCriticNetwork(**critic_nn_args, name = 'critic_1'),
                        'target_critic': DDPGCriticNetwork(**critic_nn_args, name = 'target_critic_1')}
        return DDPG_netowkrs

    def make_TD3_networks(self, actor_nn_args, critic_nn_args):
        TD3_netowkrs = {
                        'actor': TD3ActorNetwork(**actor_nn_args, name = 'actor'),
                        'target_actor': TD3ActorNetwork(**actor_nn_args, name = 'target_actor'),
                        'critic_1': TD3CriticNetwork(**critic_nn_args, name = 'critic_1'),
                        'target_critic_1': TD3CriticNetwork(**critic_nn_args, name = 'target_critic_1'),
                        'critic_2': TD3CriticNetwork(**critic_nn_args, name = 'critic_2'),
                        'target_critic_2': TD3CriticNetwork(**critic_nn_args, name = 'target_critic_2')}
        return TD3_netowkrs
    
    def make_EE_networks(self, nn_args):
        return EnvironmentEncoder(**nn_args)
