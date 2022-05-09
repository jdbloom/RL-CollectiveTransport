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

    def update_DDPG_network_parameters(self, tau, networks):
        # Update Actor Network
        for target_param, param in zip(networks['target_actor'].parameters(), networks['actor'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        # Update Critic Network
        for target_param, param in zip(networks['target_critic'].parameters(), networks['critic'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        return networks

    def update_TD3_network_parameters(self, tau, networks):
        actor_params = networks['actor'].named_parameters()
        critic_1_params = networks['critic_1'].named_parameters()
        critic_2_params = networks['critic_2'].named_parameters()
        target_actor_params = networks['target_actor'].named_parameters()
        target_critic_1_params = networks['target_critic_1'].named_parameters()
        target_critic_2_params = networks['target_critic_2'].named_parameters()

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

        networks['target_critic_1'].load_state_dict(critic_1)
        networks['target_critic_2'].load_state_dict(critic_2)
        networks['target_actor'].load_state_dict(actor)

        return networks

