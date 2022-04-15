from .networks import DDQN, DDQNComms, DQN, DDPGActorNetwork, DDPGCriticNetwork, TD3ActorNetwork, TD3CriticNetwork
from .replay_buffer import ReplayBuffer
from .communications import Mailbox

import numpy as np
import math
from collections import namedtuple
import statistics

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

Loss = nn.MSELoss()

class Agent():
    def __init__(self, num_agents, num_observations, num_actions,
                 num_ops_per_action, id, learning_scheme, no_buffer,
                 comms_memory, normalization, comms_scheme = "None", alphabet_size=4, horizon = 2,
                 min_max_action = 1, use_horizon = False, use_entropy=False, use_intention=False, heading='radial'):
        self.id = id
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.num_ops_per_action = num_ops_per_action
        self.num_observations = num_observations
        self.alphabet_size = alphabet_size
        self.comms_scheme = comms_scheme
        self.learning_scheme = learning_scheme
        self.min_max_action = min_max_action
        self.use_horizon = use_horizon
        self.object_stats = []
        self.min_obj_stats = np.zeros(4) # vel, accel, ang_vel, ang_accel
        self.max_obj_stats = np.zeros(4)
        self.normalization = normalization
        self.use_horizon = use_horizon
        self.horizon = horizon
        self.use_intention = use_intention
        self.heading = heading

        # Dictionaries and binning for loss function calculations
        self.use_entropy = use_entropy
        self.StateDict = {}
        self.StateDict["count"] = 0
        self.MsgDict = {}
        self.MsgDict["count"] = 0
        self.decimals = 2
        min_max = 1.25
        bins = 8
        self.angle_bins = np.arange(-180, 180, 360/bins)
        self.acceleration_bins = np.around(np.arange(-min_max, min_max, (min_max*2)/bins), self.decimals)
        self.binned_angle = None
        self.binned_acceleration = None
        self.obj_state = None

        self.make_comms_scheme()
        self.make_learning_scheme(comms_memory)
        if self.use_intention:
            self.build_intention(self.horizon)


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
            self.contacts = {i:[] for i in range(self.num_agents)}
        elif self.comms_scheme == 'Left':
            self.contacts = {i:[i+1] for i in range(self.num_agents)}
            self.contacts[self.num_agents-1] = 0
        elif self.comms_scheme == 'Right':
            self.contacts = {i:[i-1] for i in range(self.num_agents)}
            self.contacts[0] = self.num_agents - 1
        elif self.comms_scheme == 'Neighbors':
            self.contacts = {i:[] for i in range(self.num_agents)}
            [self.contacts[i].append(i-1) for i in range(self.num_agents)]
            [self.contacts[i].append(i+1) for i in range(self.num_agents)]
            self.contacts[self.num_agents-1][0] = 0
            self.contacts[0][1] = self.num_agents -1
        elif self.comms_scheme == 'Broadcast':
            self.contacts = {i:[] for i in range(self.num_agents)}
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:
                        self.contacts[i].append(j)
        else:
            raise Exception('Unknown Communication Scheme ' + self.comms_scheme)

        self.mailbox = Mailbox(self.contacts, self.dead_channel_code)

    def get_agent_incoming_communications(self, agent_id):
        messages = self.mailbox.inbox[agent_id]
        incoming_comms = namedtuple('incoming_comms', 'msgs')
        msgs = [self.dead_channel_message for i in range(len(self.contacts.keys()))]
        for message in messages:
            if agent_id in self.contacts[message.sender]:
                if message.contents != -1:
                    msgs[message.sender] = self.alphabet_space[message.contents]

        msg = incoming_comms(msgs)
        return msg

    def make_agent_state(self, env_obs, heading_intention, agent_id, comms_memory, message_memory):
        #env_obs=self.normalize_obs(env_obs)
        if self.use_intention:
            # import ipdb; ipdb.set_trace()
            if self.heading == 'polar':
                env_obs = np.concatenate((env_obs, heading_intention))
            else:
                env_obs = np.concatenate((env_obs, np.array([heading_intention])))
        if self.comms_scheme == 'None':
            # need to append empty messages for all agents to keep networks the same size
            return np.concatenate((env_obs, np.zeros(self.alphabet_size*self.num_agents))), self.dead_channel_code
        msg = self.get_agent_incoming_communications(agent_id)
        if not comms_memory:
            messages = np.concatenate(msg.msgs)
            agent_state = np.concatenate((env_obs, messages))
        else:
            agent_state = env_obs
            for i in range(len(message_memory)):
                agent_state = np.concatenate((agent_state, message_memory[i]))
            if len(agent_state) < (self.num_observations + self.num_agents * self.alphabet_size):
                zeros_to_add = (self.num_observations + self.num_agents * self.alphabet_size) - len(agent_state)
                agent_state = np.concatenate((agent_state, np.zeros(zeros_to_add)))
        return agent_state, msg

    def normalize_obs(self, env_obs):
        env_obs[0] = env_obs[0]/self.normalization['distance']
        env_obs[1] = (env_obs[1]+180)/self.normalization['angle']
        env_obs[2] = (env_obs[2]+10)/self.normalization['wheel_speeds']
        env_obs[3] = (env_obs[3]+10)/self.normalization['wheel_speeds']
        env_obs[4] = env_obs[4]
        env_obs[5] = env_obs[2] = (env_obs[5]+180)/self.normalization['angle']
        env_obs[6] = env_obs[6]/self.normalization['distance']
        return env_obs

    def clear_agent_inbox(self, agent_id):
        self.mailbox.clear_inbox(agent_id)

    def schedule_message_to_all_contacts(self, sender, contents):
        for receiver in self.mailbox.contacts[sender]:
            self.mailbox.schedule_message(sender, receiver, contents)

    def carry_mail(self):
        self.mailbox.carry_mail()


    def make_learning_scheme(self, comms_memory):
        # These are the hyper parameters relating to the neural networks
        # HYPER PARAMETERS COME FROM "Continuous Control with Deep Reinforcement Learning"
        self.action_space = [i for i in range(self.num_ops_per_action**self.num_actions)]
        self.gamma = 0.99997
        self.tau = 0.005
        self.alpha = 0.001
        self.beta = 0.002
        self.lr = 0.0001
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-5
        
        if self.use_intention == "DQN" or self.use_intention == "DDQN":
            self.intn_epsilon = 1.0
            self.intn_eps_min = 0.01
            self.intn_eps_dec = 1e-5

        self.batch_size = 100
        self.replace_target_cnt = 1000
        self.failed = False
        self.failure_action = [0, 0, 1]
        self.failure_action_code = len(self.action_space)
        self.learn_step_counter = 0
        self.intention_learn_step_counter = 0
        self.noise = 0.1
        self.update_actor_iter = 2
        self.warmup = 1000
        self.time_step = 0

        if self.comms_scheme != 'None':
            self.make_comms_network(comms_memory)

        print('######## LEARNING SCHEME ########')
        print('SCHEME:', self.learning_scheme)
        print('MODEL SPECIFICS:')
        if self.learning_scheme == 'None':
            # define logic for no learning?
            return

        if self.learning_scheme == 'DQN':
            self.make_DQN(comms_memory)

        elif self.learning_scheme == 'DDQN':
            self.make_DDQN(comms_memory)

        elif self.learning_scheme == 'DDPG':
            self.make_DDPG()

        elif self.learning_scheme == 'TD3':
            self.make_TD3()

        else:
            raise Exception('Unknown Learning Scheme' + self.learning_scheme)

    def make_comms_network(self, comms_memory):
        if not comms_memory:
            obs_size = self.num_observations + self.num_agents*self.alphabet_size
        else:
            obs_size = self.num_observations + self.num_agents*self.alphabet_size
        print('MODEL SPECIFICS:')
        comms_nn_args = {'id':self.id, 'lr':self.lr, 'observation_size':obs_size, 'alphabet_size':self.alphabet_size}
        self.q_comms_eval = DDQNComms(**comms_nn_args)
        self.q_comms_next = DDQNComms(**comms_nn_args)
        print('Communication Network:')
        print(self.q_comms_eval)
        print(self.q_comms_next)
        if self.use_entropy:
            self.comms_memory = ReplayBuffer(100000, obs_size, 1, 'Discrete', 2, self.num_agents)
        else:
            self.comms_memory = ReplayBuffer(100000, obs_size, 1, 'Discrete')

    def make_DQN(self, comms_memory):
        #define networks for DQN
        self.min_max_action = 0.1
        obs_size = self.num_observations + self.num_agents*self.alphabet_size
        if self.use_intention:
            if self.heading == 'radial':
                obs_size += 1
            if self.heading == 'polar':
                obs_size += 2

        actions_nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':self.num_actions, 'observation_size':obs_size,
                   'num_ops_per_action':self.num_ops_per_action}

        self.q_eval = DQN(**actions_nn_args)
        self.q_next = DQN(**actions_nn_args)
        print(self.q_eval)
        print(self.q_next)

        self.memory = ReplayBuffer(100000, obs_size, 1, 'Discrete')

    def make_DDQN(self, comms_memory):
        self.min_max_action = 0.1
        obs_size = self.num_observations + self.num_agents*self.alphabet_size
        if self.use_intention:
            if self.heading == 'radial':
                obs_size += 1
            if self.heading == 'polar':
                obs_size += 2

        actions_nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':self.num_actions, 'observation_size':obs_size,
                   'num_ops_per_action':self.num_ops_per_action}

        self.q_eval = DDQN(**actions_nn_args)
        self.q_next = DDQN(**actions_nn_args)
        print(self.q_eval)
        print(self.q_next)
        if self.use_entropy:
            self.memory = ReplayBuffer(100000, obs_size, 1, 'Discrete', 2, self.num_agents)
        else:
            self.memory = ReplayBuffer(100000, obs_size, 1, 'Discrete')

    def make_DDPG(self):
        #define networks for DDPG
        self.min_max_action = 1
        obs_size = self.num_observations + self.num_agents*self.alphabet_size
        if self.use_intention:
            if self.heading == 'radial':
                obs_size += 1
            if self.heading == 'polar':
                obs_size += 2

        actor_nn_args = {'id':self.id, 'num_actions':self.num_actions, 'observation_size':obs_size,
                         'num_ops_per_action':self.num_ops_per_action,
                         'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':self.num_actions, 'observation_size':obs_size}

        self.actor = DDPGActorNetwork(**actor_nn_args, name = 'actor')
        self.target_actor = DDPGActorNetwork(**actor_nn_args, name = 'target_actor')
        self.actor_optimizer = Adam(self.actor.parameters(), lr = self.lr, weight_decay = 1e-4)
        print(self.actor)
        print(self.target_actor)
        self.critic = DDPGCriticNetwork(**critic_nn_args, name = 'critic')
        self.target_critic = DDPGCriticNetwork(**critic_nn_args, name = 'target_critic')
        self.critic_optimizer = Adam(self.critic.parameters(), lr = self.lr, weight_decay = 1e-4)
        print(self.critic)
        print(self.target_critic)

        self.update_network_parameters(tau = 1, learning_scheme ='DDPG')

        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

        self.memory = ReplayBuffer(100000, obs_size, self.num_actions, 'Continuous')

    def make_TD3(self):
        #difine networks for TD3
        self.min_max_action = 1
        obs_size = self.num_observations + self.num_agents*self.alphabet_size
        if self.use_intention:
            if self.heading == 'radial':
                obs_size += 1
            if self.heading == 'polar':
                obs_size += 2

        actor_nn_args = {'id':self.id, 'alpha':self.alpha, 'input_dims':obs_size, 'fc1_dims':400,
                         'fc2_dims':300, 'n_actions':self.num_actions}
        critic_nn_args = {'id':self.id, 'beta':self.beta, 'input_dims':obs_size, 'fc1_dims':400,
                          'fc2_dims':300, 'n_actions':self.num_actions}

        self.actor = TD3ActorNetwork(**actor_nn_args, name = 'actor')
        self.target_actor = TD3ActorNetwork(**actor_nn_args, name = 'target_actor')
        print(self.actor)
        print(self.target_actor)
        self.critic_1 = TD3CriticNetwork(**critic_nn_args, name = 'critic_1')
        self.target_critic_1 = TD3CriticNetwork(**critic_nn_args, name = 'target_critic_1')
        print(self.critic_1)
        print(self.target_critic_1)
        self.critic_2 = TD3CriticNetwork(**critic_nn_args, name = 'critic_2')
        self.target_critic_2 = TD3CriticNetwork(**critic_nn_args, name = 'target_critic_2')
        print(self.critic_2)
        print(self.target_critic_2)

        self.update_network_parameters(tau = 1, learning_scheme ='TD3')

        self.memory = ReplayBuffer(100000, obs_size, self.num_actions, 'Continuous')

    def update_network_parameters(self, tau = None, learning_scheme = None):
        # If tau = 1 -> hard update (should only be done during init)
        if tau is None:
            tau = self.tau

        if learning_scheme == 'DDPG':
            # Update Actor Network
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            # Update Critic Network
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        elif learning_scheme == "intention_DDPG":
            for target_param, param in zip(self.intention_target_actor.parameters(), self.intention_actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            # Update Critic Network
            for target_param, param in zip(self.intention_target_critic.parameters(), self.intention_critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        elif learning_scheme == 'TD3':
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

        elif learning_scheme == 'intention_TD3':
            actor_params = self.intention_actor.named_parameters()
            critic_1_params = self.intention_critic_1.named_parameters()
            critic_2_params = self.intention_critic_2.named_parameters()
            target_actor_params = self.intention_target_actor.named_parameters()
            target_critic_1_params = self.intention_target_critic_1.named_parameters()
            target_critic_2_params = self.intention_target_critic_2.named_parameters()

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

            self.intention_target_critic_1.load_state_dict(critic_1)
            self.intention_target_critic_2.load_state_dict(critic_2)
            self.intention_target_actor.load_state_dict(actor)
        else:
            raise Exception('UNKNOWN NETWORK UPDATE RULE'+self.learning_scheme)


    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def replace_target_comms_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_comms_next.load_state_dict(self.q_comms_eval.state_dict())

    def choose_action(self, observation, failure, test = False):
        #print('[DEBUG] Failure:', failure)
        if failure:
            #print('DEBUG Retruning Failure State')
            self.failed = True
            return self.failure_action, self.failure_action_code
        else:
            self.failed = False

        if self.learning_scheme == 'None':
            # Not sure what to do here for no learning
            return [0, 0, 0], 0

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
            state = T.tensor([observation], dtype = T.float).to(self.q_eval.device)
            # need to turn off batch norm and dropout for network evaluation (batch size = 1)
            self.q_eval.eval()
            action_values = self.q_eval.forward(state)
            # need to turn on batch norm and dropout for network training
            self.q_eval.train()
            action = T.argmax(action_values[0]).item()
        else:
            action = np.random.choice(self.action_space)
        actions = self.parse_action(action)
        return (actions, action)

    def DDPG_choose_action(self, observation, test = False):
        state = T.tensor([observation], dtype = T.float).to(self.actor.device).unsqueeze(0)
        actions = self.actor(state)
        if not test:
            actions += T.normal(0.0, self.noise, size = (1, self.num_actions)).to(self.actor.device)
        actions = T.clamp(actions, -self.min_max_action, self.min_max_action)
        gripper = np.zeros((1, 1))
        actions = np.append(actions[0].cpu().detach().numpy(), gripper)
        return (actions, None)

    def TD3_choose_action(self, observation, test = False):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale = self.noise,
                                           size = (self.num_actions,))
                          ).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype = T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
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
            outgoing_message = self.alphabet_space[outgoing_message_code]
        else:
            outgoing_message_code = np.random.choice(self.alphabet_size)
            outgoing_message = self.alphabet_space[outgoing_message_code]
        return outgoing_message, outgoing_message_code

    def learn(self):
        if self.use_intention:
            self.learn_intention()
        if self.learning_scheme == 'None':
            return 0,0
        elif self.learning_scheme == 'DQN':
            return self.DQN_learn()
        elif self.learning_scheme == 'DDQN':
            return self.DDQN_learn()
        elif self.learning_scheme == 'DDPG':
            return self.DDPG_learn()
        elif self.learning_scheme == 'TD3':
            return self.TD3_learn()

    def DQN_learn(self):
        if self.memory.mem_ctr < (self.num_agents*self.batch_size + self.batch_size):
            return 0,0
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones, state_vec, message_vec = self.sample_memory(network = self.q_eval)

        indices = T.LongTensor(np.arange(self.batch_size).astype(np.long))

        q_pred = self.q_eval(states)[indices, actions]

        q_next = self.q_next(states_).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        gradients = []
        #for param in self.q_eval.parameters():
        #  gradients.append(param.grad)
        #var_grad = statistics.variance(gradients)

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss.item(), 0

    def DDQN_learn(self):
        if self.memory.mem_ctr < (self.num_agents*self.batch_size + self.batch_size):
            return 0, 0

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones, state_vec, message_vec = self.sample_memory(network = self.q_eval, get_entropy = self.use_entropy)

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval(states)[indices, actions]

        q_next = self.q_next(states_)
        q_eval = self.q_eval(states_)

        max_actions = T.argmax(q_eval, dim = 1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        if self.use_entropy:
            listener_loss = self.calculate_entropy_loss(state_vec, message_vec, type = 'listener')
        else: listener_loss = 0

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device) + abs(listener_loss)
        #if loss > 100:
        #    import ipdb; ipdb.set_trace()

        loss.backward()

        gradients = []
        var_grad = 0
        #for param in self.q_eval.parameters():
        #    gradients = np.concatenate((gradients, T.clip(param.grad, -10, 10).cpu().detach().numpy().flatten()))

        #var_grad = np.var(gradients)


        self.q_eval.optimizer.step()

        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss.item(), var_grad

    def learn_no_buffer(self, sarsd):
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = sarsd

        network = self.q_eval

        states = T.tensor(states).to(network.device)
        actions = T.tensor(actions).to(network.device)
        rewards = T.tensor(rewards).to(network.device)
        states_ = T.tensor(states_).to(network.device)
        dones = T.tensor(dones).to(network.device)

        indices = np.arange(len(dones))

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
        if self.memory.mem_ctr < (self.num_agents*self.batch_size + self.batch_size):
            return 0, 0

        states, actions, rewards, states_, dones, state_vec, message_vec = self.sample_memory(network = self.actor)
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

        self.update_network_parameters(learning_scheme = 'DDPG')

        self.learn_step_counter += 1

        return 0, 0

    def TD3_learn(self):
        if self.memory.mem_ctr < (self.num_agents*self.batch_size + self.batch_size):
            return 0, 0

        states, actions, rewards, states_, dones, state_vec, message_vec = self.sample_memory(network = self.target_actor)

        target_actions = self.target_actor.forward(states_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale = 0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, -self.min_max_action, self.min_max_action)

        q1_ = self.target_critic_1.forward(states_, target_actions)
        q2_ = self.target_critic_2.forward(states_, target_actions)

        q1 = self.critic_1.forward(states, actions)
        q2 = self.critic_2.forward(states, actions)

        q1_[dones] = 0.0
        q2_[dones] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_iter != 0:
            return 0, 0
        #print('Actor Learn Step')
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(states, self.actor.forward(states))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters(learning_scheme = 'TD3')

        return 0, 0

    def learn_comms(self):
        if self.comms_memory.mem_ctr < (self.num_agents*self.batch_size + self.batch_size):
            return 0

        self.q_comms_eval.optimizer.zero_grad()

        self.replace_target_comms_network()

        states, message, rewards, states_, dones, state_vec, message_vec = self.sample_comms_memory(get_entropy = self.use_entropy)

        indices = np.arange(self.batch_size)

        q_pred = self.q_comms_eval.forward(states)[indices, message]

        q_next = self.q_comms_next.forward(states_)
        q_eval = self.q_comms_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim = 1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        if self.use_entropy:
            speaker_loss = self.calculate_entropy_loss(state_vec, message_vec, type = 'speaker')
        else: speaker_loss = 0
        loss = self.q_comms_eval.loss(q_target, q_pred).to(self.q_comms_eval.device) + abs(speaker_loss)
        loss.backward()
        self.q_comms_eval.optimizer.step()
        return loss.item()

    def learn_no_buffer_comms(self, sarsd):

        self.q_comms_eval.optimizer.zero_grad()

        self.replace_target_comms_network()

        states, messages, rewards, states_, dones = sarsd

        network = self.q_eval

        states = T.tensor(states).to(network.device)
        messages = T.tensor(messages).to(network.device)
        rewards = T.tensor(rewards).to(network.device)
        states_ = T.tensor(states_).to(network.device)
        dones = T.tensor(dones).to(network.device)

        indices = np.arange(len(dones))

        q_pred = self.q_comms_eval.forward(states)[indices, messages]

        q_next = self.q_comms_next.forward(states_)
        q_eval = self.q_comms_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim = 1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_comms_eval.loss(q_target, q_pred).to(self.q_comms_eval.device)
        loss.backward()
        self.q_comms_eval.optimizer.step()

    def store_transition(self, state, action, reward, state_, done, state_vec = None, message_vec = None):
        if self.use_entropy:
            self.memory.store_transition(state, action, reward, state_, done, state_vec, message_vec)
        else:
            self.memory.store_transition(state, action, reward, state_, done)

    def store_comms_transition(self, state, action, reward, state_, done, state_vec=None, message_vec=None):
        if self.use_entropy:
            self.comms_memory.store_transition(state, action, reward, state_, done, state_vec, message_vec)
        else:
            self.comms_memory.store_transition(state, action, reward, state_, done)

    def store_intention_transition(self, state, action, reward, state_, done, state_vec=None, message_vec=None):
        if self.use_entropy:
            self.intention_memory.store_transition(state, action, reward, state_, done, state_vec, message_vec)
        else:
            self.intention_memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def sample_memory(self, network, get_entropy = False, intention = False):
        if get_entropy:
            state, action, reward, new_state, done, state_vec, message_vec = self.memory.sample_buffer(self.batch_size, self.use_horizon, self.num_agents, get_entropy)
        else:
            state, action, reward, new_state, done, state_vec, message_vec = self.memory.sample_buffer(self.batch_size, self.use_horizon, self.num_agents)

        states = T.tensor(state).to(network.device)
        actions = T.tensor(action).to(network.device)
        rewards = T.tensor(reward).to(network.device)
        states_ = T.tensor(new_state).to(network.device)
        dones = T.tensor(done).to(network.device)
        return states, actions, rewards, states_, dones, state_vec, message_vec

    def sample_comms_memory(self, get_entropy = False):
        if get_entropy:
            state, action, reward, new_state, done, state_vec, message_vec = self.comms_memory.sample_buffer(self.batch_size, self.use_horizon, self.num_agents, get_entropy = get_entropy)
        else:
            state, action, reward, new_state, done, state_vec, message_vec = self.comms_memory.sample_buffer(self.batch_size, self.use_horizon, self.num_agents)

        states = T.tensor(state).to(self.q_comms_eval.device)
        actions = T.tensor(action).to(self.q_comms_eval.device)
        rewards = T.tensor(reward).to(self.q_comms_eval.device)
        states_ = T.tensor(new_state).to(self.q_comms_eval.device)
        dones = T.tensor(done).to(self.q_comms_eval.device)

        return states, actions, rewards, states_, dones, state_vec, message_vec

    def sample_intention_memory(self, network):
        if self.use_entropy:
            state, action, reward, new_state, done, state_vec, message_vec = self.intention_memory.sample_buffer(self.batch_size, get_entropy=self.use_entropy)
        else:
            state, action, reward, new_state, done, _, _ = self.intention_memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(network.device)
        actions = T.tensor(action).to(network.device)
        rewards = T.tensor(reward).to(network.device)
        states_ = T.tensor(new_state).to(network.device)
        dones = T.tensor(done).to(network.device)
        
        if self.use_entropy:
            return states, actions, rewards, states_, dones, state_vec, message_vec

        return states, actions, rewards, states_, dones

    def save_model(self, path):
        if self.learning_scheme == 'DQN' or self.learning_scheme == 'DDQN':
            self.q_eval.save_model(path)

        elif self.learning_scheme == 'DDPG':
            self.actor.save_checkpoint(path)
            self.target_actor.save_checkpoint(path)
            self.critic.save_checkpoint(path)
            self.target_critic.save_checkpoint(path)

        elif self.learning_scheme == 'TD3':
            self.actor.save_checkpoint(path)
            self.target_actor.save_checkpoint(path)
            self.critic_1.save_checkpoint(path)
            self.target_critic_1.save_checkpoint(path)
            self.critic_2.save_checkpoint(path)
            self.target_critic_2.save_checkpoint(path)

        if self.comms_scheme != 'None':
            self.q_comms_eval.save_model(path)

        if self.use_intention == "DQN" or self.use_intention == "DDQN":
            self.intention_q_eval.save_model(path)

        elif self.use_intention == "DDPG":
            self.intention_actor.save_checkpoint(path)
            self.intention_target_actor.save_checkpoint(path)
            self.intention_critic.save_checkpoint(path)
            self.intention_target_critic.save_checkpoint(path)

        elif self.use_intention == "TD3":
            self.intention_actor.save_checkpoint(path)
            self.intention_target_actor.save_checkpoint(path)
            self.intention_critic_1.save_checkpoint(path)
            self.intention_target_critic_1.save_checkpoint(path)
            self.intention_critic_2.save_checkpoint(path)
            self.intention_target_critic_2.save_checkpoint(path)

    def load_model(self, path):
        if self.learning_scheme == 'DQN' or self.learning_scheme == 'DDQN':
            self.q_eval.load_model(path)
            #print('-------------------- Weights ------------------')
            #for param in self.q_eval.parameters():
            #    print(param.data)
        elif self.learning_scheme == 'DDPG':
            self.actor.load_checkpoint(path)
            self.target_actor.load_checkpoint(path)
            self.critic.load_checkpoint(path)
            self.target_critic.load_checkpoint(path)

        elif self.learning_scheme == 'TD3':
            self.actor.load_checkpoint(path)
            self.target_actor.load_checkpoint(path)
            self.critic_1.load_checkpoint(path)
            self.target_critic_1.load_checkpoint(path)
            self.critic_2.load_checkpoint(path)
            self.target_critic_2.load_checkpoint(path)
        if self.comms_scheme != 'None':
            self.q_comms_eval.load_model(path)

        if self.use_intention == "DQN" or self.learning_scheme == "DDQN":
            self.intention_q_eval.load_model(path)

        if self.use_intention == "DDPG":
            self.intention_actor.load_checkpoint(path)
            self.intention_target_actor.load_checkpoint(path)
            self.intention_critic.load_checkpoint(path)
            self.intention_target_critic.load_checkpoint(path)

        elif self.use_intention == "TD3":
            self.intention_actor.load_checkpoint(path)
            self.intention_target_actor.load_checkpoint(path)
            self.intention_critic_1.load_checkpoint(path)
            self.intention_target_critic_1.load_checkpoint(path)
            self.intention_critic_2.load_checkpoint(path)
            self.intention_target_critic_2.load_checkpoint(path)

    def angle_difference(self, a1, a2):
        diff = a1 - a2
        while diff < -180.0:
            diff += 360.0
        while diff > 180.0:
            diff -= 360
        return diff

    def store_object_stats(self, obj_stats, calculate):
        # len = 3 is so that we can calculate velocity and acceleration
        # calculate is a flag that tracks based on episode time steps.
        # we dont want to do any calculations within the first 3 time
        # steps of an episode.
        if not calculate:
            self.object_stats.append(obj_stats)
        else:
            # get rid of the oldest stat
            self.object_stats.pop(0)
            self.object_stats.append(obj_stats)
            velocity_t0 = math.sqrt((self.object_stats[2][0] - self.object_stats[1][0])**2 + (self.object_stats[2][1] - self.object_stats[1][1])**2)/0.1
            velocity_t1 = math.sqrt((self.object_stats[1][0] - self.object_stats[0][0])**2 + (self.object_stats[1][1] - self.object_stats[0][1])**2)/0.1
            acceleration = (velocity_t0 - velocity_t1)/0.1

            angular_velocity_t0 = self.angle_difference(self.object_stats[2][5], self.object_stats[1][5])/0.1

            angular_velocity_t1 = self.angle_difference(self.object_stats[1][5], self.object_stats[0][5])/0.1
            angular_acceleration = (angular_velocity_t0 - angular_velocity_t1)/0.1

            self.binned_acceleration = int(min(self.acceleration_bins, key=lambda x:abs(x-acceleration))*(10**self.decimals))
            self.binned_angle = int(min(self.angle_bins, key=lambda x:abs(x-self.object_stats[2][5])))
            self.obj_state = [self.binned_acceleration, self.binned_angle]


            # This is for finding bin limits only
            if velocity_t0 > self.max_obj_stats[0]: self.max_obj_stats[0] = velocity_t0
            if velocity_t0 < self.min_obj_stats[0]: self.min_obj_stats[0] = velocity_t0
            if acceleration > self.max_obj_stats[1]: self.max_obj_stats[1] = acceleration
            if acceleration < self.min_obj_stats[1]: self.min_obj_stats[1] = acceleration
            if angular_velocity_t0 > self.max_obj_stats[2]: self.max_obj_stats[2] = angular_velocity_t0
            if angular_velocity_t0 < self.min_obj_stats[2]: self.min_obj_stats[2] = angular_velocity_t0
            if angular_acceleration > self.max_obj_stats[3]: self.max_obj_stats[3] = angular_acceleration
            if angular_acceleration < self.min_obj_stats[3]: self.min_obj_stats[3] = angular_acceleration

            #print("[INFO] Angle: %0.2f" %self.object_stats[0][5])
            #print("[INFO] Velocity[0]: %0.2f m/s" % velocity_t0)
            #print("[INFO] Velocity[1]: %0.2f m/s" % velocity_t1)
            #print("[INFO] Acceleration: %0.5f m/s^2" % acceleration)
            #print("[INFO] Angular Velocity[0]: %0.2f deg/s" % angular_velocity_t0)
            #print("[INFO] Angular Velocity[1]: %0.2f deg/s" % angular_velocity_t1)
            #print("[INFO] Angular Acceleration: %0.5f deg/s^2" % angular_acceleration)

    def reset_obj_stats(self):
        self.object_stats = []

    def store_state_message(self, message_codes, calculate):
        if calculate:
            # Create Dict Keys from message and state
            message_key = [str(i) for i in message_codes]
            message_key = ("".join(message_key))
            state_key = [str(i) for i in self.obj_state]
            state_key = ("".join(state_key))
            # Increment the total counts
            self.StateDict["count"] += 1
            self.MsgDict["count"] += 1
            # Add to StateDict and MessageDict
            if message_key in self.MsgDict.keys():
                # we have seen this message before so lets increment the total state count
                self.MsgDict[message_key]["count"] += 1
                if state_key in self.MsgDict[message_key].keys():
                    # we have seen this message-state combo so lets increment the state count
                    self.MsgDict[message_key][state_key] += 1
                else:
                    # we have not seen this message-state combo so lets add the state key and initialize at 1
                    self.MsgDict[message_key][state_key] = 1
            else:
                # we have not seen this message before so we have to create a new dict under the message key
                self.MsgDict[message_key] = {}
                # we initialize the total state count to 1
                self.MsgDict[message_key]["count"] = 1
                # we add the state key and initialize the count to 1
                self.MsgDict[message_key][state_key] = 1

            if state_key in self.StateDict.keys():
                # we have seen this state before so lets increment the total message count
                self.StateDict[state_key]["count"] += 1
                if message_key in self.StateDict[state_key].keys():
                    # we have seen this state-message combo so lets increment the message count
                    self.StateDict[state_key][message_key] += 1
                else:
                    # we have not seen this message before so lets add the message key and initialize to 1
                    self.StateDict[state_key][message_key] = 1
            else:
                # we have not seen this state before so we have to create a new dict under the state key
                self.StateDict[state_key] = {}
                # we initialize the total message count to 1
                self.StateDict[state_key]["count"] = 1
                # we add the message key and initialize the count to 1
                self.StateDict[state_key][message_key] = 1

    def calculate_probabilities(self, code_message_state):
        message_code = code_message_state[0]
        obj_state = code_message_state[1]
        message_key = [str(i) for i in message_code]
        message_key = ("".join(message_key))
        state_key = [str(i) for i in obj_state]
        state_key = ("".join(state_key))
        # check if the message exists in the dict and the state exists in the message
        if message_key in self.MsgDict.keys() and state_key in self.MsgDict[message_key].keys():
            # check if the state exists in the dict and the message exists in the state
            if state_key in self.StateDict.keys() and message_key in self.StateDict[state_key].keys():
                probability_of_message = float(self.MsgDict[message_key]["count"]) / float(self.MsgDict["count"])
                probability_of_state = float(self.StateDict[state_key]["count"]) / float(self.StateDict["count"])
                probability_of_state_given_message = float(self.MsgDict[message_key][state_key]) / float(self.MsgDict[message_key]["count"])
                probability_of_message_given_state = float(self.StateDict[state_key][message_key]) / float(self.StateDict[state_key]["count"])
                return probability_of_message, probability_of_state, probability_of_message_given_state, probability_of_state_given_message
            else: print("Bug in Storing Probabilities")
        # The way this is set up, if one doesnt exist then the other will not exist and all probabilities are 0
        else:
            print("Message Key", message_key, 'or State Key', state_key, 'Does Not Exist')
            return 0, 0, 0, 0

    def calculate_entropy_loss(self, state_vec, message_vec, type):
        probabilities = map(self.calculate_probabilities, zip(message_vec, state_vec))
        # Probabilities = (probability of message, probability of state, probability of message given state, probability of state given message)
        if type == 'speaker':
            speaker_loss = np.sum(np.fromiter(map(lambda x: x[0]*math.log(x[2]), probabilities), dtype = np.float32))
            #print("[DEBUG] Speaker Loss: %.4f" % speaker_loss)
            return speaker_loss
        elif type == 'listener':
            listener_loss = np.sum(np.fromiter(map(lambda x: x[1]*math.log(x[3]), probabilities), dtype = np.float32))
            #print("[DEBUG] Listener Loss: %.4f" % listener_loss)
            return listener_loss
        else:
            raise Exception('UNKNOWN ENTROPY LOSS'+type)


    def build_intention(self, horizon):
        # define networks for TD3
        print("----- Building Intention Model ------")

        if self.use_intention == "DQN":
            min_max_action = 0.1
            obs_size = horizon*2 + self.num_agents*self.alphabet_size
            actions_nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':1, 'observation_size':obs_size,
                               'num_ops_per_action':self.num_ops_per_action}

            self.intention_q_eval = DQN(**actions_nn_args, name="intention_DQN")
            self.intention_q_next = DQN(**actions_nn_args)

            print(self.intention_q_eval)
            print(self.intention_q_next)

            self.intention_memory = ReplayBuffer(100000, obs_size, 1, use_intention = True)

        elif self.use_intention == "DDQN":
            min_max_action = 0.1
            obs_size = horizon*2 + self.num_agents*self.alphabet_size
            actions_nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':1, 'observation_size':obs_size,
                               'num_ops_per_action':self.num_ops_per_action}

            self.intention_q_eval = DDQN(**actions_nn_args, name="intention_DDQN")
            self.intention_q_next = DDQN(**actions_nn_args)

            print(self.intention_q_eval)
            print(self.intention_q_next)

            if self.use_entropy:
                self.intention_memory = ReplayBuffer(100000, obs_size, 1, state_size=2, num_agents=self.num_agents, use_intention = True)
            else:
                self.intention_memory = ReplayBuffer(100000, obs_size, 1, use_intention = True)

        elif self.use_intention == "DDPG":
            self.min_max_action = 1
            obs_size = horizon*2 + self.num_agents*self.alphabet_size
            actor_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':obs_size,
                             'num_ops_per_action':self.num_ops_per_action,
                             'min_max_action':self.min_max_action}
            critic_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':obs_size}

            self.intention_actor = DDPGActorNetwork(**actor_nn_args, name = 'intention_actor')
            self.intention_target_actor = DDPGActorNetwork(**actor_nn_args, name = 'intention_target_actor')

            self.intention_actor_optimizer = Adam(self.intention_actor.parameters(), lr = self.lr, weight_decay = 1e-4)
            
            print(self.intention_actor)
            print(self.intention_target_actor)

            self.intention_critic = DDPGCriticNetwork(**critic_nn_args, name = 'intention_critic')
            self.intention_target_critic = DDPGCriticNetwork(**critic_nn_args, name = 'intention_target_critic')

            self.intention_critic_optimizer = Adam(self.intention_critic.parameters(), lr = self.lr, weight_decay = 1e-4)
            
            print(self.intention_critic)
            print(self.intention_target_critic)

            self.update_network_parameters(tau=1, learning_scheme="intention_DDPG")

            self.intention_actor.cuda()
            self.intention_target_actor.cuda()
            self.intention_critic.cuda()
            self.intention_target_critic.cuda()

            self.intention_memory = ReplayBuffer(100000, obs_size, self.num_actions, use_intention = True)

        elif self.use_intention == "TD3":
            min_max_action = 1
            obs_size = horizon*2 + self.num_agents*self.alphabet_size
            actor_nn_args = {'id':self.id, 'alpha':self.alpha, 'input_dims':obs_size, 'fc1_dims':400,
                            'fc2_dims':300, 'n_actions':1}
            critic_nn_args = {'id':self.id, 'beta':self.beta, 'input_dims':obs_size, 'fc1_dims':400,
                            'fc2_dims':300, 'n_actions':1}

            self.intention_actor = TD3ActorNetwork(**actor_nn_args, name = 'intention_actor')
            self.intention_target_actor = TD3ActorNetwork(**actor_nn_args, name = 'intention_target_actor')
            print(self.intention_actor)
            print(self.intention_target_actor)
            self.intention_critic_1 = TD3CriticNetwork(**critic_nn_args, name = 'intention_critic_1')
            self.intention_target_critic_1 = TD3CriticNetwork(**critic_nn_args, name = 'intention_target_critic_1')
            print(self.intention_critic_1)
            print(self.intention_target_critic_1)
            self.intention_critic_2 = TD3CriticNetwork(**critic_nn_args, name = 'intention_critic_2')
            self.intention_target_critic_2 = TD3CriticNetwork(**critic_nn_args, name = 'intention_target_critic_2')
            print(self.intention_critic_2)
            print(self.intention_target_critic_2)
            self.update_network_parameters(tau=1, learning_scheme = 'intention_TD3')
            self.intention_memory = ReplayBuffer(100000, obs_size, self.num_actions, use_intention = True)

    def learn_intention(self):
        if self.intention_memory.mem_ctr < self.batch_size:
            return 0, 0

        if self.use_intention =="DQN":
            self.intention_q_eval.zero_grad()
            
            if self.intention_learn_step_counter % self.replace_target_cnt == 0:
                self.intention_q_next.load_state_dict(self.intention_q_eval.state_dict())
            
            states, actions, rewards, states_, dones = self.sample_intention_memory(network = self.intention_q_eval)

            indices = T.LongTensor(np.arange(self.batch_size).astype(np.long))

            q_pred = self.intention_q_eval(states)[indices, actions.to(T.int64)]

            q_next = self.intention_q_next(states_).max(dim=1)[0]

            q_next[dones] = 0.0

            q_target = rewards + self.gamma*q_next

            loss = self.intention_q_eval.loss(q_target, q_pred).to(self.intention_q_eval.device)
            loss.backward()

            self.intention_q_eval.optimizer.step()
            self.intention_learn_step_counter += 1

            self.intn_epsilon = self.intn_epsilon - self.intn_eps_dec if self.intn_epsilon > self.intn_eps_min else self.intn_eps_min

            return loss.item(), 0

        elif self.use_intention == "DDQN":
            self.intention_q_eval.zero_grad()

            if self.intention_learn_step_counter % self.replace_target_cnt == 0:
                self.intention_q_next.load_state_dict(self.intention_q_eval.state_dict())
            
            if self.use_entropy:
                states, actions, rewards, states_, dones, state_vec, message_vec = self.sample_intention_memory(network = self.intention_q_eval)
            else:
                states, actions, rewards, states_, dones = self.sample_intention_memory(network = self.intention_q_eval)

            indices = np.arange(self.batch_size)
            q_pred = self.intention_q_eval(states)[indices, actions.to(T.int64)]
            q_next = self.intention_q_next(states_)
            q_eval = self.intention_q_eval(states_)
            max_actions = T.argmax(q_eval, dim=1)
            q_next[dones] = 0.0

            q_target = rewards + self.gamma*q_next[indices, max_actions]
            if self.use_entropy:
                listener_loss = self.calculate_entropy_loss(state_vec, message_vec, type = 'listener')
            else:
                listener_loss = 0
            
            loss = self.intention_q_eval.loss(q_target, q_pred).to(self.intention_q_eval.device) + abs(listener_loss)
            loss.backward()

            gradients = []
            var_grad = 0

            self.intention_q_eval.optimizer.step()
            self.intention_learn_step_counter += 1

            self.intn_epsilon = self.intn_epsilon - self.intn_eps_dec if self.intn_epsilon > self.intn_eps_min else self.intn_eps_min
            
            return loss.item(), 0

        elif self.use_intention == "DDPG":
            states, actions, rewards, states_, dones = self.sample_intention_memory(network = self.intention_actor)
            
            target_actions = self.intention_target_actor(states_)
            q_value_ = self.intention_target_critic([states_, target_actions])
            q_value_[dones] = 0.0
            target = T.unsqueeze(rewards, 1) + self.gamma*q_value_

            self.intention_critic.zero_grad()
            q_value = self.intention_critic([states, actions.unsqueeze(1)])
            value_loss = Loss(q_value, target)
            value_loss.backward()
            self.intention_critic_optimizer.step()
            
            self.intention_actor.zero_grad()
            new_policy_actions = self.intention_actor(states)
            actor_loss = -self.intention_critic([states, new_policy_actions])
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.intention_actor_optimizer.step()

            self.update_network_parameters(learning_scheme="intention_DDPG")

            self.intention_learn_step_counter += 1

            return 0, 0


        elif self.use_intention == "TD3":
            states, actions, rewards, states_, dones = self.sample_intention_memory(network = self.intention_target_actor)

            target_actions = self.intention_target_actor.forward(states_)
            target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale = 0.2)), -0.5, 0.5)
            target_actions = T.clamp(target_actions, -1, 1)

            q1_ = self.intention_target_critic_1.forward(states_, target_actions)
            q2_ = self.intention_target_critic_2.forward(states_, target_actions)

            q1 = self.intention_critic_1.forward(states, actions.unsqueeze(1))
            q2 = self.intention_critic_2.forward(states, actions.unsqueeze(1))

            q1_[dones] = 0.0
            q2_[dones] = 0.0

            q1_ = q1_.view(-1)
            q2_ = q2_.view(-1)

            critic_value_ = T.min(q1_, q2_)

            target = rewards + self.gamma*critic_value_
            target = target.view(self.batch_size, 1)

            self.intention_critic_1.optimizer.zero_grad()
            self.intention_critic_2.optimizer.zero_grad()

            q1_loss = F.mse_loss(target, q1)
            q2_loss = F.mse_loss(target, q2)
            critic_loss = q1_loss + q2_loss
            critic_loss.backward()
            self.intention_critic_1.optimizer.step()
            self.intention_critic_2.optimizer.step()

            self.intention_learn_step_counter += 1

            if self.intention_learn_step_counter % self.update_actor_iter != 0:
                return 0, 0
            
            self.intention_actor.optimizer.zero_grad()
            actor_q1_loss = self.intention_critic_1.forward(states, self.intention_actor.forward(states))
            actor_loss = -T.mean(actor_q1_loss)
            actor_loss.backward()
            self.intention_actor.optimizer.step()

            self.update_network_parameters(learning_scheme = 'intention_TD3')

            return 0, 0

    def choose_object_intention(self, positions, agent_prox_flags, test = False):
        observation = np.append(np.array(positions), np.array(agent_prox_flags))
        
        if self.use_intention == "DQN" or self.use_intention == "DDQN":
            if test or np.random.random() > self.intn_epsilon:
                state = T.tensor([observation], dtype = T.float).to(self.intention_q_eval.device)
                self.intention_q_eval.eval()
                action_values = self.intention_q_eval.forward(state)
                self.intention_q_eval.train()
                action = T.argmax(action_values[0]).item()

            else:
                action = np.random.choice(1)

            return action

        elif self.use_intention == "DDPG":
            state = T.tensor([observation], dtype = T.float).to(self.intention_actor.device).unsqueeze(0)
            actions = self.intention_actor(state)
            
            if not test:
                actions += T.normal(0.0, self.noise, size = (1,)).to(self.intention_actor.device)

            actions = T.clamp(actions, -self.min_max_action, self.min_max_action)
            gripper = np.zeros((1, 1))
            actions = actions[0].cpu().detach().numpy()

            return actions

        elif self.use_intention == "TD3":
            if self.time_step < self.warmup:
                mu = T.tensor(np.random.normal(scale = self.noise, size = (1,))).to(self.intention_actor.device)

            else:
                state = T.tensor(observation, dtype = T.float).to(self.intention_actor.device)
                mu = self.intention_actor.forward(state).to(self.intention_actor.device)

            mu_prime = mu + T.tensor(np.random.normal(scale = self.noise), dtype = T.float).to(self.intention_actor.device)
            mu_prime = T.clamp(mu_prime, -1, 1)
            actions = mu_prime.cpu().detach().item()
            
            self.time_step += 1

            return actions
