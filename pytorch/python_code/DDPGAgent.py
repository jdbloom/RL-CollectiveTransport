import numpy as np
import torch as T
import torch.nn as nn
from torch.optim import Adam

from networks_torch import ActorNetwork, CriticNetwork
from buff import ReplayBuffer

class Agent:
    def __init__(self, observation_size, num_actions, num_ops_per_action, min_max_action = 1, alpha = 0.001, beta = 0.002, gamma = 0.99, max_size = 1000000, tau = 0.005, batch_size = 64, noise = 0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, observation_size, num_actions)
        self.batch_size = batch_size
        self.noise = noise
        # sets the +/- bounds for the action space: EX. 1 -> (-1, 1)
        self.min_max_action = min_max_action

        self.actor = ActorNetwork(num_actions = num_actions, observation_size = observation_size, num_ops_per_action = num_ops_per_action)
        self.target_actor = self.actor = ActorNetwork(num_actions = num_actions, observation_size = observation_size, num_ops_per_action = num_ops_per_action)
        self.actor_optimizer = Adam(self.actor.parameters(),lr = alpha)

        self.critic = CriticNetwork(num_actions = num_actions, observation_size = observation_size)
        self.target_critic = CriticNetwork(num_actions = num_actions, observation_size = observation_size)
        self.critic_optimizer = Adam(self.critic.parameters(),lr = beta)

        self.update_network_parameters(tau = 1)

        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-6

        self.action_space = [i for i in range(self.num_ops_per_action**self.num_actions)]

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

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        target_actions = self.target_actor(states_)
        q_value_ = self.target_critic([states_, target_actions])
        target = rewards + self.gamma*q_value_*(1-dones)


        # Critic Update
        self.critic.zero_grad()
        q_value = self.critic([states, actions])
        value_loss = nn.MSELoss(q_value, target)
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        self.actor.zero_grad()
        new_policy_actions = self.actor(states)
        actor_loss = -self.critic([state, new_policy_actions])
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()

    def choose_action(self, observation, evaluate = False):
        if evaluate or np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype = T.float).to(self.q_eval.device)
            actions = self.actor(state)
            action = T.argmax(actions[0]).item()
        else:
            action = np.random.choice(aelf.action_space)

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

        return actions

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
