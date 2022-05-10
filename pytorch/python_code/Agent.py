#from .networks import DDQN, DDQNComms, DQN, DDPGActorNetwork, DDPGCriticNetwork, TD3ActorNetwork, TD3CriticNetwork, EnvironmentEncoder
#from .replay_buffer import ReplayBuffer
from .communications import Mailbox
from .Actor import Actor

import numpy as np
import math
from collections import namedtuple
import statistics

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

Loss = nn.MSELoss()

class Agent(Actor):
    def __init__(self, n_agents, n_obs, n_actions, options_per_action, id, learning_scheme,
                 n_chars=4, intention_look_back = 2, min_max_action = 1, use_intention=False, 
                 use_recurrent=False, meta_param_size = 0):


        args = {'id':id, 'n_obs':n_obs, 'n_actions':n_actions, 'options_per_action':options_per_action, 'n_agents':n_agents,
                'n_chars':n_chars, 'meta_param_size':meta_param_size, 'intention':use_intention, 'recurrent_intention':use_recurrent,
                'intention_look_back':intention_look_back}

        super().__init__(**args)

        self.learning_scheme = learning_scheme
        self.intention_look_back = intention_look_back
        self.use_intention = use_intention
        self.meta_param_size = meta_param_size
        self.use_recurrent = use_recurrent
        self.seq_len = 5

        self.object_stats = []
        self.min_obj_stats = np.zeros(4) # vel, accel, ang_vel, ang_accel
        self.max_obj_stats = np.zeros(4)
        self.decimals = 2
        min_max = 1.25
        bins = 8
        self.angle_bins = np.arange(-180, 180, 360/bins)
        self.acceleration_bins = np.around(np.arange(-min_max, min_max, (min_max*2)/bins), self.decimals)
        self.binned_angle = None
        self.binned_acceleration = None
        self.obj_state = None
        
        
        self.build_networks(learning_scheme)

        if self.use_intention:
            self.build_intention(self.intention_look_back)

    def make_agent_state(self, env_obs, heading_intention):
        if self.use_intention:
            env_obs = np.concatenate((env_obs, np.array([heading_intention]))) 
        return env_obs

   

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

    def build_intention(self, intention_look_back):
        # define networks for TD3
        print("----- Building Intention Model ------")

        if self.use_intention == "DQN":
            min_max_action = 0.1
            obs_size = intention_look_back*2 + self.n_agents*self.n_chars
            actions_nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':1, 'observation_size':obs_size,
                               'num_ops_per_action':self.options_per_action}

            self.intention_q_eval = DQN(**actions_nn_args, name="intention_DQN")
            self.intention_q_next = DQN(**actions_nn_args)

            print(self.intention_q_eval)
            print(self.intention_q_next)

            self.intention_memory = ReplayBuffer(100000, obs_size, 1, use_intention = True)

        elif self.use_intention == "DDQN":
            min_max_action = 0.1
            obs_size = intention_look_back*2 + self.n_agents*self.n_chars
            actions_nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':1, 'observation_size':obs_size,
                               'num_ops_per_action':self.options_per_action}

            self.intention_q_eval = DDQN(**actions_nn_args, name="intention_DDQN")
            self.intention_q_next = DDQN(**actions_nn_args)

            print(self.intention_q_eval)
            print(self.intention_q_next)

            if self.use_entropy:
                self.intention_memory = ReplayBuffer(100000, obs_size, 1, state_size=2, num_agents=self.n_agents, use_intention = True)
            else:
                self.intention_memory = ReplayBuffer(100000, obs_size, 1, use_intention = True)

        elif self.use_intention == "DDPG":
            self.min_max_action = 1
            obs_size = intention_look_back*2 + self.n_agents*self.n_chars
            actor_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':obs_size,
                             'num_ops_per_action':self.options_per_action,
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

            self.intention_memory = ReplayBuffer(100000, obs_size, self.n_actions, use_intention = True)

        elif self.use_intention == "TD3":
            min_max_action = 1
            obs_size = intention_look_back*2 + self.n_agents*self.n_chars
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
            self.intention_memory = ReplayBuffer(100000, obs_size, self.n_actions, use_intention = True)

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

    def choose_agent_action(self, observation, failures, test=False):
        if self.learning_scheme == 'None':
            # Not sure what to do here for no learning
            return [0, 0, 0], 0

        if failures:
            self.failed = True
            return self.failure_action, self.failure_action_code

        self.failed = False
        ########################### NEED TO APPEND A 0 TO THE END OF THE ACTION
        return np.concatenate(self.choose_action(observation, test), 0)
        

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

    def build_initial_ee_input(self,s,a,s_,r):
        if len(r) == 0:
            r.append(0)
        state = T.from_numpy(s)
        reward = T.from_numpy(np.array(r))
        state_ = T.from_numpy(s_)
        action = T.from_numpy(a)
        #Padding single inputs, EE takes in batches.
        stateP = self.pad_input(state,self.seq_len,self.batch_size)
        actionP = self.pad_input(action,self.seq_len,self.batch_size)
        state_P = self.pad_input(state_,self.seq_len,self.batch_size)
        rewardP = F.pad(reward.unsqueeze(0).unsqueeze(0), pad=(0,0,self.seq_len-1,0,self.batch_size-1,0), value=0)
        return stateP, actionP, state_P, rewardP

    def build_ee_input(self,s,a,r,s_):
        observation = T.cat((s,a,s_), -1)
        if r.dim() == 0:
            r = r.reshape([1])
        if r.dim()== 2:
            r = r.unsqueeze(-1)
        observation = T.cat((observation,r), -1)
        if observation.dim() == 1:
            observation = T.reshape(observation,(1,1,observation.shape[0]))
        return observation.to(T.float32)

    def generate_meta_param(self,s,a,s_,r):
        observation = self.build_ee_input(s,a,r,s_)
        meta_param = self.ee(observation)
        return meta_param
    
    def build_ac_input(self, state, mp):
        #TODO figure out what to replace 4 with
        state = state[:,-1,:]
        obs = T.cat((state,mp), 1)
        return obs
        
    def pad_input(self,s,seqlen,batch):
        s = s.unsqueeze(0).unsqueeze(0)
        s = F.pad(s,pad=(0,0,seqlen-1,0,batch-1,0))
        return s
