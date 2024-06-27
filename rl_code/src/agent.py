from gsp_rl.src.actors import Actor

import math
import numpy as np
import statistics
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from torch.optim import Adam


class Agent(Actor):
    def __init__(
            self,
            config: dict,
            network: str,
            n_agents: int,
            n_obs: int,
            n_actions: int,
            options_per_action: int,
            id: int,
            min_max_action: float,
            meta_param_size: int,
            gsp: bool,
            recurrent: bool,
            attention: bool,
            neighbors: bool,
            gsp_input_size: int,
            gsp_output_size: int,
            gsp_min_max_action: float,
            gsp_look_back: int,
            gsp_sequence_length: int,
            prox_filter_angle_deg: float = 45.0,
            n_hop_neighbors: int = 1,
    ):
        if neighbors:
            # 2 inputs from ownship (prev_gsp, avg_prox)
            # 2 inputs from each neighbor (prev_gsp, avg_prox)
            # 2*n_hop_neighbors for symmetry in both CW and CCW
            gsp_input_size = 2+2*(n_hop_neighbors*2)  

        output_size = n_actions
        if network in ['DQN', 'DDQN']:
            output_size = options_per_action**n_actions

        gsp_rl_args = {
            'config': config,
            'network': network,
            'id':id,
            'input_size':n_obs,
            'output_size':output_size,
            'min_max_action': min_max_action,
            'meta_param_size':meta_param_size, 
            'gsp':gsp,
            'recurrent_gsp':recurrent,
            'attention': attention,
            'gsp_input_size': gsp_input_size,
            'gsp_output_size': gsp_output_size,
            'gsp_min_max_action': gsp_min_max_action,
            'gsp_look_back':gsp_look_back,
            'gsp_sequence_length': gsp_sequence_length
        }
        super().__init__(**gsp_rl_args)

        self._n_agents = n_agents
        self._network = network
        self._n_actions = n_actions
        self._neighbors = neighbors
        self._n_hop_neighbors = n_hop_neighbors
        self.neighbors_dict = {}
        self._options_per_action = options_per_action
        self._prox_filter_angle_deg = prox_filter_angle_deg


        if self._neighbors:
            self.gsp_observation = []
            for _ in range(self._n_agents):
                self.gsp_observation.append([[0 for _ in range(self.gsp_network_input)] for _ in range(self.gsp_sequence_length)])
        else:
            self.gsp_observation = [[0 for _ in range(self.gsp_network_input)] for _ in range(self.gsp_sequence_length)]

        self._ROBOT_PROXIMITY_ANGLES = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5,
                                       112.5, 127.5, 142.5, 157.5, 172.5, -172.5, 
                                       -157.5, -142.5, -127.5, -112.5, -97.5, 
                                       -82.5, -67.5, -52.5, -37.5, -22.5, -7.5]
        if self._neighbors:
            self.build_neighbors()
        
    @property
    def gsp_neighbors(self):
        return self._neighbors

    @property
    def n_agents(self):
        return self._n_agents

    def build_neighbors(self):
        agents_available = np.arange(self.n_agents)
        for agent in range(self.n_agents):
            neighbors = []
            for i in range(1, self._n_hop_neighbors+1):
                neighbors.append(agents_available[agent-i])
                neighbors.append(agents_available[(agent+1)%self.n_agents])
            self.neighbors_dict[agent] = neighbors
    
    def make_agent_state(self, env_obs, heading_gsp=None, global_knowledge=None):
        # robot_cos_to_goal = math.cos(env_obs[1])
        # robot_sin_to_goal = math.sin(env_obs[1])
        # robot_tan_to_goal = math.tan(env_obs[1])

        # cyl_cos_to_goal = math.cos(env_obs[5])
        # cyl_sin_to_goal = math.sin(env_obs[5])
        # cyl_tan_to_goal = math.tan(env_obs[5])

        # anlges = np.array((
        #     robot_cos_to_goal, 
        #     robot_sin_to_goal,
        #     robot_tan_to_goal,
        #     cyl_cos_to_goal,
        #     cyl_sin_to_goal,
        #     cyl_tan_to_goal
        # ))
        # env_obs = np.concatenate((env_obs, anlges))
        # # Normalize the angles
        # env_obs[1] /= math.pi
        # env_obs[5] /= math.pi
        # print('===============================')
        # print('robot_dist2goal ', env_obs[0])
        # print('robot_angle2goal', env_obs[1])
        # print('robot_lwheel    ', env_obs[2])
        # print('robot_rwheel    ', env_obs[3])
        # print('cyl_dist2robot  ', env_obs[4])
        # print('cyl_angle2robot ', env_obs[5])
        # print('cyl_dist2goal   ', env_obs[6])

        if heading_gsp is not None:
            if global_knowledge is not None:
                env_obs = np.concatenate((env_obs, np.array([np.degrees(heading_gsp/10)]), global_knowledge)) 
            else:
                env_obs = np.concatenate((env_obs, np.array([np.degrees(heading_gsp/10)]))) 
        elif global_knowledge is not None:
            env_obs = np.concatenate((env_obs, global_knowledge))
        return env_obs   
    
    def make_gsp_states(self, agent_prox_values, agent_prev_gsp):
        states = []
        for agent in range(self._n_agents):
            agent_state = np.zeros(self.gsp_network_input)
            neighbors = self.neighbors_dict[agent]
            agent_state[0] = agent_prox_values[agent]
            agent_state[1] = agent_prev_gsp[agent]
            i=2
            for neighbor in neighbors:
                agent_state[i] = agent_prox_values[neighbor]
                agent_state[i+1] = agent_prev_gsp[neighbor]
                i+=2
            self.gsp_observation[agent].pop(0)
            self.gsp_observation[agent].append(agent_state)
            states.append(agent_state)
        return states
    
    def filter_prox_values(self, prox_values, angle_to_cyl):
        if angle_to_cyl > 0:
            if angle_to_cyl > 180-self._prox_filter_angle_deg:
                cw_lim = angle_to_cyl + self._prox_filter_angle_deg - 360
            else:
                cw_lim = angle_to_cyl+self._prox_filter_angle_deg
            ccw_lim = angle_to_cyl - self._prox_filter_angle_deg
        elif angle_to_cyl < 0:
            if angle_to_cyl < -180 +self._prox_filter_angle_deg:
                ccw_lim = angle_to_cyl-self._prox_filter_angle_deg+360
            else:
                ccw_lim = angle_to_cyl - self._prox_filter_angle_deg
            cw_lim = angle_to_cyl + self._prox_filter_angle_deg
        else:
            cw_lim = self._prox_filter_angle_deg
            ccw_lim = -self._prox_filter_angle_deg

        index = []
        filtered_prox_values = []
        if angle_to_cyl > 180 - self._prox_filter_angle_deg:
            for i in range(len(self._ROBOT_PROXIMITY_ANGLES)):
                if self._ROBOT_PROXIMITY_ANGLES[i] > ccw_lim:
                    index.append(i)
                elif self._ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i])
        elif angle_to_cyl < -180+self._prox_filter_angle_deg:
            for i in range(len(self._ROBOT_PROXIMITY_ANGLES)):
                if self._ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                elif self._ROBOT_PROXIMITY_ANGLES[i] > ccw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i]) 
        else:
            for i in range(len(self._ROBOT_PROXIMITY_ANGLES)):
                if self._ROBOT_PROXIMITY_ANGLES[i] > ccw_lim and self._ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i])
        return filtered_prox_values, index
    
    def choose_agent_action(self, observation, failures, test=False):
        if self._network == 'None':
            # Not sure what to do here for no learning
            return [0, 0, 0], 0

        if failures:
            self.failed = True
            return self.failure_action, self.failure_action_code

        self.failed = False
        if self.networks['learning_scheme'] in ['DQN', 'DDQN']:
            action_num = self.choose_action(observation, self.networks, test)
            actions = self.parse_action(action_num)

        if self.networks['learning_scheme'] in ['DDPG', 'TD3']:
            actions = self.choose_action(observation, self.networks, test)
            actions = np.pad(actions, (0, 1))
            action_num = None

        return actions, action_num
    
    def choose_agent_gsp(self, agent_gsp_states, test = False):
        if self._neighbors:
            actions = []
            for i in range(self._n_agents):
                if self.recurrent_gsp:
                    # print(f'[AGENT {i}], Observation: {self.gsp_observation[i]}')
                    actions.append(self.choose_action(self.gsp_observation[i], self.gsp_networks, test))
                    # print(f'[AGENT {i}], Action: {actions[i].shape}')
                else: 
                    actions.append(self.choose_action(agent_gsp_states[i], self.gsp_networks, test))
            return actions
        else:
            if self.recurrent_gsp:
                self.gsp_observation.append(agent_gsp_states)
                self.gsp_observation.pop(0)
                action = self.choose_action(self.gsp_observation, self.gsp_networks, test)
                return action
            
            observation = np.array(agent_gsp_states)
            return self.choose_action(observation, self.gsp_networks, test)

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
        if action_num < 0 or action_num >=self._options_per_action**self._n_actions:
            raise Exception('Action Number Out of Range:'+str(action_num))
        l_wheel = round((math.floor(action_num/self._options_per_action) - 1)/10.0, 1)
        r_wheel = round((action_num%self._options_per_action - 1)/10.0, 1)
        # Trailing zero is hardcoded control for gripper
        return np.array([l_wheel, r_wheel, 0])
    
    def store_agent_transition(self, s, a, r, s_, d):
        if self.networks['replay'].action_type == 'Discrete':
            a = a[0]
        elif self.networks['replay'].action_type == 'Continuous':
            a = np.array(a[1][0:2])
        return super().store_agent_transition(s, a, r, s_, d)
    