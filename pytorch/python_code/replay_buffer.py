import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, num_observations, num_actions, action_type):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.action_type = action_type
        self.state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        if self.action_type == 'Discrete':
            self.action_memory = np.zeros((self.mem_size), dtype = np.int64)
        elif self.action_type == 'Continuous':
            self.action_memory = np.zeros((self.mem_size, num_actions), dtype = np.float32)
        else:
            raise Exception('Unknown Action Type:' + action_type)
        self.reward_memory = np.zeros((self.mem_size), dtype = np.float32)
        self.terminal_memory = np.zeros((self.mem_size), dtype = np.bool)



    def store_transition(self, state, action, reward, state_, done):
        mem_index = self.mem_ctr % self.mem_size
        self.state_memory[mem_index] = state
        if self.action_type == 'Discrete':
            self.action_memory[mem_index] = action[0]
        elif self.action_type == 'Continuous':
            self.action_memory[mem_index] = action[1][0:2]
        self.reward_memory[mem_index] = reward
        self.new_state_memory[mem_index] = state_
        self.terminal_memory[mem_index] = done
        self.mem_ctr += 1


    def sample_buffer(self, batch_size, use_horizon, num_agents):
        max_mem = min(self.mem_ctr, self.mem_size)
        if not use_horizon:
            batch = np.random.choice(max_mem, batch_size, replace = False)
            states = self.state_memory[batch]
            actions = self.action_memory[batch]
            rewards = self.reward_memory[batch]
            next_states = self.new_state_memory[batch]
            dones = self.terminal_memory[batch]
        else:
            max_mem -= batch_size*num_agents
            batch_start = np.random.choice(max_mem)
            batch_end = batch_start + num_agents * batch_size
            batch_index = np.arange(batch_start, batch_end, num_agents)
            states = self.state_memory[batch_index]
            actions = self.action_memory[batch_index]
            rewards = self.reward_memory[batch_index]
            next_states = self.new_state_memory[batch_index]
            dones = self.terminal_memory[batch_index]

        return states, actions, rewards, next_states, dones
