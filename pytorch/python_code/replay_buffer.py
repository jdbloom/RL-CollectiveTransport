import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, num_observations, num_actions, action_type = None, state_size = None, num_agents = None, use_intention = False):
        self.mem_size = max_size
        self.use_intention = use_intention
        self.mem_ctr = 0
        self.action_type = action_type
        self.state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        if not use_intention:
            if self.action_type == 'Discrete':
                self.action_memory = np.zeros((self.mem_size), dtype = np.int64)
            elif self.action_type == 'Continuous':
                self.action_memory = np.zeros((self.mem_size, num_actions), dtype = np.float32)
            else:
                raise Exception('Unknown Action Type:' + action_type)
        else:
            self.action_memory = np.zeros((self.mem_size), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size), dtype = np.float32)
        self.terminal_memory = np.zeros((self.mem_size), dtype = np.bool)
        if state_size is not None and num_agents is not None:
            self.entropy_memory = self.EntropyBuffer(max_size, state_size, num_agents)


    def store_transition(self, state, action, reward, state_, done, state_vec=None, message_vec=None):
        mem_index = self.mem_ctr % self.mem_size
        self.state_memory[mem_index] = state
        if not self.use_intention:
            if self.action_type == 'Discrete':
                self.action_memory[mem_index] = action[0]
            elif self.action_type == 'Continuous':
                self.action_memory[mem_index] = action[1][0:2]
        else:
            self.action_memory[mem_index] = action
        self.reward_memory[mem_index] = reward
        self.new_state_memory[mem_index] = state_
        self.terminal_memory[mem_index] = done
        self.mem_ctr += 1
        if state_vec is not None and message_vec is not None:
            self.entropy_memory.store_transition(state_vec, message_vec)


    def sample_buffer(self, batch_size, use_horizon=False, num_agents=None, get_entropy = False):
        max_mem = min(self.mem_ctr, self.mem_size)
        if not use_horizon:
            batch = np.random.choice(max_mem, batch_size, replace = False)
            states = self.state_memory[batch]
            actions = self.action_memory[batch]
            rewards = self.reward_memory[batch]
            next_states = self.new_state_memory[batch]
            dones = self.terminal_memory[batch]
            if get_entropy:
                states_vec, messages_vec = self.entropy_memory.sample_buffer(batch, use_horizon)
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
            if get_entropy:
                states_vec, messages_vec = self.entropy_memory.sample_buffer(batch_start, use_horizon, batch_end)
        if get_entropy:
            return states, actions, rewards, next_states, dones, states_vec, messages_vec
        return states, actions, rewards, next_states, dones, None, None

    class EntropyBuffer():
        def __init__(self, max_size, state_size, num_agents):
            self.mem_size = max_size
            self.mem_ctr = 0
            self.state_memory = np.zeros((self.mem_size, state_size), dtype = np.int64)
            self.message_memory = np.zeros((self.mem_size, num_agents), dtype = np.int64)

        def store_transition(self, state_vec, message_vec):
            mem_index = self.mem_ctr % self.mem_size
            self.state_memory[mem_index] = state_vec
            self.message_memory[mem_index] = message_vec
            self.mem_ctr += 1

        def sample_buffer(self, batch_start, use_horizon, batch_end = None):
            if not use_horizon:
                states = self.state_memory[batch_start]
                messages = self.message_memory[batch_start]
            else:
                states = self.state_memory[batch_start:batch_end]
                messages = self.message_memory[batch_start:batch_end]
            return states, messages
