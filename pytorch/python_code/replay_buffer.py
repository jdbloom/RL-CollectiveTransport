import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, num_observations, num_actions, action_type = None, state_size = None, num_agents = None, use_intention = False, use_seq_buffer = False, seq_len = 5, num_sequence=100):
        self.mem_size = max_size
        self.use_intention = use_intention
        self.mem_ctr = 0
        self.action_type = action_type
        self.state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        self.seq_memory = None
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
        if use_seq_buffer:
            self.seq_memory = self.SequenceReplayBuffer(num_sequence, num_observations, num_actions, seq_len)


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
        return states, actions, rewards, next_states, dones

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

    class SequenceReplayBuffer:
        def __init__(self, max_sequence, num_observations, num_actions, seq_len):
            self.mem_size = max_sequence*seq_len
            self.num_observations = num_observations
            self.num_actions = num_actions
            self.seq_len = seq_len
            self.mem_ctr = 0
            self.seq_mem_cntr = 0

            #main buffer used for sampling
            self.state_memory = np.zeros((self.mem_size, self.num_observations), dtype = np.float64)
            self.action_memory = np.zeros((self.mem_size, self.num_actions), dtype = np.float64)
            self.new_state_memory = np.zeros((self.mem_size, self.num_observations), dtype = np.float64)
            self.reward_memory = np.zeros((self.mem_size), dtype = np.float64)
            self.terminal_memory = np.zeros((self.mem_size), dtype = np.bool)

            #sequence buffer stores 1 sequence of len seq_len, transfers seq to main buffer once full
            self.seq_state_memory = np.zeros((self.seq_len, self.num_observations), dtype=np.float64)
            self.seq_action_memory = np.zeros((self.seq_len, self.num_actions), dtype=np.float64)
            self.seq_new_state_memory = np.zeros((self.seq_len, self.num_observations), dtype = np.float64)
            self.seq_reward_memory = np.zeros((self.seq_len), dtype = np.float64)
            self.seq_terminal_memory = np.zeros((self.seq_len), dtype = np.bool)

        def store_seq_transition(self, s, a ,r, s_, d):

            if self.seq_mem_cntr == self.seq_len:
                #Transfer Seq to main mem and clear seq buffer
                self.store_transitions()
                self.seq_mem_cntr = 0
            self.seq_state_memory[self.seq_mem_cntr] = s
            self.seq_action_memory[self.seq_mem_cntr] = a[1]
            self.seq_new_state_memory[self.seq_mem_cntr] = s_
            self.seq_reward_memory[self.seq_mem_cntr] = r
            self.seq_terminal_memory[self.seq_mem_cntr] = d
            self.seq_mem_cntr += 1

        def store_transitions(self):
            mem_index = self.mem_ctr % self.mem_size
            for i in range(self.seq_len):
                self.state_memory[mem_index+i] = self.seq_state_memory[i]
                self.action_memory[mem_index+i] = self.seq_action_memory[i]
                self.new_state_memory[mem_index+i] = self.seq_new_state_memory[i]
                self.reward_memory[mem_index+i] = self.seq_reward_memory[i]
                self.terminal_memory[mem_index+i] = self.seq_terminal_memory[i]
            self.mem_ctr += self.seq_len

        def sample_memory(self, batch_size, replace=True):
            max_mem = min(self.mem_ctr, self.mem_size)
            #selecting starting indices of the sequence in buffer
            indices = [x*self.seq_len for x in range((max_mem//self.seq_len)-1)]
            samples_indices = np.random.choice(indices, batch_size, replace = replace)
            s = np.zeros((batch_size,self.seq_len,self.num_observations))
            s_ = np.zeros((batch_size,self.seq_len,self.num_observations))
            a = np.zeros((batch_size,self.seq_len,self.num_actions))
            r = np.zeros((batch_size, self.seq_len), dtype= np.float64)
            d = np.zeros((batch_size, self.seq_len), dtype= np.bool)
            for i,j in enumerate(samples_indices):
                s[i] = self.state_memory[j:j+self.seq_len]
                s_[i] = self.new_state_memory[j:j+self.seq_len]
                a[i] = self.action_memory[j:j+self.seq_len]
                r[i] = self.reward_memory[j:j+self.seq_len]
                d[i] = self.terminal_memory[j:j+self.seq_len]
            return s, a, r, s_, d, None, None