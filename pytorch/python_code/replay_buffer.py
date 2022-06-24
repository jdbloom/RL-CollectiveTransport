import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, num_observations, num_actions, action_type = None, use_intention = False):
        self.mem_size = max_size
        self.use_intention = use_intention
        self.mem_ctr = 0
        self.action_type = action_type
        self.state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        if not use_intention:
            if self.action_type == 'Discrete':
                self.action_memory = np.zeros((self.mem_size), dtype = int)
            elif self.action_type == 'Continuous':
                self.action_memory = np.zeros((self.mem_size, num_actions), dtype = np.float32)
            else:
                raise Exception('Unknown Action Type:' + action_type)
        else:
            self.action_memory = np.zeros((self.mem_size), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size), dtype = np.float32)
        self.terminal_memory = np.zeros((self.mem_size), dtype = np.bool)


    def store_transition(self, state, action, reward, state_, done):
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
        

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace = False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones

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

    def store_transition(self, s, a ,r, s_, d):
        mem_index = self.mem_ctr % self.mem_size
        # import ipdb; ipdb.set_trace()
        self.seq_state_memory[self.seq_mem_cntr] = s
        self.seq_action_memory[self.seq_mem_cntr] = a
        self.seq_new_state_memory[self.seq_mem_cntr] = s_
        self.seq_reward_memory[self.seq_mem_cntr] = r
        self.seq_terminal_memory[self.seq_mem_cntr] = d
        self.seq_mem_cntr += 1
        
        if self.seq_mem_cntr == self.seq_len:
            #Transfer Seq to main mem and clear seq buffer
            for i in range(self.seq_len):
                self.state_memory[mem_index+i] = self.seq_state_memory[i]
                self.action_memory[mem_index+i] = self.seq_action_memory[i]
                self.new_state_memory[mem_index+i] = self.seq_new_state_memory[i]
                self.reward_memory[mem_index+i] = self.seq_reward_memory[i]
                self.terminal_memory[mem_index+i] = self.seq_terminal_memory[i]
            self.mem_ctr += self.seq_len
            self.seq_mem_cntr = 0

    def get_current_sequence(self):
        j = self.mem_ctr % self.mem_size
        s = self.state_memory[j-self.seq_len+1:j+1]
        s_ = self.new_state_memory[j-self.seq_len+1:j+1]
        a = self.action_memory[j-self.seq_len+1:j+1]
        r = self.reward_memory[j-self.seq_len+1:j+1]
        d = self.terminal_memory[j-self.seq_len+1:j+1]
        return s,s_,a,r,d

    def sample_buffer(self, batch_size, replace=True):
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
        return s, a, r, s_, d

class AttentionSequenceReplayBuffer:
    def __init__(self, max_sequence, num_observations, seq_len):
        self.mem_size = 10000
        self.num_observations = num_observations
        self.seq_len = seq_len
        self.mem_ctr = 0
        self.seq_mem_cntr = 0

        #main buffer used for sampling
        self.state_memory = np.zeros((self.mem_size, self.num_observations), dtype = np.float64)
        self.label_memory = np.zeros(self.mem_size, dtype = np.float64)

        #sequence buffer stores 1 sequence of len seq_len, transfers seq to main buffer once full
        self.seq_state_memory = np.zeros((self.seq_len, self.num_observations), dtype=np.float64)

    def store_transition(self, s, y):
        mem_index = self.mem_ctr % self.mem_size
        #import ipdb; ipdb.set_trace()
        self.seq_state_memory[self.seq_mem_cntr] = s
        self.seq_mem_cntr += 1
        if self.seq_mem_cntr == self.seq_len:
            #Transfer Seq to main mem and clear seq buffer
            for i in range(self.seq_len):
                self.state_memory[mem_index+i] = self.seq_state_memory[i]
            self.label_memory[mem_index] = y
            self.mem_ctr += self.seq_len
            self.seq_mem_cntr = 0

    def get_current_sequence(self):
        j = self.mem_ctr % self.mem_size
        s = self.state_memory[j-self.seq_len+1:j+1]
        y = self.label_memory[j-self.seq_len+1:j+1]
        return s,y

    def sample_buffer(self, batch_size, replace=True):
        max_mem = min(self.mem_ctr, self.mem_size)
        #selecting starting indices of the sequence in buffer
        indices = [x*self.seq_len for x in range((max_mem//self.seq_len)-1)]
        samples_indices = np.random.choice(indices, batch_size, replace = replace)
        s = np.zeros((batch_size,self.seq_len,self.num_observations))

        for i,j in enumerate(samples_indices):
            s[i] = self.state_memory[j:j+self.seq_len]
        y = self.label_memory[samples_indices] 
        return s, y