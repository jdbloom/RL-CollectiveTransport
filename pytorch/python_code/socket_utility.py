from collections import namedtuple
from struct import pack, unpack, Struct
import struct
import numpy as np
import socket

class Socket_Utility:
    def __init__(self):
        self.PARAMS_FIELDS = ['num_robots','num_obs','num_actions', 'num_stats', 'alphabet_size']
        self.PARAMS_FMT = '!5I'
        self.EXPERIMENT_FIELDS = ['exp_done', 'episode_done', 'reached_goal']
        self.EXPERIMENT_FMT = '!3h'
        self.OBS_FIELDS = ['robot_dist2goal', 'robot_angle2goal', 'robot_lwheel',
                           'robot_rwheel', 'cyl_dist2robot', 'cyl_angle2robot', 'cyl_dist2goal',
                           'ProxVal_0',  'ProxVal_1',  'ProxVal_2',  'ProxVal_3',
                           'ProxVal_4',  'ProxVal_5',  'ProxVal_6',  'ProxVal_7',
                           'ProxVal_8',  'ProxVal_9',  'ProxVal_10', 'ProxVal_11',
                           'ProxVal_12', 'ProxVal_13', 'ProxVal_14', 'ProxVal_15',
                           'ProxVal_16', 'ProxVal_17', 'ProxVal_18', 'ProxVal_19',
                           'ProxVal_20', 'ProxVal_21', 'ProxVal_22', 'ProxVal_23']
        self.OBS_FMT = '!31f'
        self.FAILURE_FIELDS = ['failure']
        self.FAILURE_FMT = '!1I'
        self.REWARDS_FIELDS = ['reward']
        self.REWARDS_FMT = '!1f'
        self.STATS_FIELDS = ['magnitude', 'angle']
        self.STATS_FMT = '!2f'
        self.ACTIONS_FIELDS = ['lwheel', 'rwheel', 'failure']
        self.ACTIONS_FMT = '!3f'

        # Byte size of float in C++
        self.FLOAT_SIZE = 4
        # Byte size of int in C++
        self.INT_SIZE = 4

    def get_params(self, msg):
        self.params = self.parse_msg(msg, 'params', self.PARAMS_FIELDS, self.PARAMS_FMT)
    #
    # Parse the fields from a message
    # Returns a dictionary
    #
    def parse_msg(self, msg, msgtype, fields, fmt):
        # Make an empty named tuple with the given fields
        #print('[DEBUG]', msg)
        Tx = namedtuple(msgtype, fields)
        # Fill the tuple with the contents of the message
        x = Tx._make(unpack(fmt, msg))
        # Return the tuple as a dictionary
        #print('[DEBUG]', x, fmt)
        return x._asdict()

    #
    # Parse the experiment status
    # Returns a tuple (exp_done, episode_done, reached_goal)
    #
    def parse_status(self, msg):
        data = self.parse_msg(msg, 'status', self.EXPERIMENT_FIELDS, self.EXPERIMENT_FMT)
        exp_done = (data['exp_done'] == 1)
        episode_done = (data['episode_done'] == 1)
        reached_goal = (data['reached_goal'] == 1)
        return exp_done, episode_done, reached_goal

    #
    # Parse the observations
    # Returns a list of numpy arrays, one per robot
    #
    def parse_obs(self, msg):
        obs = []
        # For each robot
        for r in range(0, self.params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * self.params['num_obs'] * self.FLOAT_SIZE:(r+1) * self.params['num_obs'] * self.FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'obs', self.OBS_FIELDS, self.OBS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count=len(data))
            # Append it to the observations
            obs.append(nparr)
        return obs

    def parse_failures(self, msg):
        failures = []
        for r in range(0, self.params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * self.INT_SIZE:(r+1)*self.INT_SIZE]
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'failure', self.FAILURE_FIELDS, self.FAILURE_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.intc, count = len(data))
            # Append it to the rewards
            failures.append(nparr)
        return failures

    def parse_rewards(self, msg):
        rewards = []
        for r in range(0, self.params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * self.FLOAT_SIZE:(r+1)*self.FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'reward', self.REWARDS_FIELDS, self.REWARDS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
            # Append it to the rewards
            rewards.append(nparr)
        return rewards

    def parse_stats(self, msg):
        stats = []
        for r in range(0, self.params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * self.params['num_stats'] * self.FLOAT_SIZE:(r+1) * self.params['num_stats'] * self.FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'stats', self.STATS_FIELDS, self.STATS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
            # Append it to the rewards
            stats.append(nparr)
        return stats

    def serialize_actions(self, actions):
        packer = Struct(self.ACTIONS_FMT)
        msg = bytearray(self.FLOAT_SIZE * self.params['num_actions'] * self.params['num_robots'])
        # For each robot
        for r in range(0, self.params['num_robots']):
            offset = self.FLOAT_SIZE * self.params['num_actions'] * r
            packer.pack_into(msg, offset, *(actions[r]))
        return msg

class Server:
    class Msg:
        def __init__(self, ps = 0, m = 0):
            self.payload_size = ps
            self.more = m
            self.HEADER_SIZE = 3

        #@staticmethod
        def parse(self, buf):
            if len(buf) != self.HEADER_SIZE:
                raise Exception("Wrong msg size")
            (ps, m) = struct.unpack("!HB", buf)
            return [ps, m]


    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection = None
        self.address = None


    def connect(self, HOST, PORT):
        self.socket.bind((HOST, PORT))
        self.socket.listen()
        self.connection, self.address = self.socket.accept()
        if self.connection == None:
            raise Exception('SERVER CONNECTION FAILED ON', HOST, PORT)
        print('Connected to', self.address)

    def recv(self, byte_size = 1024):
        data = self.connection.recv(byte_size)
        return data

    def send(self, msg):
        self.connection.sendall(msg)

    def ack(self):
        self.connection.sendall(b"ok")

    def send_multipart(self, payload, more = False):
        # Make Header
        #print('[DEBUG]', len(payload))
        h = struct.pack('!HB', len(payload), (1 if more else 0))
        # Send Header
        self.connection.sendall(h)
        # Send payload
        self.connection.sendall(payload)

    def recv_multipart(self):
        buf = []
        more = 1
        Msg = self.Msg()
        while more == 1:
            #print('[INFO] Receiving Header')
            hbuf = self.connection.recv(Msg.HEADER_SIZE)
            hmsg = Msg.parse(hbuf)
            more = hmsg[1]
            #print('[MSG]', more)
            #print('[MSG]', hmsg[0])
            #print('[INFO] Receiving Message')
            pl = self.connection.recv(hmsg[0])
            #print(pl, more)
            buf.append(pl)
        return buf
