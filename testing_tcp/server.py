import socket
import struct
from struct import pack, unpack, Struct

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
            print('[INFO] Receiving Header')
            hbuf = self.connection.recv(Msg.HEADER_SIZE)
            hmsg = Msg.parse(hbuf)
            more = hmsg[1]
            print('[MSG]', more)
            print('[MSG]', hmsg[0])
            print('[INFO] Receiving Message')
            pl = self.connection.recv(hmsg[0])
            #print(pl, more)
            buf.append(pl)
        return buf

    def serialize_actions(self, actions):
        packer = Struct('!3f')
        msg = bytearray(4 * 3 * 4)
        # For each robot
        for r in range(0, 4):
            offset = 4 * 3 * r
            packer.pack_into(msg, offset, *(actions[r]))
        return msg

# Connect
print('[SERVER] Connecting to Client')
s = Server()
s.connect('127.0.0.1', 65432)
print('[SERVER] Connceted to Client')
print('[SERVER] Receiving Params')
# Get Params
params = s.recv_multipart()
print('[SERVER] Message has', len(params), 'Sub Messages')
params = struct.unpack('!5I', params[0])
numRobots = params[0]
numObs = params[1]
numActions = params[2]
numStats = params[3]
print('[MESSAGE] Num Robots    =', params[0])
print('[MESSAGE] Num Obs       =', params[1])
print('[MESSAGE] Num Actions   =', params[2])
print('[MESSAGE] Num Stats     =', params[3])
print('[MESSAGE] Alphabet Size =', params[4])
print('[SERVER] Sending ACK')
# Send Ack
s.send_multipart(b"ok")
# Receive Multipart:
#   Episode State
#   Env Observations
#   Failures
#   Rewards
#   Stats
print('[SERVER] Receiving Multipart')
msgs = s.recv_multipart()
state = struct.unpack('!3h', msgs[0])
obs_fmt = '!' + str(numObs*numRobots) + 'f'
print('[DEBUG] Obs Fmt', obs_fmt, len(msgs[1]))
obs = struct.unpack(obs_fmt, msgs[1])
fail_fmt = '!' + str(numRobots) + 'i'
print('[DEBUG] Fail Fmt', fail_fmt)
failures = struct.unpack(fail_fmt, msgs[2])
#reward_fmt = '!' + str(numRobots) + 'f'
#print('[DEBUG] Reward Fmt', reward_fmt)
#rewards = struct.unpack(reward_fmt, msgs[3])
#stats_fmt = '!' + str(numRobots*numStats)+'f'
#print('[DEBUG] Stats Fmt', stats_fmt)
#stats = struct.unpack(stats_fmt, msgs[4])
#print('[MESSAGE] State:', state)

observations = []
for i in range(numRobots):
    observations.append(obs[i*numObs:(i+1)*numObs])

print('[MESSAGE] Observations', observations, len(observations), len(observations[0]))
#print('[MESSAGE] Failures', failures, len(failures))
#print('[MESSAGE] Rewards', rewards, len(rewards))
#print('[MESSAGE] Stats', stats, len(stats))
# Send Actions
actions = []
for i in range(numRobots):
    actions.append((0.7, -1.1, 0))
print(actions)

s.send_multipart(s.serialize_actions(actions))

'''
with s.connection:
    data = s.recv_multipart()
    #print("DATA:", data)
    if not data:
        print('[ERROR] No Data')
    print('[DATA]', struct.unpack('!3I', data[0]), struct.unpack('!4I',data[1]))
    s.send_multipart(b'ok')
    #s.ack()
'''
