#!/usr/bin/env python3

#import agent_dqn
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import zmq

#
# Message fields
#
# Parameters
PARAMS_FIELDS = ['num_robots','num_obs','num_actions']
PARAMS_FMT = '3I'
# Episode state
EXPERIMENT_FIELDS = ['exp_done', 'episode_done']
EXPERIMENT_FMT = '2B'
# Observations
OBS_FIELDS = ['robot_dist2goal', 'robot_angle2goal', 'robot_lwheel', 'robot_rwheel', 'cyl_dist2robot', 'cyl_angle2robot', 'cyl_dist2goal', 'reward']
OBS_FMT = '8f'
# Actions
ACTIONS_FIELDS = ['lwheel', 'rwheel']
ACTIONS_FMT = '2f'

#
# Other constants
#
# Byte size of float in C++
FLOAT_SIZE = 4

#
# Parse the fields from a message
# Returns a dictionary
#
def parse_msg(msg, msgtype, fields, fmt):
    # Make an empty named tuple with the given fields
    Tx = namedtuple(msgtype, fields)
    # Fill the tuple with the contents of the message
    x = Tx._make(unpack(fmt, msg))
    # Return the tuple as a dictionary
    return x._asdict()

#
# Parse the experiment status
# Returns a tuple (exp_done, episode_done)
#
def parse_status(msg):
    data = parse_msg(msgs[0], 'status', EXPERIMENT_FIELDS, EXPERIMENT_FMT)
    exp_done = (data['exp_done'] == 1)
    episode_done = (data['episode_done'] == 1)
    return exp_done, episode_done

#
# Parse the observations
# Returns a list of numpy arrays, one per robot
#
def parse_obs(msg):
    obs = []
    # For each robot
    for r in range(0, params['num_robots']):
        # Get message bytes for this robot
        m = msg[r * params['num_obs'] * FLOAT_SIZE:(r+1) * params['num_obs'] * FLOAT_SIZE]
        # Parse the bytes into a dictionary
        data = parse_msg(m, 'obs', OBS_FIELDS, OBS_FMT)
        # Make a numpy array
        nparr = np.fromiter(data.values(), dtype=np.float32, count=len(data))
        # Append it to the observations
        obs.append(nparr)
    return obs

def serialize_actions(actions):
    packer = Struct(ACTIONS_FMT)
    msg = bytearray(FLOAT_SIZE * params['num_actions'] * params['num_robots'])
    # For each robot
    for r in range(0, params['num_robots']):
        offset = FLOAT_SIZE * params['num_actions'] * r
        packer.pack_into(msg, offset, *(actions[r]))
    return msg

#
# Send an acknowledgment to the client
#
def ack():
    socket.send(b"ok")
    
#
# Initialization
#
# Create context
context = zmq.Context()
# Create socket
socket = context.socket(zmq.REP)
# Wait for connections on port 55555
socket.bind("tcp://*:55555")
print("Server started")
# Get parameters
params = parse_msg(socket.recv(), 'params', PARAMS_FIELDS, PARAMS_FMT)
print("PARAMETERS:")
print("  num_robots  =", params['num_robots'])
print("  num_obs     =", params['num_obs'])
print("  num_actions =", params['num_actions'])
# Create the models
# TODO why doesn't the model take the number of actions as input? It's hardcoded in the Agent_DQN class...
#models = [agent_dqn.Agent_DQN(params['num_robots'], params['num_obs'], i) for i in range(params['num_robots'])]
# Send acknowledgment
ack()

#
# Main loop
#
exp_done = False
while not exp_done:
    # Main loop for episode
    episode_done = False
    while not episode_done:
        # Receive the message
        # It's in two parts
        # 1. whether the episode and the experiment are done
        # 2. the list of observations
        msgs = socket.recv_multipart()
        # Experiment or episode done?
        exp_done, episode_done = parse_status(msgs[0])
        if not exp_done:
            # Receive the observations
            obs = parse_obs(msgs[1])
            if not episode_done:
                # Learn
                # TODO
                # Send actions
                # TODO here I expect a list of numpy arrays, one per robot
                # PLACEHOLDER code
                actions = [np.array([1.0, 0.0]) for r in range(0, params['num_robots'])]
                socket.send(serialize_actions(actions))
            else:
                print("Episode done\n")
                ack()

#
# All done
#
print("Experiment done\n")
