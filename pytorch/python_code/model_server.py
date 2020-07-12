# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:42:37 2020

@author: aaaaambition
"""
import numpy as np
import zmq
import struct
import agent_dqn
from utils import make_debug_print

# !!! Add timeouts !!!
class ModelInterface():
    def __init__(self, environment_port, listening_port):
        self.environment_port = environment_port
        self.listening_port =  listening_port
        self.num_agents = None
        self.size_obs = None
        self.agent_models = None
        self.debug_print = make_debug_print(set(["receive_obs"]))

    def receive_agent_obs(self, data):
        # This should somehow communicate this to the model
        for agent in range(self.num_agents):
            agent_obs = []
            done = None
            reward = None
            for obs in range(self.size_obs):
                i = agent*(self.size_obs) + obs
                val = struct.unpack("<f", data[4*i:(4*i) + 4])[0]
                if obs == self.size_obs - 2:
                    done = val
                elif obs == self.size_obs - 1:
                    reward = val
                else:
                    agent_obs.append(val)
            self.agent_models[agent].receive_observation(agent_obs, done, reward)

    def make_agent_actions(self):
        actions = []
        for agent in range(self.num_agents):
            agent_actions = self.agent_models[agent].make_action(test=False)
            actions.append(agent_actions)
        # Make a new array containing the agent id and actions so only one
        # send needs to be performed per action
        return np.hstack(actions)

    def listen_for_requests(self):
        HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
        PORT = self.listening_port

        SEND_AGENT_OBSERVATIONS = '0000807f'
        REQUEST_AGENT_ACTIONS = '000080ff'
        REQUEST_SERVER_CLOSE = '0000c07f'
        SEND_SERVER_INIT = '00000080'

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:" + str(PORT))

        while True:

            data = socket.recv()
            # First byte of data determines the request type
            request_type = data[0:4].hex()
            if request_type == SEND_SERVER_INIT:
                self.num_agents = int(struct.unpack("<f",data[4:8])[0])
                self.size_obs = int(struct.unpack("<f",data[8:12])[0])
                self.action_size = int(struct.unpack("<f", data[12:16])[0])
                self.agent_models = [agent_dqn.Agent_DQN(self.num_agents, self.size_obs, i) for i in range(self.num_agents)]
                socket.send(b'ok')

            elif request_type == SEND_AGENT_OBSERVATIONS:
                self.receive_agent_obs(data[4:(4 * (self.num_agents*self.size_obs + 1))])
                socket.send(b'ok')

            elif request_type == REQUEST_AGENT_ACTIONS:
                actions_arr = self.make_agent_actions()
                socket.send(_actions_to_bytes(actions_arr, size = self.num_agents*self.action_size))
            elif request_type == REQUEST_SERVER_CLOSE:
                print("Server requested to close, shutting down")
                socket.send(b'ok')
                break
            else:
                print("Unknown option: " + request_type + " in command position")
                raise Exception


def _bytes_to_np_arr(byts, dsize=4, dtype=int):
    """
    Converts a bytes object to a np array of dtype
    """
    arr = np.zeros(len(byts)//dsize, dtype=dtype)
    for j, i in enumerate(range(0,len(byts),dsize)):
        arr[j] = struct.unpack("<f",byts[i:i + dsize])[0]
    return arr

def _actions_to_bytes(actions, size=4):
    '''
    Given a np array of size 4, return a byte array of all the elements in seq.
    '''
    b = struct.pack(str(size) + "f", *actions)
    return b

def _int_to_bytes(i, signed=True):
    # Code copy and pasted from stack overflow
    length = ((i + ((i * signed) < 0)).bit_length() + 7 + signed) // 8
    return i.to_bytes(length, byteorder='little', signed=signed)

if __name__ == "__main__":
    eint = ModelInterface(5555,5555)
    eint.listen_for_requests()
