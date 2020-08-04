#!/usr/bin/env python3

import python_code.DQNAgent as Agent_DQN
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import copy
import zmq
import csv

#
# Message fields
#
# Parameters
PARAMS_FIELDS = ['num_robots','num_obs','num_actions']
PARAMS_FMT = '3I'
# Episode state
EXPERIMENT_FIELDS = ['exp_done', 'episode_done', 'reached_goal']
EXPERIMENT_FMT = '3B'
# Observations
OBS_FIELDS = ['robot_dist2goal', 'robot_angle2goal', 'robot_lwheel', 'robot_rwheel', 'cyl_dist2robot', 'cyl_angle2robot', 'cyl_dist2goal']
OBS_FMT = '7f'
# Rewards
REWARDS_FIELDS = ['reward']
REWARDS_FMT = '1f'
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
    reached_goal = (data['reached_goal'] == 1)
    return exp_done, episode_done, reached_goal

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

def parse_rewards(msg):
    rewards = []
    for r in range(0, params['num_robots']):
        # Get message bytes for this robot
        m = msg[r*FLOAT_SIZE:(r+1)*FLOAT_SIZE]
        # Parse the bytes into a dictionary
        data = parse_msg(m, 'reward', REWARDS_FIELDS, REWARDS_FMT)
        # Make a numpy array
        nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        # Append it to the rewards
        rewards.append(nparr)
    return rewards

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
test = False
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
# Path to save/ load models:
model_file_path = 'python_code/Data/test/Models/'
data_file_path = 'python_code/Data/test/Data/'
# Create the models for multi-agent individual model
#models = [Agent_DQN.Agent_DQN(params['num_robots'], params['num_obs'],params['num_actions'] , 3, i) for i in range(params['num_robots'])]
# Create Single Model
if test:
    for i, agent_model in enumerate(models):
        file_name = 'Model_'+str(i)+'_Episode_60'
        path = file_path+file_name
        agent_model.load_model(path)

# Send acknowledgment
ack()

def handle_communications(agent_models):
    for sender_id, sender_model in enumerate(models):
        for recepient, messages in sender_model.mailbox.outbox.items():
            for message in messages:
                models[recepient].mailbox.receive_message(message, sender_id)
        sender_model.mailbox.clear_outbox()
#
# Main loop
#
exp_done = False
ep_counter = 0
exp_rewards = []
exp_mean_rewards = []
epsilon = []
high_score = -np.inf
mean_axis = []

while not exp_done:
    # Main loop for episode
    # Recieve Initial Observations
    # It's in two parts
    # 1. whether the episode and the experiment are done
    # 2. the list of observations
    msgs = socket.recv_multipart()
    # Experiment or episode done?
    exp_done, episode_done, reached_goal = parse_status(msgs[0])

    data_file_name = 'Data_Episode_'+str(ep_counter)+'.csv'
    with open(data_file_path+data_file_name, 'w') as output:
        writer = csv.writer(output, delimiter = ',')
        writer.writerow(['reward', 'epsilon', 'loss', 'termination'])

        if not exp_done:
            time_steps = 0
            # Recieve initial Observation
            obs = parse_obs(msgs[1])
            rewards = parse_rewards(msgs[2])

            observations = []
            actions = []
            actions_to_take = []
            running_reward = 0

            for i , agent_model in enumerate(models):
                observations.append(obs[i])
                reward = rewards[i]
            running_reward+=reward

            while not episode_done:
                if not exp_done:

                    reward = []
                    epsilon = []
                    loss = []

                    time_steps += 1
                    # Get Actions
                    for i , agent_model in enumerate(models):
                        action, action_num = agent_model.choose_action(observations[i])
                        actions_to_take.append(action)
                        actions.append(action_num)
                    # Take Step
                    socket.send(serialize_actions(actions_to_take))
                    msgs = socket.recv_multipart()
                    exp_done, episode_done, reached_goal = parse_status(msgs[0])
                    obs = parse_obs(msgs[1])
                    rewards = parse_rewards(msgs[2])
                    # Store Transitions and Learn
                    new_observations = []
                    loss = []
                    r = [] # place holder to extract the values from the reward
                    # Collect all messages to be sent
                    handle_communications(models)

                    for i, agent_model in enumerate(models):
                        new_observations.append(obs[i])
                        reward = rewards[i]
                        # Handle sending and receiving messages here !!!

                        if not test:
                            agent_model.store_transition(observations[i],
                                                         actions[i],
                                                         reward,
                                                         new_observations[i],
                                                         episode_done)
                            loss.append(agent_model.doubleQLearn())
                        epsilon.append(agent_model.epsilon)
                        r.append(reward[0])

                    running_reward += reward
                    # Store New Observations
                    observations = new_observations
                    actions = []
                    actions_to_take = []

                    writer.writerow([r, epsilon, loss, reached_goal])


                    if episode_done:
                        ep_counter += 1
                        exp_rewards.append(running_reward)
                        if not reached_goal:
                            print("Episode", ep_counter ,"timed out")
                        else:
                            print("Episode", ep_counter ,"reached goal")

                        for i, agent_model in enumerate(models):
                            print('Agent', i, 'reward %.1f' % running_reward,
                                  'epsilon:%.2f' % agent_model.epsilon,
                                  'steps:', agent_model.learn_step_counter)
                            if ep_counter % 10 == 0:
                                exp_mean_rewards.append(np.mean(exp_rewards))
                                exp_rewards = []
                                if exp_mean_rewards[-1] > high_score:
                                    high_score = exp_mean_rewards[-1]
                                    print('****************************************')
                                    print('         NEW HIGH SCORE: %.2f'%high_score)
                                    print('****************************************')
                                    for i, agent_model in enumerate(models):
                                        file_name = 'Model_'+str(i)+'_High_Score'
                                        path = model_file_path+file_name
                                        agent_model.save_model(path)
                                file_name = 'Model_'+str(i)+'_Episode_'+str(ep_counter)
                                path = model_file_path+file_name
                                agent_model.save_model(path)
                                print('reward last 10 eps:%.2f'%exp_mean_rewards[-1],'\n')

                        running_reward = 0
                        ack()

#
# All done
#
#import ipdb; ipdb.set_trace()
print("Experiment done\n")
