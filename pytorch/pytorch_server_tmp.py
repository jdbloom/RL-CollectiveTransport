#!/usr/bin/env python3

import python_code.DQNAgent as Agent_DQN

import argparse
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import math
import copy
import zmq
import csv
import os

# get path to containing folder so this works whereever it's used
containing_folder = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--model_path")
parser.add_argument("--comm_scheme", default="None")
parser.add_argument("--port", default="55555")
args = parser.parse_args()

recording_path = os.path.join(containing_folder, args.recording_path)
if args.model_path is not None:
    model_file_path = os.path.join(containing_folder, args.model_path)
comm_scheme = args.comm_scheme
port = args.port

#
# Message fields
#
# Parameters
PARAMS_FIELDS = ['num_robots','num_obs','num_actions', 'num_stats', 'alphabet_size']
PARAMS_FMT = '5I'
# Episode state
EXPERIMENT_FIELDS = ['exp_done', 'episode_done', 'reached_goal']
EXPERIMENT_FMT = '3B'
# Observations
OBS_FIELDS = ['robot_dist2goal', 'robot_angle2goal', 'robot_lwheel', 'robot_rwheel', 'cyl_dist2robot', 'cyl_angle2robot', 'cyl_dist2goal',
              'ProxVal_0', 'ProxVal_1', 'ProxVal_2', 'ProxVal_3', 'ProxVal_4', 'ProxVal_5', 'ProxVal_6', 'ProxVal_7', 'ProxVal_8', 'ProxVal_9', 'ProxVal_10', 'ProxVal_11',
              'ProxVal_12', 'ProxVal_13', 'ProxVal_14', 'ProxVal_15', 'ProxVal_16', 'ProxVal_17', 'ProxVal_18', 'ProxVal_19', 'ProxVal_20', 'ProxVal_21', 'ProxVal_22', 'ProxVal_23']
OBS_FMT = '31f'
# Failures
FAILURE_FIELDS = ['failure']
FAILURE_FMT = '1I'
# Rewards
REWARDS_FIELDS = ['reward']
REWARDS_FMT = '1f'
# Stats
STATS_FIELDS = ['magnitude', 'angle']
STATS_FMT = '2f'
# Actions
#ACTIONS_FIELDS = ['lwheel', 'rwheel', 'broadcast', 'failure']
ACTIONS_FIELDS = ['lwheel', 'rwheel', 'failure']
ACTIONS_FMT = '3f'

#
# Other constants
#
# Byte size of float in C++
FLOAT_SIZE = 4
# Byte size of int in C++
INT_SIZE = 4
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

def parse_failures(msg):
    failures = []
    for r in range(0, params['num_robots']):
        # Get message bytes for this robot
        m = msg[r * INT_SIZE:(r+1)*INT_SIZE]
        # Parse the bytes into a dictionary
        data = parse_msg(m, 'failure', FAILURE_FIELDS, FAILURE_FMT)
        # Make a numpy array
        nparr = np.fromiter(data.values(), dtype=np.intc, count = len(data))
        # Append it to the rewards
        failures.append(nparr)
    return failures

def parse_rewards(msg):
    rewards = []
    for r in range(0, params['num_robots']):
        # Get message bytes for this robot
        m = msg[r * FLOAT_SIZE:(r+1)*FLOAT_SIZE]
        # Parse the bytes into a dictionary
        data = parse_msg(m, 'reward', REWARDS_FIELDS, REWARDS_FMT)
        # Make a numpy array
        nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        # Append it to the rewards
        rewards.append(nparr)
    return rewards

def parse_stats(msg):
    stats = []
    for r in range(0, params['num_robots']):
        # Get message bytes for this robot
        m = msg[r * params['num_stats'] * FLOAT_SIZE:(r+1) * params['num_stats'] * FLOAT_SIZE]
        # Parse the bytes into a dictionary
        data = parse_msg(m, 'stats', STATS_FIELDS, STATS_FMT)
        # Make a numpy array
        nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        # Append it to the rewards
        stats.append(nparr)
    return stats

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
# Test the code? or should we be learning?
test_mode = args.test
train_mode = not test_mode
# Flag for CSRL or ILRL
SingleModel = True
# Create context
context = zmq.Context()
# Create socket
socket = context.socket(zmq.REP)
# Wait for connections on specified port, defaults to 55555
socket.bind("tcp://*:" + port)
print("Server started")
# Get parameters
params = parse_msg(socket.recv(), 'params', PARAMS_FIELDS, PARAMS_FMT)
print("PARAMETERS:")
print("  num_robots  =", params['num_robots'])
print("  num_obs     =", params['num_obs'])
print("  alphabet_size  =", params['alphabet_size'])
print("  num_actions =", params['num_actions'])
print("  num_stats   =", params['num_stats'])
# Path to save/ load models:
data_file_path = recording_path + '/Data/'

# num_actions -1 is to exclude control of the gripper from the neural network
if SingleModel:
    # Create Single Model
    # -1 for failure code
    model = Agent_DQN.Agent_DQN(params['num_robots'], params['num_obs'], params['num_actions'] - 1, 3, 0, comm_scheme=comm_scheme, alphabet_size=params['alphabet_size'])
else:
    # Create the models for multi-agent individual model
    models = [Agent_DQN.Agent_DQN(params['num_robots'], params['num_obs'], params['num_actions'], 3, i) for i in range(params['num_robots'])]

if test_mode:
    if SingleModel:
        model.load_model(model_file_path)
    else:
        for i, agent_model in enumerate(models):
            model.load_model(model_file_path)

# Send acknowledgment
ack()
'''
def insert_communications(obs, agent_id):
    # Insert incoming comms into obs
    incoming_comms = model.get_agent_incoming_communications(agent_id)

    #Uncomment for debugging communications

    print("Observations (no comms):\n", obs[:])
    print("Left comms:\n", incoming_comms.left_comm)
    print("right_comm:\n",incoming_comms.right_comm)
    if len(obs) <= len(OBS_FIELDS):
        obs = np.concatenate([obs,
                              incoming_comms.left_comm,
                              incoming_comms.right_comm])
    #print("Observations (with comms):\n", obs)
    return obs
'''
def get_communications_input(obs, agent_id):
    #assert(len(obs) == )
    # Insert incoming comms into obs
    incoming_comms = model.get_agent_incoming_communications(agent_id)

    #Uncomment for debugging communications

    print("Observations (no comms):\n", obs[:])
    print("Left comms:\n", incoming_comms.left_comm)
    print("right_comm:\n",incoming_comms.right_comm)
    '''
    '''
    if len(obs) <= len(OBS_FIELDS):
        obs = np.concatenate([obs,
                              incoming_comms.left_comm,
                              incoming_comms.right_comm])
    #print("Observations (with comms):\n", obs)
    return obs

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
        writer.writerow(['reward', 'epsilon', 'termination', 'force magnitude', 'force angle', 'average force vector'])

        if not exp_done:
            time_steps = 0
            # Recieve initial Observation
            obs = parse_obs(msgs[1])
            failures = parse_failures(msgs[2])
            rewards = parse_rewards(msgs[3])
            stats = parse_stats(msgs[4])

            observations = []
            comms_observations = []
            actions = []
            messages = []
            message_codes = []
            force_mags = []
            force_angs = []
            actions_to_take = []
            running_reward = 0

            for i in range(params['num_robots']):
                observations.append(obs[i])
                reward = rewards[i]
                failure = failures[i]
                force_mags.append(stats[i][0])
                force_angs.append(stats[i][1])
            running_reward+=reward

            while not episode_done:
                if not exp_done:

                    reward = []
                    epsilon = []
                    loss = []

                    time_steps += 1
                    # Get Actions
                    if SingleModel:
                        for i in range(params['num_robots']):
                            model.clear_agent_inbox(i)
                            action, action_num = model.choose_action(observations[i], failure, test_mode)
                            # Insert incoming comms into obs
                            observations_comms = get_communications_input(observations[i], i)
                            comms_observations.append(observations_comms)
                            outgoing_message, message_num = model.choose_message(observations_comms, failure, test_mode)
                            # Schedule this agent's messages to send
                            model.schedule_message_to_all_contacts(i, outgoing_message)
                            message_codes.append(message_num)
                            actions_to_take.append(action)
                            actions.append(action_num)
                    else:
                        for i , agent_model in enumerate(models):
                            action, action_num = agent_model.choose_action(observations[i], failure, test_mode)
                            actions_to_take.append(action)
                            actions.append(action_num)

                    if SingleModel:
                        model.carry_mail()

                    # Back up failure information
                    old_failures = failures[:]
                    # Take Step
                    socket.send(serialize_actions(actions_to_take))
                    msgs = socket.recv_multipart()
                    exp_done, episode_done, reached_goal = parse_status(msgs[0])
                    obs = parse_obs(msgs[1])
                    import ipdb; ipdb.set_trace()
                    new_comms_observations = [np.concatenate([o, messages[i]]) for i,o in enumerate(obs)]
                    failures = parse_failures(msgs[2])
                    rewards = parse_rewards(msgs[3])
                    stats = parse_stats(msgs[4])
                    # Store Transitions and Learn
                    new_observations = []
                    loss = []
                    force_mags = []
                    force_angs = []
                    r = [] # place holder to extract the values from the reward

                    for i in range(params['num_robots']):
                        # Don't clear the inbox this time, we'll use those messages next timestep
                        new_observations.append(obs[i])
                        reward = rewards[i]
                        force_mags.append(stats[i][0])
                        force_angs.append(stats[i][1])
                        if SingleModel:
                            if train_mode:
                                if not old_failures[i] and not failures[i]:
                                    print('Agent', i, 'message obs:', comms_observations[i][31:])
                                    model.store_transition(comms_observations[i],
                                                                 actions[i],
                                                                 reward,
                                                                 new_comms_observations[i],
                                                                 episode_done)
                                    #print(messages)
                                    #print(messages[i])
                                    print(i, comms_observations[i], message_codes[i]-1)
                                    model.store_comms_transition(comms_observations[i],
                                                                 message_codes[i] - 1,
                                                                 reward,
                                                                 new_comms_observations[i],
                                                                 episode_done)

                            epsilon.append(model.epsilon)
                            r.append(reward[0])

                    if not SingleModel:
                        for i, agent_model in enumerate(models):
                            if train_mode:
                                agent_model.store_transition(observations[i],
                                                             actions[i],
                                                             reward,
                                                             new_observations[i],
                                                             episode_done)
                                loss.append(agent_model.doubleQLearn())
                            epsilon.append(agent_model.epsilon)
                            r.append(reward[0])

                    if train_mode:
                        if SingleModel:
                            model.doubleQLearn()
                            model.doubleQLearnComms()
                        else:
                            for i, agent_model in enumerate(models):
                                agent_model.doubleQLearn()
                    running_reward += reward
                    # Store New Observations
                    observations = new_observations
                    actions = []
                    actions_to_take = []

                    # Calculate average force vector
                    average_force_mag = None
                    average_force_ang = None
                    for i in range(params['num_robots']):
                        if average_force_mag is None:
                            average_force_mag = force_mags[i]
                            average_force_ang = force_angs[i]
                        else:
                            angle = abs(average_force_ang - force_angs[i])
                            average_force_mag = math.sqrt(average_force_mag**2 + force_mags[i]**2 + 2*(average_force_mag)*(force_mags[i])*math.cos(math.radians(angle)))
                            average_force_ang = math.asin(force_mags[i]*math.sin(math.radians(180 - angle)) / average_force_mag)

                    writer.writerow([r, epsilon, reached_goal, force_mags, force_angs, [average_force_mag, math.degrees(average_force_ang)]])



                    if episode_done:

                        exp_rewards.append(running_reward)
                        if not reached_goal:
                            print("Episode", ep_counter ,"timed out")
                        else:
                            print("Episode", ep_counter ,"reached goal")
                        if SingleModel:
                            for i in range(params['num_robots']):
                                print('Agent', i, 'reward %.1f' % running_reward,
                                      'epsilon:%.2f' % model.epsilon,
                                      'steps:', model.learn_step_counter)
                            if ep_counter % 10 == 0:
                                exp_mean_rewards.append(np.mean(exp_rewards))
                                exp_rewards = []
                                if exp_mean_rewards[-1] > high_score and ep_counter > 1500:
                                    high_score = exp_mean_rewards[-1]
                                    print('****************************************')
                                    print('         NEW HIGH SCORE: %.2f'%high_score)
                                    print('****************************************')
                                    file_name = 'Model_'+str(i)+'_High_Score'
                                    path = model_file_path+file_name
                                    model.save_model(path)
                                file_name = 'Model_'+str(i)+'_Episode_'+str(ep_counter)
                                path = recording_path + "/Models/" +file_name
                                model.save_model(path)
                                print('reward last 10 eps:%.2f'%exp_mean_rewards[-1],'\n')
                        else:
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
                                        path = recording_path+"/Models/"+file_name
                                        agent_model.save_model(path)
                                file_name = 'Model_'+str(i)+'_Episode_'+str(ep_counter)
                                path = recording_path+"/Models/"+file_name
                                agent_model.save_model(path)
                                print('reward last 10 eps:%.2f'%exp_mean_rewards[-1],'\n')

                        ep_counter += 1

                        running_reward = 0
                        ack()

#
# All done
#
#import ipdb; ipdb.set_trace()
print("Experiment done\n")
