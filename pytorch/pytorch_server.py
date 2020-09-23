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


parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--model_path")
args = parser.parse_args()

recording_path = args.recording_path

#
# Message fields
#
# Parameters
PARAMS_FIELDS = ['num_robots','num_obs','num_actions', 'num_stats']
PARAMS_FMT = '4I'
# Episode state
EXPERIMENT_FIELDS = ['exp_done', 'episode_done', 'reached_goal']
EXPERIMENT_FMT = '3B'
# Observations
OBS_FIELDS = ['robot_dist2goal', 'robot_angle2goal', 'robot_lwheel', 'robot_rwheel', 'cyl_dist2robot', 'cyl_angle2robot', 'cyl_dist2goal', 'failed']
OBS_FMT = '8f'
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
test = args.test
# Flag for CSRL or ILRL
SingleModel = True
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
print("  num_stats   =", params['num_stats'])
# Path to save/ load models:
model_file_path = args.model_path
data_file_path = recording_path + '/Data/'

# num_obs - 1 is to exclude the "failed" observation from the neural network
# num_actions -1 is to exclude control of the gripper from the neural network
if SingleModel:
    # Create Single Model
    # +1 extra obs for communications, -1 for failure code
    model = Agent_DQN.Agent_DQN(params['num_robots'], params['num_obs'], params['num_actions'] - 1, 3, 0)
else:
    # Create the models for multi-agent individual model
    models = [Agent_DQN.Agent_DQN(params['num_robots'], params['num_obs'], params['num_actions'] - 1, 3, i) for i in range(params['num_robots'])]

if test:
    if SingleModel:
        model.load_model(model_file_path)
    else:
        for i, agent_model in enumerate(models):
            model.load_model(model_file_path)

# Send acknowledgment
ack()

def insert_communications(obs, agent_id):
    # Insert incoming comms into obs
    incoming_comms = model.get_agent_incoming_communications(agent_id)
    if len(obs) <= len(OBS_FIELDS):
        obs = np.concatenate([obs[:-1],
                              incoming_comms,
                              [obs[-1]]])
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
            rewards = parse_rewards(msgs[2])
            stats = parse_stats(msgs[3])

            observations = []
            actions = []
            force_mags = []
            force_angs = []
            actions_to_take = []
            running_reward = 0

            for i in range(params['num_robots']):
                observations.append(obs[i])
                reward = rewards[i]
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
                            # Insert incoming comms into obs
                            """
                            incoming_comms = model.get_agent_incoming_communications(i)
                            model.clear_agent_inbox(i)
                            observations[i] = np.concatenate([observations[i][:-1],
                                                              incoming_comms,
                                                              [observations[i][-1]]])
                            """
                            observations[i] = insert_communications(observations[i], i)
                            model.clear_agent_inbox(i)
                            action, action_num, outgoing_message = model.choose_action(observations[i], test)
                            # Schedule this agent's messages to send
                            model.schedule_message_to_all_contacts(i, outgoing_message)
                            actions_to_take.append(action)
                            actions.append(action_num)
                    else:
                        for i , agent_model in enumerate(models):
                            action, action_num = agent_model.choose_action(observations[i], test)
                            actions_to_take.append(action)
                            actions.append(action_num)

                    # Once everyone's acted, deliver the mail
                    if SingleModel:
                        model.carry_mail()

                    # Take Step
                    socket.send(serialize_actions(actions_to_take))
                    msgs = socket.recv_multipart()
                    exp_done, episode_done, reached_goal = parse_status(msgs[0])
                    obs = parse_obs(msgs[1])
                    rewards = parse_rewards(msgs[2])
                    stats = parse_stats(msgs[3])
                    # Store Transitions and Learn
                    new_observations = []
                    loss = []
                    force_mags = []
                    force_angs = []
                    r = [] # place holder to extract the values from the reward

                    for i in range(params['num_robots']):
                        # Insert incoming comms into obs
                        obs[i] = insert_communications(obs[i], i)
                        # Don't clear the inbox this time, we'll use those messages next timestep
                        """
                        incoming_comms = model.get_agent_incoming_communications(i)
                        obs[i] = np.concatenate([obs[i][:-1],
                                                 incoming_comms,
                                                 [obs[i][-1]]])
                        """
                        new_observations.append(obs[i])
                                                
                        reward = rewards[i]
                        force_mags.append(stats[i][0])
                        force_angs.append(stats[i][1])
                        if SingleModel:
                            if not test:
                                if not observations[i][-1] and not new_observations[i][-1]:
                                    model.store_transition(observations[i][:-1],
                                                                 actions[i],
                                                                 reward,
                                                                 new_observations[i][:-1],
                                                                 episode_done)

                            epsilon.append(model.epsilon)
                            r.append(reward[0])

                    if not SingleModel:
                        for i, agent_model in enumerate(models):
                            if not test:
                                agent_model.store_transition(observations[i][:-1],
                                                             actions[i],
                                                             reward,
                                                             new_observations[i],
                                                             episode_done)
                                loss.append(agent_model.doubleQLearn())
                            epsilon.append(agent_model.epsilon)
                            r.append(reward[0])

                    if not test:
                        if SingleModel:
                            model.doubleQLearn()
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
