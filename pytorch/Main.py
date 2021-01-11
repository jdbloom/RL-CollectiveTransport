import python_code.Agent as Agent
import python_code.parser as Parser

import argparse
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import math
import copy
import zmq
import csv
import os


# get path to containing folder so this works where ever it is used
containing_folder = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--learning_scheme")
parser.add_argument("--comms_scheme", default = "Neighbors")
parser.add_argument("--test", default = False, action = "store_true")
parser.add_argument("--port", default = "55555")
parser.add_argument("--model_path")
args = parser.parse_args()

recording_path = os.path.join(containing_folder, args.recording_path)
if args.model_path is not None:
    model_file_path = os.path.join(containing_folder, args.model_path)
learning_scheme = args.learning_scheme
comms_scheme = args.comms_scheme
port = args.port
test_mode = args.test
train_mode = not test_mode

Parser = Parser()

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
# Initialize zmq
#
# Create context
context = zmq.Context()
# create socket
socket = context.socket(zmq.REP)
# wait for connections on specified port, defaults to 55555
socket.bind("tcp://*:" + port)
print("Server Started")
# Get Parameters
params = Parser.parse_msg(socket.recv(), 'params', PARAMS_FIELDS, PARAMS_FMT)
print("PARAMETERS:")
print("  num_robots ----", params['num_robots'])
print("  num_obs -------", params['num_obs'])
print("  alphabet_size -", params['alphabet_size'])
print("  num_actions ---", params['num_actions'])
print("  num_stats -----", params['num_stats'])
# Path to save data
data_file_path = recording_path + '/Data/'

model = Agent.Agent(params['num_robots'],
                    params['num_obs'],
                    params['num_actions'] - 1, # -1 to account for gripper
                    num_ops_per_action = 3,
                    id = 0,
                    learning_scheme = learning_scheme,
                    comm_scheme = comm_scheme,
                    alphabet_size = params['alphabet_size'])
if test_mode:
    model.load_model(model_file_path)

# Send acknowledgment
ack()

#######################################################################
#                           MAIN LOOP
#######################################################################
exp_done = False
ep_counter = 0
exp_rewards = []
exp_mean_rewards = []
high_score = -np.inf
mean_axis = []

while not exp_done:
    #receive initial observations
    msgs = socket.recv_multipart()
    exp_done, episode_done, reached_goal = Parser.parse_status(msgs[0])
    data_file_name = 'Data_Episode_'+str(ep_counter)+'.csv'
    with open(data_file_path+data_file_name, 'w') as output:
        writer = csv.writer(output, delimiter = ',')
        writer.writerow(['reward', 'epsilon', 'termination', 'force magnitude', 'force angle', 'average force vector'])

        if not exp_done:
            time_steps = 0
            # Receive initial observations from the environment
            env_observations = Parser.parse_obs(msgs[1])
            failures = Parser.parse_failures(msgs[2])
            rewards = Parser.parse_rewards(msgs[3])
            stats = Parser.parse_stats(msgs[4])

            agent_states = []
            actions = []
            messages = []
            message_codes = []
            force_mags = []
            force_angs = []
            actions_to_take = []
            running_reward = 0

            for i in range(params['num_robots']):
                # append env observations and messages in inbox to make agent state
                agent_state = model.make_agent_state(observations[i], i)
                agent_states.append(agent_state)
                force_mags.append(stats[i][0])
                force_angs.append(stats[i][1])
            # reward is the same across all agents. If it were per agent then this would need to move into the loop above
            running_reward += rewards[0]
            # failures should all be false because we havent started the episode yet
            failure = failure[0]

            #
            # Start the Episode Loop
            #
            while not episode_done:
                if not exp_done:
                    reward = []
                    time_steps += 1

                    # Get Actions
                    for i in range(params['num_robots']):
                        # Choose an action
                        action, action_num = model.choose_action(agent_states[i], failure, test_mode)
                        # Choose a message
                        message, message_num = model.choose_messgae(agent_states[i], failure, test_mode)
                        # Schedule the message to neighbors
                        model.schedule_message_to_all_contacts(i, message)
                        message_codes.append(message_num)
                        actions_to_take.append(action)
                        actions.append(action_num)
                    # Carry scheduled messages
                    model.carry_mail()

                    old_failures = failures[:]
                    # Take Step
                    socket.send(serialize_actions(actions_to_take))
                    msgs = socket.recv_multipart()
                    exp_done, episode_done, reached_goal = Parser.parse_status(msgs[0])
                    env_observationso = Parser.parse_obs(msgs[1])
                    failures = Parser.parse_failures(msgs[2])
                    rewards = Parser.parse_rewards(msgs[3])
                    stats = Parser.parse_stats(msgs[4])

                    # Store Transitions and Learn
                    new_agent_states = []
                    force_mags = []
                    force_angs = []
                    r = []

                    for i in range(params['num_robots']):
                        reward = rewards[i]
                        force_mags.append(stats[i][0])
                        force_angs.append(stats[i][1])
                        new_agent_state = model.make_agent_state(env_observations[i], i)
                        new_agent_states.append(new_agent_state)
                        if train_mode:
                            if not old_failures[i] and not failures[i]:
                                model.store_transition(agent_states[i],
                                                       actions[i],
                                                       reward,
                                                       new_agent_states[i],
                                                       episode_done)
                                model.store_comms_transition(agent_states[i],
                                                             message_codes[i] - 1,
                                                             reward,
                                                             new_agent_states[i],
                                                             episode_done)
                        r.append(reward[0])
                    if train_mode:
                        model.learn()
                        model.learn_comms()

                    running_reward += reward
                    # Store New Observations
                    agent_states = new_agent_states
                    actions = []
                    message_codes = []

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
                        ep_counter += 1
                        running_reward = 0
                        ack()
print("Experiment Done\n")
