import python_code.Agent as Agent
import python_code.zmq_utility as zmq_utility

import argparse
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import math
import copy
import zmq
import csv
import os

Utility = zmq_utility.ZMQ_Utility()

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
Utility.get_params(socket.recv())
print("PARAMETERS:")
print("  num_robots ----", Utility.params['num_robots'])
print("  num_obs -------", Utility.params['num_obs'])
print("  alphabet_size -", Utility.params['alphabet_size'])
print("  num_actions ---", Utility.params['num_actions'])
print("  num_stats -----", Utility.params['num_stats'])
# Path to save data
data_file_path = recording_path + '/Data/'

model = Agent.Agent(Utility.params['num_robots'],
                    Utility.params['num_obs'],
                    Utility.params['num_actions'] - 1, # -1 to account for gripper
                    num_ops_per_action = 3,
                    id = 0,
                    learning_scheme = learning_scheme,
                    comms_scheme = comms_scheme,
                    alphabet_size = Utility.params['alphabet_size'])
if test_mode:
    model.load_model(model_file_path)

# Send acknowledgment
socket.send(b"ok")

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
    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
    data_file_name = 'Data_Episode_'+str(ep_counter)+'.csv'
    with open(data_file_path+data_file_name, 'w') as output:
        writer = csv.writer(output, delimiter = ',')
        writer.writerow(['reward', 'epsilon', 'termination', 'force magnitude', 'force angle', 'average force vector'])

        if not exp_done:
            time_steps = 0
            # Receive initial observations from the environment
            env_observations = Utility.parse_obs(msgs[1])
            failures = Utility.parse_failures(msgs[2])
            rewards = Utility.parse_rewards(msgs[3])
            stats = Utility.parse_stats(msgs[4])

            agent_states = []
            force_mags = []
            force_angs = []
            running_reward = 0

            for i in range(Utility.params['num_robots']):
                # append env observations and messages in inbox to make agent state
                agent_state = model.make_agent_state(env_observations[i], i)
                agent_states.append(agent_state)
                force_mags.append(stats[i][0])
                force_angs.append(stats[i][1])
            # reward is the same across all agents. If it were per agent then this would need to move into the loop above
            running_reward += rewards[0]
            # failures should all be false because we havent started the episode yet
            failure = failures[0]

            #
            # Start the Episode Loop
            #
            while not episode_done:
                if not exp_done:
                    reward = []
                    actions = []
                    actions_to_take = []
                    messages = []
                    message_codes = []
                    time_steps += 1
                    # Get Actions
                    #print('-----------------')
                    for i in range(Utility.params['num_robots']):
                        # Choose an action
                        action, action_num = model.choose_action(agent_states[i], failure, test_mode)
                        actions_to_take.append(action)
                        actions.append(action_num)
                        #print(i, action)
                        # Choose a message
                        if model.comms_scheme != 'None':
                            message, message_num = model.choose_message(agent_states[i], failure, test_mode)
                            # Schedule the message to neighbors
                            model.schedule_message_to_all_contacts(i, message_num)
                            message_codes.append(message_num)

                    # Carry scheduled messages
                    if model.comms_scheme != 'None':
                        model.carry_mail()

                    old_failures = failures[:]
                    # Take Step
                    socket.send(Utility.serialize_actions(actions_to_take))
                    msgs = socket.recv_multipart()
                    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
                    env_observations = Utility.parse_obs(msgs[1])
                    failures = Utility.parse_failures(msgs[2])
                    rewards = Utility.parse_rewards(msgs[3])
                    stats = Utility.parse_stats(msgs[4])

                    # Store Transitions and Learn
                    new_agent_states = []
                    force_mags = []
                    force_angs = []
                    r = []

                    for i in range(Utility.params['num_robots']):
                        reward = rewards[i]
                        force_mags.append(stats[i][0])
                        force_angs.append(stats[i][1])
                        new_agent_state = model.make_agent_state(env_observations[i], i)
                        new_agent_states.append(new_agent_state)

                        if train_mode:
                            if learning_scheme != 'None':
                                if not old_failures[i] and not failures[i]:
                                    model.store_transition(agent_states[i],
                                                           (actions[i], actions_to_take[i]),
                                                           reward,
                                                           new_agent_states[i],
                                                           episode_done)
                                    if model.comms_scheme != 'None':
                                        model.store_comms_transition(agent_states[i],
                                                                     (message_codes[i] - 1, None),
                                                                     reward,
                                                                     new_agent_states[i],
                                                                     episode_done)
                        r.append(reward[0])
                    if train_mode:
                        model.learn()
                        if model.comms_scheme != 'None':
                            model.learn_comms()

                    running_reward += reward
                    # Store New Observations
                    agent_states = new_agent_states
                    actions = []
                    message_codes = []

                    # Calculate average force vector
                    average_force_mag = None
                    average_force_ang = None
                    for i in range(Utility.params['num_robots']):
                        if average_force_mag is None:
                            average_force_mag = force_mags[i]
                            average_force_ang = force_angs[i]
                        else:
                            angle = abs(average_force_ang - force_angs[i])
                            average_force_mag = math.sqrt(average_force_mag**2 + force_mags[i]**2 + 2*(average_force_mag)*(force_mags[i])*math.cos(math.radians(angle)))
                            average_force_ang = math.asin(force_mags[i]*math.sin(math.radians(180 - angle)) / average_force_mag)

                    writer.writerow([r, model.epsilon, reached_goal, force_mags, force_angs, [average_force_mag, math.degrees(average_force_ang)]])

                    if episode_done:
                        exp_rewards.append(running_reward)
                        if not reached_goal:
                            print("Episode", ep_counter ,"timed out")
                        else:
                            print("Episode", ep_counter ,"reached goal")
                        for i in range(Utility.params['num_robots']):
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
                                file_name = 'High_Score'
                                path = model_file_path+file_name
                                model.save_model(path)
                            file_name = 'Episode_'+str(ep_counter)
                            path = recording_path + "/Models/" +file_name
                            if train_mode:
                                model.save_model(path)
                            print('reward last 10 eps:%.2f'%exp_mean_rewards[-1],'\n')
                        ep_counter += 1
                        running_reward = 0
                        # Send acknowledgment
                        socket.send(b"ok")
print("Experiment Done\n")
