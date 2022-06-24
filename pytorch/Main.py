from urllib.parse import uses_relative
import python_code.Agent as Agent
import python_code.zmq_utility as zmq_utility
#from python_code.comms_viz import viz

import argparse
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import math
import copy
import zmq
import csv
import os
import time
import torch as T
import matplotlib.pyplot as plt

Utility = zmq_utility.ZMQ_Utility()

# get path to containing folder so this works where ever it is used
containing_folder = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--learning_scheme")
parser.add_argument("--test", default = False, action = "store_true")
parser.add_argument("--model_path")
parser.add_argument("--trained_num_robots")                                          # if we are testing a model trained on a different number of robots. This should be set to the training number of robots so that the network is built properly.
parser.add_argument("--no_print", default = False, action = "store_true")
parser.add_argument("--port", default = "55555")
parser.add_argument("--intention", default=False, action = "store_true")
parser.add_argument("--independent_learning", default = False, action = "store_true")
parser.add_argument("--recurrent", default= False, action= 'store_true')
parser.add_argument("--attention", default= False, action= 'store_true')
parser.add_argument("--meta_param_size", default=0, type=int)
args = parser.parse_args()

recording_path = os.path.join(containing_folder, args.recording_path)
if args.model_path is not None:
    model_file_path = os.path.join(containing_folder, args.model_path)
learning_scheme = args.learning_scheme
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
if not args.no_print:
    print("PARAMETERS:")
    print("  num_robots ----", Utility.params['num_robots'])
    print("  num_obstacles -", Utility.params['num_obstacles'])
    print("  num_obs -------", Utility.params['num_obs'])
    print("  alphabet_size -", Utility.params['alphabet_size'])
    print("  num_actions ---", Utility.params['num_actions'])
    print("  num_stats -----", Utility.params['num_stats'])

Utility.set_obstacles_fields()
# Path to save data
data_file_path = recording_path + '/Data/'

agent_args = {'n_agents':Utility.params['num_robots'],
              'n_obs':Utility.params['num_obs'], 
              'n_actions': Utility.params['num_actions']-1,
              'learning_scheme': args.learning_scheme,
              'options_per_action':3,
              'n_chars':Utility.params['alphabet_size'],
              'meta_param_size':1, 
              'use_intention':args.intention, 
              'use_recurrent':args.recurrent,
              'attention':args.attention, 
              'intention_look_back':2}


if args.independent_learning:
    models = [Agent.Agent(id=i, **agent_args) for i in range(Utility.params['num_robots'])]
    if test_mode:
        [models[i].load_model(model_file_path) for i in range(Utility.params['num_robots'])]
else:
    if args.trained_num_robots is not None:
        agent_args['n_agents'] = int(args.trained_num_robots)
        model = Agent.Agent(id = 0, **agent_args)
    else:
        model = Agent.Agent(id = 0, **agent_args)
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
experiment_start_time = time.time()
Testing_Failures = 0
Testing_Successes = 0
var_grad = 0
gate = 0
gate_stats = 0
obstacles = 0
obstacle_stats = 0



while not exp_done:
    #receive initial observations
    msgs = socket.recv_multipart()
    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
    data_file_name = 'Data_Episode_'+str(ep_counter)+'.csv'
    with open(data_file_path+data_file_name, 'w') as output:
        writer = csv.writer(output, delimiter = ',')
        writer.writerow(['reward', 'epsilon', 'termination', 'loss', 'force magnitude', 'force angle', 'average force vector', 'cyl_x_pos', 'cyl_y_pos', 'gate_stats', 'obstacle_stats', 'intention_reward', 'run_time', 'robots_x_pos', 'robots_y_pos', 'robot_angle', 'env_observations', 'agent_actions'])

        if not exp_done:
            time_steps = 0
            
            object_positions = []
            agent_prox_flags = []
            last_object_heading = None

            next_heading_intention = np.zeros(Utility.params['num_robots'])

            episode_intention_rewards = np.zeros(Utility.params['num_robots'])

            # Receive initial observations from the environment
            env_observations = Utility.parse_obs(msgs[1])
            failures = Utility.parse_failures(msgs[2])
            rewards = Utility.parse_rewards(msgs[3])
            stats = Utility.parse_stats(msgs[4])
            robot_stats = Utility.parse_robot_stats(msgs[5])
            obj_stats = Utility.parse_obj_stats(msgs[6])

            object_positions.append([obj_stats[0], obj_stats[1]])

            if Utility.params['num_obstacles'] > 0:
                obstacle_stats = Utility.parse_obstacle_stats(msgs[7])
            elif Utility.params['use_gate'] == 1:
                gate_stats = Utility.parse_gate_stats(msgs[7])

            # Store the object stats in agent for learning later
            if args.independent_learning:
                [models[i].store_object_stats(obj_stats, time_steps>2) for i in range(Utility.params['num_robots'])]
                [models[i].reset_intention_sequence()]
            else:
                model.store_object_stats(obj_stats, time_steps>2)
                model.reset_intention_sequence()

            agent_states = []
            force_mags = []
            force_angs = []
            if args.independent_learning:
                running_reward = []
            else:
                running_reward = 0

            for i in range(Utility.params['num_robots']):
                # append env observations and messages in inbox to make agent state
                if args.independent_learning:
                    running_reward.append(0)
                    if args.intention:
                        agent_state = models[i].make_agent_state(env_observations[i], next_heading_intention[i])
                    else:
                        agent_state = env_observations[i]
                        
                else:
                    if args.intention:
                        agent_state = model.make_agent_state(env_observations[i], next_heading_intention[i])
                    else: 
                        agent_state = env_observations[i]

                agent_states.append(agent_state)
                force_mags.append(stats[i][0])
                force_angs.append(stats[i][1])

            # reward is the same across all agents. If it were per agent then this would need to move into the loop above
            if args.independent_learning:
                for i in range(Utility.params['num_robots']):
                    running_reward[i]+= rewards[i]
            else:
                running_reward += rewards[0]
            # failures should all be false because we havent started the episode yet
            failure = failures[0]

            #
            # Start the Episode Loop
            #

            episode_start_time = time.time()
            while not episode_done:
                if not exp_done:
                    reward = []
                    actions = []
                    actions_to_take = []
                    time_steps += 1

                    for i in range(Utility.params['num_robots']):
                        # Choose an action
                        if args.independent_learning:
                            action, action_num = models[i].choose_agent_action(agent_states[i], failures[i], test_mode)
                        else:
                            action, action_num = model.choose_agent_action(agent_states[i], failures[i], test_mode)
                        actions_to_take.append(action)
                        actions.append(action_num)

                    old_failures = failures[:]
                    # Take Step
                    socket.send(Utility.serialize_actions(actions_to_take))
                    msgs = socket.recv_multipart()

                    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
                    env_observations = Utility.parse_obs(msgs[1])
                    failures = Utility.parse_failures(msgs[2])
                    #print('[DEBUG] Received Failures ', failures)
                    rewards = Utility.parse_rewards(msgs[3])
                    stats = Utility.parse_stats(msgs[4])
                    robot_stats = Utility.parse_robot_stats(msgs[5])
                    obj_stats = Utility.parse_obj_stats(msgs[6])
                    
                    robot_x_pos = []
                    robot_y_pos = []
                    robot_angle = []
                    for i in range(Utility.params['num_robots']):
                        robot_x_pos.append(robot_stats[i][0])
                        robot_y_pos.append(robot_stats[i][1])
                        robot_angle.append(robot_stats[i][5])
                    if Utility.params['num_obstacles'] > 0:
                        obstacle_stats = Utility.parse_obstacle_stats(msgs[7])
                    elif Utility.params['use_gate'] == 1:
                        gate_stats = Utility.parse_gate_stats(msgs[7])

                    old_object_positions = copy.deepcopy(object_positions)
                    object_positions.append([obj_stats[0], obj_stats[1]])

                    intention_reward = []
                    label = 0
                    ############################## INTENTION REWARD ##############################################
                    if args.intention:
                        if len(object_positions) > 2:
                            object_positions.pop(0)

                            last_object_heading = math.atan2((object_positions[0][1] - object_positions[1][1]), (object_positions[0][0] - object_positions[1][0]))
                            x1 = math.cos(last_object_heading)
                            y1 = math.sin(last_object_heading)
                            label = last_object_heading

                            for i in range(Utility.params['num_robots']):
                                x2 = math.cos(next_heading_intention[i] * math.pi)
                                y2 = math.sin(next_heading_intention[i] * math.pi)

                                diff = np.dot([x1, y1], [x2, y2])
                                intention_reward.append(-1 + diff)
                                episode_intention_rewards[i] += intention_reward[i]

                        else:
                            intention_reward = [0 for i in range(Utility.params['num_robots'])]

                    else:
                        intention_reward = [0 for i in range(Utility.params['num_robots'])]


                    # store object stats for learning later
                    if args.independent_learning:
                        for i in range(Utility.params['num_robots']):
                            models[i].store_object_stats(obj_stats, time_steps>2)
                    else:
                        model.store_object_stats(obj_stats, time_steps>2)

                    # Store Transitions and Learn
                    old_agent_prox_flags = copy.deepcopy(agent_prox_flags)

                    new_agent_states = []
                    force_mags = []
                    force_angs = []
                    r = []
                    agent_prox_flags = []
                    next_object_heading = np.zeros(Utility.params['num_robots'])

                    if args.intention:
                        for i in range(Utility.params['num_robots']):
                            prox_values = env_observations[i][7:]
                            prox_value = np.sum(prox_values)              
                            agent_prox_flags.append(prox_value/24.0)
                            #if prox_value/24 > 0.5:
                            #    agent_prox_flags.append(1)
                            #else:
                            #    agent_prox_flags.append(0)
                        if args.attention:
                            if args.independent_learning:
                                for i in range(Utility.params['num_robots']):
                                    import ipdb; ipdb.set_trace()
                                    # Check the syntax on object positions to make sure we are giving an X and Y
                                    next_object_heading[i] = models[i].choose_object_intention(object_positions[-1], agent_prox_flags, test_mode)
                                    next_heading_intention[i] = next_object_heading[i]
                            else:
                                ctde_intention = model.choose_object_intention(object_positions[-1], agent_prox_flags, test_mode)
                                for i in range(Utility.params['num_robots']):
                                    next_heading_intention[i] = ctde_intention

                        elif len(object_positions) == 2:
                            if args.independent_learning:
                                for i in range(Utility.params['num_robots']):
                                    next_object_heading[i] = models[i].choose_object_intention(object_positions, agent_prox_flags, test_mode)
                                    next_heading_intention[i] = next_object_heading[i]
                            else:
                                ctde_intention = model.choose_object_intention(object_positions, agent_prox_flags, test_mode)
                                for i in range(Utility.params['num_robots']):
                                    next_heading_intention[i] = ctde_intention
                        #store transitions of intentions
                        if args.attention:
                            if args.independent_learning:
                                    models[i].store_intention_transition(np.append(np.array(old_object_positions[-1]).flatten(), agent_prox_flags), label, 0, 0, 0)
                            else:
                                model.store_intention_transition(np.append(np.array(old_object_positions[-1]).flatten(), agent_prox_flags), label, 0, 0, 0)

                        elif len(old_object_positions) == 2:
                            for i in range(Utility.params['num_robots']):
                                if args.independent_learning:
                                    models[i].store_intention_transition(np.append(np.array(old_object_positions).flatten(), agent_prox_flags), next_heading_intention[i], intention_reward[i], np.append(object_positions, old_agent_prox_flags),0)
                                else:
                                    state = np.append(np.array(old_object_positions).flatten(), agent_prox_flags)
                                    action = next_heading_intention[i]
                                    reward = intention_reward[i]
                                    new_state = np.append(object_positions, old_agent_prox_flags)
                                    model.store_intention_transition(state, action, reward, new_state, 0)


                    for i in range(Utility.params['num_robots']):
                        #print('[DEBUG] ROBOT', i)
                        #reward = rewards[i]
                        #print('[DEBUG] OBS:', env_observations[i])
                        prox_values = env_observations[i][7:]
                        #print('[DEBUG] Prox Values', prox_values)
                        prox_value = np.sum(prox_values)
                        #print('[DEBUG] Prox Value Reward:', (-0.1)*prox_value)
                        #reward += (-1)*prox_value
                        rewards[i] += (-1)*prox_value
                        force_mags.append(stats[i][0])
                        force_angs.append(stats[i][1])
                        if args.independent_learning:
                            if args.intention:
                                new_agent_state = models[i].make_agent_state(env_observations[i], next_heading_intention[i])
                            else:
                                new_agent_state = env_observations[i]
                        else:
                            if args.intention:
                                new_agent_state= model.make_agent_state(env_observations[i], next_heading_intention[i])
                            else:
                                new_agent_state = env_observations[i]

                        new_agent_states.append(new_agent_state)
                        if time_steps > 2:
                            if train_mode:
                                if learning_scheme != 'None':
                                    if not old_failures[i] and not failures[i]:
                                        if not episode_done:
                                            if args.independent_learning:
                                                models[i].store_agent_transition(agent_states[i],
                                                                    (actions[i], actions_to_take[i]),
                                                                    rewards[i],
                                                                    new_agent_states[i],
                                                                    episode_done)
                                            else:
                                                model.store_agent_transition(agent_states[i],
                                                                    (actions[i], actions_to_take[i]),
                                                                    rewards[i],
                                                                    new_agent_states[i],
                                                                    episode_done)
                                                    
                        r.append(rewards[i][0])
                    #print('[DEBUG] Rewards:', r)
                    #print('[DEBUG] Reward Average:', np.average(r))
                    #for i in range(Utility.params['num_robots']):
                    #    print('[DEBUG] robot %i memory:'%i, message_memory[i])

                    if train_mode:
                        if args.independent_learning:
                            for i in range(Utility.params['num_robots']):
                                loss = models[i].learn()
                        else:
                            loss = model.learn()
                    else:
                        loss = 0

                    if args.independent_learning:
                        for i in range(Utility.params['num_robots']):
                            running_reward[i] += r[i]
                    else:
                        running_reward += np.average(r)
                    # Store New Observations
                    agent_states = new_agent_states
                    actions = []

                    # Calculate average force vector
                    average_force_mag = None
                    average_force_ang = None
                    for i in range(Utility.params['num_robots']):
                        if average_force_mag is None:
                            average_force_mag = force_mags[i]
                            average_force_ang = force_angs[i]
                        else:
                            angle = abs(average_force_ang - force_angs[i])
                            #average_force_mag = math.sqrt(average_force_mag**2 + force_mags[i]**2 + 2*(average_force_mag)*(force_mags[i])*math.cos(math.radians(angle)))
                            #average_force_ang = math.asin(force_mags[i]*math.sin(math.radians(180 - angle)) / average_force_mag)
                            average_force_mag = 0
                            average_force_ang = 0

                    if type(gate_stats) != int:
                        gate = []
                        for i in range(len(gate_stats)):
                            gate.append(gate_stats[i])
                    if type(obstacle_stats) != int:
                        obstacles = []
                        for i in range(len(obstacle_stats)):
                            obstacles.append(obstacle_stats[i])
                    if args.independent_learning:
                        tmp_epsilon = models[0].epsilon
                    else:
                        tmp_epsilon = model.epsilon


                    writer.writerow([r, tmp_epsilon, reached_goal, loss, force_mags, force_angs, 
                                    [average_force_mag, math.degrees(average_force_ang)], obj_stats[0], obj_stats[1],
                                    gate, obstacles, intention_reward, time.time() - episode_start_time, robot_x_pos, 
                                    robot_y_pos, robot_angle, env_observations, actions_to_take])

                    if episode_done:
                        run_time = time.time() - episode_start_time
                        if not args.no_print:
                            print('[RUN TIME] %.2f' % run_time)
                        if args.independent_learning:
                            exp_rewards.append(np.average(running_reward))
                        else:
                            exp_rewards.append(running_reward)
                        if not reached_goal:
                            if not args.no_print:
                                print("Episode", ep_counter ,"timed out")
                            if test_mode:
                                Testing_Failures += 1
                        else:
                            if not args.no_print:
                                print("Episode", ep_counter ,"reached goal")
                            if test_mode:
                                Testing_Successes += 1
                        if not args.no_print:
                            for i in range(Utility.params['num_robots']):
                                if args.independent_learning:
                                    print('Agent', i, 'reward %.1f' % running_reward[i],
                                          'epsilon:%.2f' % models[i].epsilon,
                                          'steps:', models[i].learn_step_counter)
                                else:
                                    print('Agent', i, 'reward %.1f' % running_reward,
                                          'epsilon:%.2f' % model.epsilon,
                                          'steps:', model.networks['learn_step_counter'])
                                    print('Intention rewards %.2f' % episode_intention_rewards[i])

                        if ep_counter % 10 == 0:
                            exp_mean_rewards.append(np.mean(exp_rewards))
                            exp_rewards = []
                            file_name = 'Episode_'+str(ep_counter)
                            path = recording_path + "/Models/" +file_name
                            if train_mode:
                                if args.independent_learning:
                                    for i in range(Utility.params['num_robots']):
                                        models[i].save_model(path)
                                else:
                                    model.save_model(path)
                            if not args.no_print:
                                print('reward last 10 eps:%.2f'%exp_mean_rewards[-1],'\n')
                        ep_counter += 1
                        if args.independent_learning:
                            for i in range(Utility.params['num_robots']):
                                models[i].reset_obj_stats()
                        else:
                            model.reset_obj_stats()
                        #if not args.no_print:
                        #    print("[INFO] Max Object Statistics: ", model.max_obj_stats)
                        #    print("[INFO] Min Object Statistics: ", model.min_obj_stats)

                        # Send acknowledgment
                        socket.send(b"ok")
print("[RUN TIME] Experiment: %.2f" % (time.time() - experiment_start_time))
if test_mode:
    print('Experiment:', args.recording_path)
    print("[Statistics] Success Percentage", (Testing_Successes/(Testing_Successes+Testing_Failures)))
    print("[Statistics] Failure Percentage", (Testing_Failures/(Testing_Successes+Testing_Failures)))
print("Closing Server")
#socket.unbind("tcp://:" + port)
#socket.close()
print("Experiment Done\n")
