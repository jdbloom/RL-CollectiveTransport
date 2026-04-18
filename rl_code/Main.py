from urllib.parse import uses_relative
#import python_code.Agent as Agent
import src.agent as Agent
from src.env import calculate_gsp_reward, ZMQ_Utility
from src.hdf5_logger import HDF5Logger
from src.zmq_diagnostics import DiagnosticSocket
from src.diagnostics import ExperimentLogger

#from python_code.comms_viz import viz

import argparse
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import math

import zmq
import csv
import os
import time
import torch as T
import matplotlib.pyplot as plt
import yaml
import logging
import traceback

Utility = ZMQ_Utility()

# get path to containing folder so this works where ever it is used
containing_folder = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--test", default = False, action = "store_true")
parser.add_argument("--model_path")
parser.add_argument("--best_gsp_ckpt", default=None,
                    help="Path to a saved GSP-head snapshot (from Task 1). "
                         "If set in test mode, the GSP head is loaded from this "
                         "checkpoint AFTER load_model, overriding the bundled final weights.")
parser.add_argument("--trained_num_robots")                                          # if we are testing a model trained on a different number of robots. This should be set to the training number of robots so that the network is built properly.
parser.add_argument("--no_print", default = False, action = "store_true")
parser.add_argument("--independent_learning", default = False, action = "store_true")
parser.add_argument("--global_knowledge", default = False, action = "store_true")   # append knowledge of other agents to the observation space
parser.add_argument("--share_prox_values", default=False, action = 'store_true')    # Robots will share their averaged prox values with eachother

args = parser.parse_args()

recording_path = os.path.join(containing_folder, args.recording_path)
exp_logger = ExperimentLogger(os.path.basename(recording_path))
log = exp_logger.get_logger("main")
log.info("Starting experiment: %s", recording_path)
config_path = os.path.join(recording_path, 'agent_config.yml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

if args.model_path is not None:
    model_file_path = os.path.join(containing_folder, args.model_path)
learning_scheme = config['LEARNING_SCHEME']
learn_every = int(config.get('LEARN_EVERY', 1))
port = str(config['PORT'])
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
socket = DiagnosticSocket(
    socket, os.path.basename(recording_path),
    logger=exp_logger.get_logger("zmq"),
)
socket.setsockopt(zmq.RCVTIMEO, 600000)  # 10 minute recv timeout — detect ARGoS crashes
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

# Initialize HDF5 logger (one per experiment)
hdf5_path = os.path.join(recording_path, os.path.basename(recording_path) + ".h5")
hdf5_writer = HDF5Logger(hdf5_path)

# Per-episode diagnostics (FAU / weight norms / effective rank / Q-gap / pred
# diversity). Opt-in via config['DIAGNOSTICS_ENABLED']. The rolling gsp_obs_pool
# buffers recent head-input vectors so the eval batch for GSP diagnostics can
# be drawn from the same distribution the head actually sees during training.
# Cap at max size so memory stays bounded on long runs.
_DIAG_POOL_MAX_SIZE = 8192
diag_gsp_obs_pool: list = []
diag_eval_batch_frozen: bool = False
diag_episode_predictions: list = []  # per-step GSP predictions this episode, reset each ep

if args.share_prox_values:
    num_obs = Utility.params['num_obs'] +Utility.params['num_robots']   #need to account for num_robots extra observations
elif args.global_knowledge:
    num_obs = Utility.params['num_obs']+(Utility.params['num_robots']-1)*4  #need to account for the x and y positions and the x and y velocitis for each robot
else:
    num_obs = Utility.params['num_obs']

agent_nn_args = {
    'config': config,
    'network': config['LEARNING_SCHEME'],
    'n_agents': Utility.params['num_robots'],
    'n_obs': num_obs, # + 6,  # to account for the sin, cos, and tan of the two angles
    'n_actions': Utility.params['num_actions']-1,  #remove control of the gripper
    'options_per_action':config['OPTIONS_PER_ACTION'],
    'min_max_action':config['MIN_MAX_ACTION'],
    'meta_param_size':config['META_PARAM_SIZE'],
    'gsp': config['GSP'],
    'recurrent': config['RECURRENT'],
    'attention': config['ATTENTION'],
    'neighbors': config['NEIGHBORS'],
    'broadcast': config.get('BROADCAST', False),
    'gsp_input_size':config['GSP_INPUT_SIZE'],
    'gsp_output_size':config['GSP_OUTPUT_SIZE'],
    'gsp_look_back':config['GSP_LOOK_BACK'],
    'gsp_min_max_action':config['GSP_MIN_MAX_ACTION'],
    'gsp_sequence_length':config['GSP_SEQUENCE_LENGTH'],
    'prox_filter_angle_deg':config['PROX_FILTER_ANGLE_DEG'],
}


if args.independent_learning:
    models = [Agent.Agent(id=i, **agent_nn_args) for i in range(Utility.params['num_robots'])]
    if test_mode:
        [models[i].load_model(model_file_path) for i in range(Utility.params['num_robots'])]
        if args.best_gsp_ckpt:
            log.info(f'Loading best GSP-head checkpoint from {args.best_gsp_ckpt}')
            for m in models:
                m.load_gsp_head_snapshot(args.best_gsp_ckpt)
else:
    if args.trained_num_robots is not None:
        agent_nn_args['n_agents'] = int(args.trained_num_robots)
        model = Agent.Agent(id = 0, **agent_nn_args)
    else:
        model = Agent.Agent(id = 0, **agent_nn_args)
    if test_mode:
        model.load_model(model_file_path)
        if args.best_gsp_ckpt:
            log.info(f'Loading best GSP-head checkpoint from {args.best_gsp_ckpt}')
            model.load_gsp_head_snapshot(args.best_gsp_ckpt)


# Send acknowledgment
socket.send(b"ok")

# Prism handshake (non-uniform objects)
if Utility.params['num_prisms'] > 0:
    Utility.set_prism_sizes()
    prism_sizes = Utility.parse_prism_sizes(socket.recv())
    socket.send(b"ok")
    Utility.set_prism_points(prism_sizes)
    prism_points = Utility.parse_prism_points(socket.recv())
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
ep_ticks = 0

try:
    while not exp_done:
        #receive initial observations
        msgs = socket.recv_multipart()
        exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
        socket.set_episode(ep_counter)
        log.info("Episode %d starting", ep_counter)

        if not exp_done:
            time_steps = 0
        
            agent_prox_flags = []
            last_object_heading = None

            next_heading_gsp = np.zeros(Utility.params['num_robots'])
            old_heading_gsp = np.zeros(Utility.params['num_robots'])
            episode_gsp_rewards = np.zeros(Utility.params['num_robots'])

            # Receive initial observations from the environment
            env_observations, failures, rewards, stats, robot_stats, obj_stats = Utility.parse_msgs(msgs)
            old_cyl_ang = obj_stats[5]

            if Utility.params['num_obstacles'] > 0:
                obstacle_stats = Utility.parse_obstacle_stats(msgs[7])
            elif Utility.params['use_gate'] == 1:
                gate_stats = Utility.parse_gate_stats(msgs[7])

            agent_states = []
            force_mags = []
            force_angs = []
            if args.independent_learning:
                running_reward = []
            else:
                running_reward = 0
        
            for i in range(Utility.params['num_robots']):
                if failures[i][0]:
                    agent_prox_flags.append(0)
                else:
                    prox_values = env_observations[i][7:]
                    # Add logic to filter prox values that are observing the object
                    prox_values, filtered_indeces = model.filter_prox_values(prox_values, env_observations[i][5])
                    for j in range(len(filtered_indeces)):
                        env_observations[i][7+filtered_indeces[j]] = 0.0
                    prox_value = np.sum(prox_values)
                    agent_prox_flags.append(prox_value/float(len(filtered_indeces)))
        
            #Define Global Knowledge: [positions, velocities]
            global_knowledge=np.zeros((Utility.params['num_robots'])*4)
            for i in range(Utility.params['num_robots']):
                global_knowledge[i*4] = robot_stats[i][0]           #x position
                global_knowledge[i*4+1] = robot_stats[i][1]         #y position
                global_knowledge[i*4+2] = stats[i][2]               #velocity X
                global_knowledge[i*4+3] = stats[i][3]               #velocity Y

            for i in range(Utility.params['num_robots']):
                g_knowledge = np.zeros((Utility.params['num_robots']-1)*4)
                counter = 0
                for j in range(Utility.params['num_robots']):
                    if i != j:
                        g_knowledge[counter*4] = global_knowledge[j*4]
                        g_knowledge[counter*4+1] = global_knowledge[j*4+1]
                        g_knowledge[counter*4+2] = global_knowledge[j*4+2]
                        g_knowledge[counter*4+3] = global_knowledge[j*4+3]
                        counter+=1
                if args.independent_learning:
                    running_reward.append(0)
                    if config['GSP']:
                        if args.global_knowledge:
                            agent_state = models[i].make_agent_state(env_observations[i], heading_gsp = next_heading_gsp[i], global_knowledge=g_knowledge) 
                        else:
                            agent_state = models[i].make_agent_state(env_observations[i], heading_gsp = next_heading_gsp[i])
                    else:
                        if args.global_knowledge:
                            agent_state = models[i].make_agent_state(env_observations[i], global_knowledge = g_knowledge)
                        else:
                            agent_state = env_observations[i]
                    
                else:
                    if config['GSP']:
                        if args.global_knowledge:
                            agent_state = model.make_agent_state(env_observations[i], heading_gsp=next_heading_gsp[i], global_knowledge=g_knowledge)
                        else:
                            agent_state = model.make_agent_state(env_observations[i], heading_gsp=next_heading_gsp[i])
                    else: 
                        if args.share_prox_values:
                            agent_state = np.concatenate((env_observations[i], agent_prox_flags))
                        else:
                            if args.global_knowledge:
                                agent_state = model.make_agent_state(env_observations[i], global_knowledge=g_knowledge)
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
                    robot_failures = []

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
                    env_observations, failures, rewards, stats, robot_stats, obj_stats = Utility.parse_msgs(msgs)
                    com_X_poses = obj_stats[7]
                    com_Y_poses = obj_stats[8]
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

                    ############################## gsp REWARD ##############################################
                    gsp_reward, label, gsp_squared_error = calculate_gsp_reward(
                        config['GSP'],
                        old_cyl_ang,
                        obj_stats[5],
                        next_heading_gsp,
                        Utility.params['num_robots']
                    )
                    # print('[MAIN] GSP Reward', gsp_reward)
                    # print('[MAIN] GSP Label ', label)
                    e2e_gsp_label = None
                    if config.get('GSP_E2E_ENABLED'):
                        e2e_gsp_label = np.array([label], dtype=np.float32)
                    for i in range(len(gsp_reward)):
                        episode_gsp_rewards[i] += gsp_reward[i]

                    old_cyl_ang = obj_stats[5]

                    old_agent_prox_flags = list(agent_prox_flags)
                    neighbors_old_heading_gsp = old_heading_gsp.copy()
                    old_heading_gsp = next_heading_gsp.copy()

                    new_agent_states = []
                    force_mags = []
                    force_angs = []
                    r = []
                    agent_prox_flags = []
                    next_object_heading = np.zeros(Utility.params['num_robots'])
                
                    # Build proximity observation
                    for i in range(Utility.params['num_robots']):
                        robot_failures.append(failures[i][0])
                        if failures[i][0]:
                            agent_prox_flags.append(0)
                        else:
                            prox_values = env_observations[i][7:]
                            prox_values, filtered_indeces = model.filter_prox_values(prox_values, env_observations[i][5])
                            for j in range(len(filtered_indeces)):
                                env_observations[i][7+filtered_indeces[j]] = 0.0
                            prox_value = np.sum(prox_values)              
                            agent_prox_flags.append(prox_value/float(len(filtered_indeces)))

                    e2e_gsp_obs = [None] * Utility.params['num_robots']
                    # H-14 / first-principles diagnostic: capture the GSP head's per-robot
                    # input vector at this timestep so it can be logged alongside gsp_target.
                    # Set in the GSP branch below; remains None for non-GSP runs.
                    gsp_obs_per_robot = None

                    if config['GSP']:
                        # GSP Predict
                        if args.independent_learning:
                            for i in range(Utility.params['num_robots']):
                                next_object_heading[i] = models[i].choose_agent_gsp(agent_prox_flags, test_mode)
                                next_heading_gsp[i] = next_object_heading[i]
                        else:
                            if model.gsp_neighbors:
                                agent_gsp_states = model.make_gsp_states(agent_prox_flags, old_heading_gsp)
                                ctde_gsp = model.choose_agent_gsp(agent_gsp_states, test_mode)
                                gsp_obs_per_robot = agent_gsp_states
                            elif model.gsp_broadcast:
                                # GSP-B: per-agent self-centric view with full-broadcast
                                # [self_prox, self_prev_gsp, other_i_prox, other_i_prev_gsp, ...]
                                agent_gsp_states = model.make_gsp_states_broadcast(agent_prox_flags, old_heading_gsp)
                                ctde_gsp = model.choose_agent_gsp(agent_gsp_states, test_mode)
                                gsp_obs_per_robot = agent_gsp_states
                            else:
                                ctde_gsp = model.choose_agent_gsp(agent_prox_flags, test_mode)
                            for i in range(Utility.params['num_robots']):
                                if len(ctde_gsp) > 1:
                                    next_heading_gsp[i] = ctde_gsp[i][-1].item()
                                else:
                                    next_heading_gsp[i] = ctde_gsp[-1].item()
                        # print("-------------------------------------------------")
                        # print('[GSP]', next_heading_gsp)

                        # Store GSP Transition — guard by per-robot force magnitude.
                        # GSP_STORE_FORCE_THRESHOLD concentrates training on samples where
                        # the robot is actively applying force (top ~25% of samples at
                        # threshold ~4.0), which multiplies the linear-R² ceiling of the
                        # prediction problem 3–4× (see
                        # docs/research/2026-04-13-gsp-ddpg-vs-attention-collapse.md).
                        # 0.0 = filter disabled (legacy behavior).
                        force_thr = float(config.get('GSP_STORE_FORCE_THRESHOLD', 0.0))
                        if model.gsp_neighbors:
                            states, state_prox_flags = model.make_gsp_states(old_agent_prox_flags, neighbors_old_heading_gsp, True)
                            new_states = model.make_gsp_states(agent_prox_flags, old_heading_gsp)
                            if config.get('GSP_E2E_ENABLED'):
                                for i in range(Utility.params['num_robots']):
                                    e2e_gsp_obs[i] = np.array(states[i], dtype=np.float32)

                            # Candidate A: future-prox target — store transitions with per-robot
                            # prox K steps ahead as the label, instead of the shared Δθ scalar.
                            # Buffer accumulates (state_t) snapshots; only when matured at t+K
                            # do we have (state_{t-K}, prox_t) pairs to store.
                            if getattr(model, 'gsp_prediction_target', 'delta_theta') == 'future_prox':
                                model.push_pending_gsp_obs(states, states)
                                matured = model.pop_matured_gsp_label(
                                    np.asarray(agent_prox_flags, dtype=np.float32)
                                )
                                if matured is not None:
                                    for i in range(Utility.params['num_robots']):
                                        s_to_store = matured['state_per_robot'][i]
                                        label_to_store = float(matured['label_per_robot'][i])
                                        model.store_gsp_transition(s_to_store, label_to_store, 0, s_to_store, 0)
                            else:
                                for i in range(Utility.params['num_robots']):
                                    if np.sum(state_prox_flags[i]) > 0 and stats[i][0] > force_thr:
                                        if model.gsp_networks['learning_scheme'] == 'attention':
                                            model.store_gsp_transition(states[i], label, 0, 0, 0)
                                        else:
                                            state = states[i]
                                            new_state = new_states[i]
                                            model.store_gsp_transition(state, label, 0, new_state, 0)
                        elif model.gsp_broadcast:
                            states = model.make_gsp_states_broadcast(old_agent_prox_flags, neighbors_old_heading_gsp)
                            new_states = model.make_gsp_states_broadcast(agent_prox_flags, old_heading_gsp)
                            for i in range(Utility.params['num_robots']):
                                if states[i][0] != 0 and stats[i][0] > force_thr:
                                    model.store_gsp_transition(states[i], label, 0, new_states[i], 0)
                        else:
                            for i in range(Utility.params['num_robots']):
                                state = np.array(old_agent_prox_flags)
                                if np.sum(state) > 0 and stats[i][0] > force_thr:
                                    if model.gsp_networks['learning_scheme'] == 'attention':
                                        model.store_gsp_transition(state, label, 0, 0, 0)
                                    elif args.independent_learning:
                                        new_state = np.array(agent_prox_flags)
                                        models[i].store_gsp_transition(state, label, 0, new_state, 0)
                                    else:
                                        new_state = np.array(agent_prox_flags)
                                        model.store_gsp_transition(state, label, 0, new_state, 0)


                    #Define Global Knowledge: [positions, velocities]
                    global_knowledge=np.zeros((Utility.params['num_robots'])*4)
                    for i in range(Utility.params['num_robots']):
                        global_knowledge[i*4] = robot_stats[i][0]           #x position
                        global_knowledge[i*4+1] = robot_stats[i][1]         #y position
                        global_knowledge[i*4+2] = stats[i][2]               #velocity X
                        global_knowledge[i*4+3] = stats[i][3]               #velocity Y


                    for i in range(Utility.params['num_robots']):
                        g_knowledge = np.zeros((Utility.params['num_robots']-1)*4)
                        counter = 0
                        for j in range(Utility.params['num_robots']):
                            if i != j:
                                g_knowledge[counter*4] = global_knowledge[j*4]
                                g_knowledge[counter*4+1] = global_knowledge[j*4+1]
                                g_knowledge[counter*4+2] = global_knowledge[j*4+2]
                                g_knowledge[counter*4+3] = global_knowledge[j*4+3]
                                counter+=1
                        prox_values = env_observations[i][7:]
                        prox_value = np.sum(prox_values)
                        rewards[i] += (-1)*prox_value
                        force_mags.append(stats[i][0])
                        force_angs.append(stats[i][1])

                        if args.independent_learning:
                            if config['GSP']:
                                if args.global_knowledge:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], heading_gsp = next_heading_gsp[i], global_knowledge=g_knowledge) 
                                else:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], heading_gsp = next_heading_gsp[i])
                            else:
                                if args.global_knowledge:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], global_knowledge = g_knowledge)
                                else:
                                    new_agent_state = env_observations[i]
                            
                        else:
                            if config['GSP']:
                                if args.global_knowledge:
                                    new_agent_state = model.make_agent_state(env_observations[i], heading_gsp=next_heading_gsp[i], global_knowledge=g_knowledge)
                                else:
                                    new_agent_state = model.make_agent_state(env_observations[i], heading_gsp=next_heading_gsp[i])
                            else: 
                                if args.share_prox_values:
                                    new_agent_state = np.concatenate((env_observations[i], agent_prox_flags))
                                else:
                                    if args.global_knowledge:
                                        new_agent_state = model.make_agent_state(env_observations[i], global_knowledge=g_knowledge)
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
                                                                    episode_done,
                                                                    gsp_obs=e2e_gsp_obs[i] if config.get('GSP_E2E_ENABLED') else None,
                                                                    gsp_label=e2e_gsp_label if config.get('GSP_E2E_ENABLED') else None)
                                            else:
                                                model.store_agent_transition(agent_states[i],
                                                                    (actions[i], actions_to_take[i]),
                                                                    rewards[i],
                                                                    new_agent_states[i],
                                                                    episode_done,
                                                                    gsp_obs=e2e_gsp_obs[i] if config.get('GSP_E2E_ENABLED') else None,
                                                                    gsp_label=e2e_gsp_label if config.get('GSP_E2E_ENABLED') else None)
                                                
                        r.append(rewards[i][0])

                    if train_mode and config['LEARNING_SCHEME'] != 'None':
                        if time_steps % learn_every == 0:
                            if args.independent_learning:
                                # Aggregate GSP losses across per-robot models to a single
                                # scalar per learn tick. Otherwise the 1D gsp_loss dataset
                                # would have (num_learn_steps × num_robots) entries in
                                # independent mode vs. num_learn_steps in shared mode,
                                # breaking cross-mode comparability of the
                                # information-collapse diagnostic.
                                for i in range(Utility.params['num_robots']):
                                    loss = models[i].learn()
                                    # TD3's learn_TD3 returns (0, 0) on non-actor-update steps;
                                    # unwrap so the hdf5 logger's 1D loss array stays homogeneous.
                                    if isinstance(loss, tuple):
                                        loss = loss[0]
                                gsp_losses = [
                                    m.last_gsp_loss for m in models
                                    if getattr(m, "last_gsp_loss", None) is not None
                                ]
                                if gsp_losses:
                                    hdf5_writer.record_gsp_loss(float(np.mean(gsp_losses)))
                                if config.get('GSP_E2E_ENABLED'):
                                    e2e_diag = getattr(models[0], 'last_e2e_diagnostics', None)
                                    if e2e_diag is not None:
                                        hdf5_writer.record_e2e_diagnostics(e2e_diag)
                            else:
                                loss = model.learn()
                                if isinstance(loss, tuple):
                                    loss = loss[0]
                                gsp_step_loss = getattr(model, "last_gsp_loss", None)
                                if gsp_step_loss is not None:
                                    hdf5_writer.record_gsp_loss(gsp_step_loss)
                                if config.get('GSP_E2E_ENABLED'):
                                    e2e_diag = getattr(model, 'last_e2e_diagnostics', None)
                                    if e2e_diag is not None:
                                        hdf5_writer.record_e2e_diagnostics(e2e_diag)
                        else:
                            loss = 0
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

                    # gsp_target: broadcast the scalar payload delta-theta label to per-robot list
                    # so it aligns with the (timesteps × robots) HDF5 schema. Needed for the
                    # information-collapse diagnostic (gsp_output_std, gsp_pred_target_corr).
                    gsp_target_per_robot = [float(label)] * Utility.params['num_robots']
                    hdf5_writer.writerow(r, tmp_epsilon, reached_goal, loss, force_mags, force_angs,
                                    [average_force_mag, math.degrees(average_force_ang)], obj_stats[0], obj_stats[1],
                                    obj_stats[5], gate, obstacles, gsp_reward, next_heading_gsp,
                                    time.time() - episode_start_time, robot_x_pos, robot_y_pos, robot_angle,
                                    robot_failures, com_X_poses=com_X_poses, com_Y_poses=com_Y_poses,
                                    gsp_target=gsp_target_per_robot, gsp_squared_error=gsp_squared_error,
                                    gsp_obs=gsp_obs_per_robot)

                    # Populate diagnostics pools from the live training loop.
                    # See docs/specs/2026-04-17-diagnostics-instrumentation.md.
                    if getattr(model, 'diagnostics_enabled', False):
                        # Rolling pool of recent GSP head inputs (one per robot per step).
                        # Capped to keep memory bounded on long runs.
                        if gsp_obs_per_robot is not None:
                            for obs in gsp_obs_per_robot:
                                diag_gsp_obs_pool.append(np.asarray(obs, dtype=np.float32))
                            while len(diag_gsp_obs_pool) > _DIAG_POOL_MAX_SIZE:
                                diag_gsp_obs_pool.pop(0)
                        # Accumulate this-episode GSP predictions for the diversity entropy metric.
                        if next_heading_gsp is not None:
                            diag_episode_predictions.extend(
                                float(v) for v in np.asarray(next_heading_gsp, dtype=np.float32).ravel()
                            )

                    if episode_done:
                        if args.independent_learning:
                            for m in models:
                                if hasattr(m, 'reset_hidden_states'):
                                    m.reset_hidden_states()
                                if hasattr(m, 'reset_gsp_label_buffer'):
                                    m.reset_gsp_label_buffer()
                        else:
                            if hasattr(model, 'reset_hidden_states'):
                                model.reset_hidden_states()
                            if hasattr(model, 'reset_gsp_label_buffer'):
                                model.reset_gsp_label_buffer()
                        run_time = time.time() - episode_start_time

                        # Per-episode diagnostics hook. Runs before write_episode so the
                        # diag_* attrs and the (optional) diag_eval_batch_states dataset
                        # land on the same episode group. Gated on DIAGNOSTICS_ENABLED.
                        if getattr(model, 'diagnostics_enabled', False):
                            freeze_ep = getattr(model, 'diagnostics_freeze_episode', 50)
                            cadence = getattr(model, 'diagnostics_cadence', 10)
                            # Freeze the eval batch once, on/after freeze_ep, when the
                            # replay buffer is big enough and the gsp_obs pool has ≥
                            # batch_size samples.
                            if (
                                not diag_eval_batch_frozen
                                and ep_counter >= freeze_ep
                            ):
                                pool_np = (
                                    np.stack(diag_gsp_obs_pool)
                                    if len(diag_gsp_obs_pool) >= model.diagnostics_batch_size
                                    else None
                                )
                                model.freeze_diagnostic_batch(gsp_obs_pool=pool_np)
                                if getattr(model, 'diag_actor_eval_batch', None) is not None:
                                    diag_eval_batch_frozen = True
                                    hdf5_writer.record_eval_batch_states(
                                        model.diag_actor_eval_batch
                                    )
                            # Compute diagnostics on the cadence schedule once frozen.
                            if (
                                diag_eval_batch_frozen
                                and (ep_counter - freeze_ep) % cadence == 0
                            ):
                                preds = (
                                    np.asarray(diag_episode_predictions, dtype=np.float32)
                                    if diag_episode_predictions
                                    else None
                                )
                                diag_result = model.compute_diagnostics(
                                    gsp_predictions_this_episode=preds
                                )
                                if diag_result:
                                    hdf5_writer.record_episode_diagnostics(diag_result)
                            # Reset per-episode prediction accumulator regardless of cadence.
                            diag_episode_predictions = []

                        # h5py is a hard dep of src.hdf5_logger, so the previous HAS_HDF5
                        # gate was always-true dead code. Removed during the same cleanup
                        # that dropped the data_logger references.
                        hdf5_writer.write_episode(ep_counter)
                        log.info(
                            "Episode %d done: success=%s duration=%.1fs timesteps=%d",
                            ep_counter, reached_goal, run_time, time_steps,
                        )
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
                                    print('Agent', i, 'reward %.1f' % running_reward[0],
                                            'epsilon:%.2f' % model.epsilon,
                                            'steps:', model.networks['learn_step_counter'])
                                    print('gsp rewards %.2f' % episode_gsp_rewards[i])

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

                        # GSP head snapshot checkpointing (Task 1 of the stability plan).
                        # Captures the GSP prediction-network weights every N episodes so
                        # post-hoc best-checkpoint selection can recover the top-correlation
                        # predictor even if it regresses during later training. 0 disables.
                        gsp_ckpt_every = int(config.get('GSP_CHECKPOINT_EVERY', 0))
                        if (train_mode and gsp_ckpt_every > 0
                                and ep_counter > 0 and ep_counter % gsp_ckpt_every == 0
                                and not args.independent_learning):
                            try:
                                ckpt_dir = os.path.join(recording_path, 'Models', 'gsp_snapshots')
                                os.makedirs(ckpt_dir, exist_ok=True)
                                ckpt_path = os.path.join(ckpt_dir, f'gsp_ep{ep_counter:04d}.pt')
                                model.save_gsp_head_snapshot(ckpt_path)
                                idx_path = os.path.join(ckpt_dir, 'index.json')
                                import json as _json_ckpt
                                idx = []
                                if os.path.exists(idx_path):
                                    try:
                                        idx = _json_ckpt.load(open(idx_path))
                                    except (ValueError, OSError):
                                        idx = []
                                idx.append({'episode': ep_counter, 'path': ckpt_path})
                                _json_ckpt.dump(idx, open(idx_path, 'w'))
                            except Exception as _e:
                                log.warning(f'GSP checkpoint save failed at ep {ep_counter}: {_e}')
                        ep_counter += 1

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
    exp_logger.finish(success=True)
except zmq.error.Again:
    error_msg = f"ZMQ timeout at episode {ep_counter} — ARGoS likely crashed"
    log.critical(error_msg)
    exp_logger.write_crash_dump(
        last_state={"episode": ep_counter, "timestep": time_steps if 'time_steps' in dir() else 0,
                    "msg_count": socket._msg_count},
        error_message=error_msg,
    )
    exp_logger.finish(success=False, error_message=error_msg)
    raise
except Exception as e:
    error_msg = f"Unexpected error at episode {ep_counter if 'ep_counter' in dir() else '?'}: {e}"
    log.critical(error_msg)
    log.critical(traceback.format_exc())
    exp_logger.write_crash_dump(
        last_state={"episode": ep_counter if 'ep_counter' in dir() else -1},
        error_message=f"{error_msg}\n{traceback.format_exc()}",
    )
    exp_logger.finish(success=False, error_message=error_msg)
    raise
