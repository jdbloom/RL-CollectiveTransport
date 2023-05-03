import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
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
from torch_geometric.data import Data
from python_code.networks import DQN, DDQN,SharedGATDQNAgent,GAT_QNetwork,SharedGATDQNAgent
import python_code.zmq_utility as zmq_utility
prox_filter_angle = 45
episode_counter =0
gate = 0
gate_stats = 0
obstacles = 0
obstacle_stats = 0
ROBOT_PROXIMITY_ANGLES = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5,
                                       112.5, 127.5, 142.5, 157.5, 172.5, -172.5, 
                                       -157.5, -142.5, -127.5, -112.5, -97.5, 
                                       -82.5, -67.5, -52.5, -37.5, -22.5, -7.5]

def filter_prox_values(prox_values, angle_to_cyl):
    if angle_to_cyl > 0:
        if angle_to_cyl > 180-prox_filter_angle:
            cw_lim = angle_to_cyl + prox_filter_angle - 360
        else:
            cw_lim = angle_to_cyl+prox_filter_angle
        ccw_lim = angle_to_cyl - prox_filter_angle
    elif angle_to_cyl < 0:
        if angle_to_cyl < -180 +prox_filter_angle:
            ccw_lim = angle_to_cyl-prox_filter_angle+360
        else:
            ccw_lim = angle_to_cyl - prox_filter_angle
        cw_lim = angle_to_cyl + prox_filter_angle
    else:
        cw_lim = prox_filter_angle
        ccw_lim = -prox_filter_angle

    index = []
    if angle_to_cyl > 180 - prox_filter_angle:
        for i in range(len(ROBOT_PROXIMITY_ANGLES)):
            if ROBOT_PROXIMITY_ANGLES[i] > ccw_lim:
                index.append(i)
            elif ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                index.append(i)
    elif angle_to_cyl < -180+prox_filter_angle:
        for i in range(len(ROBOT_PROXIMITY_ANGLES)):
            if ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                index.append(i)
            elif ROBOT_PROXIMITY_ANGLES[i] > ccw_lim:
                index.append(i)
    else:
        for i in range(len(ROBOT_PROXIMITY_ANGLES)):
            if ROBOT_PROXIMITY_ANGLES[i] > ccw_lim and ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                index.append(i)
    return index

def build_sensorvalues(env_observations, edge_index):
    agent_prox_flags = []
    for i in range(Utility.params['num_robots']):
        prox_values = env_observations[i][7:]
        filtered_indeces = filter_prox_values(prox_values, env_observations[i][5])
        #print("filtered_indeces",np.shape(filtered_indeces))
        #print("prox_values",np.shape(prox_values))
        for j in range(len(filtered_indeces)):
            env_observations[i][7+filtered_indeces[j]] = 0.0
        agent_prox_flags.append(env_observations[i])
    proximity_values = torch.stack([torch.tensor(prox_vals, dtype=torch.float) for prox_vals in agent_prox_flags], dim=0)
    #print("proxvals",proximity_values)

    data = Data(x=proximity_values,edge_index=edge_index)

    return data


def build_sensorvalues1(env_observations,edge_index):
    agent_prox_flags = []
    #print("Utility.params['num_robots']",Utility.params['num_robots'])
    for i in range(Utility.params['num_robots']):
        prox_values = env_observations[i][7:]
        #print("prox_values",prox_values)
                            # Add logic to filter prox values that are observing the object
        prox_values, filtered_indeces = filter_prox_values(prox_values, env_observations[i][5])
        #print("filtered_indeces",np.shape(filtered_indeces))
        #print("prox_values",np.shape(prox_values))
        for j in range(len(filtered_indeces)):
            env_observations[i][7+filtered_indeces[j]] = 0.0
            prox_value = np.sum(prox_values)
            agent_prox_flags.append(prox_value)
    #print("agent_prox_flags",agent_prox_flags)
    proximity_values = torch.tensor(agent_prox_flags, dtype=torch.float).view(-1, 1)
    #print("proxvals",proximity_values)
    data = Data(x=proximity_values,edge_index=edge_index)

    return data



Utility = zmq_utility.ZMQ_Utility()

data_dir = "python_code/Data/GNN/tmp/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data_file_path = data_dir


# Initialize zmq
# Create context
context = zmq.Context()
# create socket
socket = context.socket(zmq.REP)
# wait for connections on specified port, defaults to 55555
port = "55555"

socket.bind("tcp://*:" + port)
print("Server Started")
# Get Parameters
Utility.get_params(socket.recv())

print("PARAMETERS:")
print("  num_robots ----", Utility.params['num_robots'])
print("  num_obstacles -", Utility.params['num_obstacles'])
print("  num_obs -------", Utility.params['num_obs'])
print("  alphabet_size -", Utility.params['alphabet_size'])
print("  num_actions ---", Utility.params['num_actions'])
print("  num_stats -----", Utility.params['num_stats'])

socket.send(b"ok")

TEST = False

#num_robots = 4
in_channels=31
hidden_channels=32
out_channels =32
num_heads=4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# edge_index for connectivity using a circular topology
#edge_index = torch.tensor([
#    [0, 1, 2, 3, 0, 1, 2, 3],
#    [1, 2, 3, 0, 3, 0, 1, 2]
edge_index = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
], dtype=torch.long)
edge_index = edge_index.to(device)

exp_reward = []
exp_loss = []

exp_done = False
ep_counter = 0
num_actions=9
agent = SharedGATDQNAgent(in_channels, hidden_channels, num_actions, num_heads, Utility.params['num_robots'])

# agent.load_model(model_file_path)

#num_episodes = 1000
robot_positions = []
#for episode in range(num_episodes):
while not exp_done:
    msgs = socket.recv_multipart()
    #print("first msg:",msgs)
    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
    #print("episode no:",ep_counter)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    data_file_name = 'Data/Data_Episode_'+str(ep_counter)+'.csv'
    output_file = os.path.join(data_file_path, data_file_name)
    with open(output_file, 'w') as output:
        writer = csv.writer(output, delimiter=',')
        writer.writerow(['reward', 'epsilon', 'termination', 'loss', 'force magnitude', 'force angle', 'average force vector', 'cyl_x_pos', 'cyl_y_pos', 'cyl_angle', 'gate_stats', 'obstacle_stats', 'intention_reward', 'intention_heading', 'run_time', 'robots_x_pos', 'robots_y_pos', 'robot_angle', 'robot_failures', 'env_observations', 'agent_actions'])
        env_observations = Utility.parse_obs(msgs[1])
        #print("env_observations:",len(env_observations))
        init_data = build_sensorvalues(env_observations,edge_index)
        rewards = Utility.parse_rewards(msgs[3])
        stats = Utility.parse_stats(msgs[4])
        robot_stats = Utility.parse_robot_stats(msgs[5])
        obj_stats = Utility.parse_obj_stats(msgs[6])
        initial_obj_pos = np.array([obj_stats[0],obj_stats[1]])
        prev_cyl_dist_goal = env_observations[1][6]
        #print("prev_cyl_dist_goal",prev_cyl_dist_goal)
        if Utility.params['num_obstacles'] > 0:
            obstacle_stats = Utility.parse_obstacle_stats(msgs[7]) #position of obstacles in the environment
        else:
            obstacle_stats = 0
        if Utility.params['use_gate'] == 1:
            gate_stats = Utility.parse_gate_stats(msgs[7])
        #env.reset
        time_step = 0
        episode_reward = 0
        episode_loss = 0
     #start episode loop
        episode_start_time = time.time()
        while not episode_done:#episode loop
            if not exp_done:
                t_rewards = []
                #print("x (node features) shape:\n", init_data.x.shape)
                #print("edge_index shape before act:", edge_index.shape)
                actions = agent.act(init_data.x, init_data.edge_index, TEST)
 
                #print("actions in act ----", actions.shape)
                action_numbers = np.argmax(actions ,axis=1)
                wheel_speeds = [agent.parse_actions(action_num) for action_num in action_numbers]
                wheel_speeds = np.array(wheel_speeds)
                #print("wheel_speeds after act ----", wheel_speeds.shape)
                socket.send(Utility.serialize_actions(wheel_speeds))
                msgs = socket.recv_multipart()
                exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
                env_observations = Utility.parse_obs(msgs[1])
                rewards = Utility.parse_rewards(msgs[3])
                stats = Utility.parse_stats(msgs[4])
                robot_stats = Utility.parse_robot_stats(msgs[5])
                obj_stats = Utility.parse_obj_stats(msgs[6])
                obj_pos = np.array([obj_stats[0],obj_stats[1]],dtype=np.float32)
                if Utility.params['num_obstacles'] > 0:
                    obstacle_stats = Utility.parse_obstacle_stats(msgs[7])
                elif Utility.params['use_gate'] == 1:
                    gate_stats = Utility.parse_gate_stats(msgs[7])
                cyl_dist_goal = env_observations[1][6]
                for i in range(Utility.params['num_robots']):
                    robot_pos = np.array([robot_stats[i][0], robot_stats[i][1]], dtype=np.float32)
                    robot_positions.append(robot_pos)

                robot_x_pos = []
                robot_y_pos = []
                robot_angle = []
                for i in range(Utility.params['num_robots']):
                    robot_x_pos.append(robot_stats[i][0])
                    robot_y_pos.append(robot_stats[i][1])
                    robot_angle.append(robot_stats[i][5])
                #print("edge_index shape befire step:", edge_index.shape)
                new_data = build_sensorvalues(env_observations,edge_index)
                reward = agent.build_reward(cyl_dist_goal,prev_cyl_dist_goal,robot_positions, obstacle_stats, time_step,obj_pos,0.5)
                for i in range(Utility.params['num_robots']):
                    t_rewards.append(rewards[i].item()+reward)
                episode_reward += np.average(t_rewards)
                loss = agent.step(init_data.x,actions,t_rewards,new_data.x,episode_done,edge_index)
                episode_loss += loss
                init_data = new_data
                prev_cyl_dist_goal = cyl_dist_goal
                time_step += 1

                if type(gate_stats) != int:
                    gate = []
                    for i in range(len(gate_stats)):
                        gate.append(gate_stats[i])
                if type(obstacle_stats) != int:
                    obstacles = []
                    for i in range(len(obstacle_stats)):
                        obstacles.append(obstacle_stats[i])
                
                # These are values that are needed for viz but are not calculated in this script
                force_mags = [0 for i in range(Utility.params['num_robots'])]
                force_angs = [0 for i in range(Utility.params['num_robots'])]
                average_force_mag =0
                average_force_ang = 0
                intention_reward = [0 for i in range(Utility.params['num_robots'])]
                next_heading_intention = [0 for i in range(Utility.params['num_robots'])]
                robot_failures = [0 for i in range(Utility.params['num_robots'])]
                actions_to_take = actions
                writer.writerow([t_rewards, agent.epsilon, reached_goal, loss, force_mags, force_angs, 
                                    [average_force_mag, math.degrees(average_force_ang)], obj_stats[0], obj_stats[1],
                                    obj_stats[5], gate, obstacles, intention_reward, next_heading_intention[0], 
                                    time.time() - episode_start_time, robot_x_pos, robot_y_pos, robot_angle, 
                                    robot_failures, env_observations, actions_to_take])
                if episode_done:
                    print("episode",ep_counter, episode_reward, episode_loss, agent.epsilon)
                    exp_reward.append(episode_reward)
                    exp_loss.append(episode_loss)
                    ep_counter+=1
                    index = np.linspace(0, len(exp_reward), len(exp_reward))
                    plt.clf()
                    plt.scatter(index, exp_reward, c='b')
                    plt.title('GNN Baseline Reward')
                    plt.xlabel('Episodes')
                    plt.ylabel('Rewards')
                    plt.savefig(data_dir+'Exp_Reward_Plot.png')

                    plt.clf()
                    plt.plot(exp_loss, c='r')
                    plt.title('GNN Baseline Loss')
                    plt.xlabel('Episodes')
                    plt.ylabel('loss')
                    plt.savefig(data_dir+'Exp_Loss_Plot.png')
                    if ep_counter % 10 == 0:
                        print("------------------------------------------")
                        print("last 10 episode reward", np.average(exp_reward[ep_counter-10:ep_counter]))
                        print("last 10 episode loss", np.average(exp_loss[ep_counter-10:ep_counter]))
                        current_time = time.strftime("%Y%m%d-%H%M%S")
                        models_dir = "gmodels"
                        if not os.path.exists(models_dir):
                            os.makedirs(models_dir)
                        model_name = f"trained_gat_dqn_{current_time}.pth"
                        save_path = os.path.join(models_dir,model_name)
                        torch.save(agent.qnetwork_local.state_dict(), save_path)
                    socket.send(b"ok")













