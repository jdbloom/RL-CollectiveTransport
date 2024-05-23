import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import math
import json
import os
import argparse
import pandas as pd

from itertools import chain 

# $ python cmVizPlot.py --source_path recording_folder/April11DDPGAngleLengRobPlotting/ --Episode 400
# Current run command ^, when saving as figures, uncomment figure path and name and put those before all vars
parser = argparse.ArgumentParser()
# parser.add_argument("figure_path")
# parser.add_argument("figure_name")
parser.add_argument("--source_path")
parser.add_argument("--Episode", default=0)
parser.add_argument("--num_robots", default = 4)

args = parser.parse_args()

data_path = args.source_path

print('. . . Loading Model Data')
file_names = []
for file in os.listdir(data_path):
    file_names.append(file)

df_list = []
for i in range(len(file_names)-1):
    name = 'Data_Episode_'+str(i)+'.csv'
    file_path = data_path+name
    df = pd.read_csv(file_path)
    df_list.append((i, df))

episode_rewards = []
episode_intentions = []
episode_intention_rewards = []

print('. . . Consolodating Model Data')
IL_flag = False
for episode in df_list:
    rewards = []
    intention_reward = []
    for t in range(len(episode[1])):
        rewards.append(episode[1]['reward'][t].strip('][').split(','))
        intention_reward.append(episode[1]['intention_reward'][t].strip('][').split(','))

    robot_rewards = []
    robot_intentions = []
    for i in range(len(rewards[0])):
        tmp_r = 0
        tmp_i = 0
        for j in range(len(rewards)):
                if rewards[j][0] != 'reward':
                    tmp_r += float(rewards[j][i])
                    tmp_i += float(intention_reward[j][i])


        robot_rewards.append(tmp_r)
        robot_intentions.append(tmp_i)
    episode_rewards.append(robot_rewards)
    episode_intentions.append(robot_intentions)
    
robot_exp_rewards = []
robot_exp_rewards_avg = []
robot_exp_intentions = []
robot_exp_intentions_avg = []
for j in range(len(episode_rewards[0])):
    robot_exp_rewards.append([])
    robot_exp_rewards_avg.append([])
    robot_exp_intentions.append([])
    robot_exp_intentions_avg.append([])
    episode_success_reward.append([])
    episode_success_index.append([])
    episode_failure_reward.append([])
    episode_failure_index.append([])
    for i in range(len(episode_rewards)):
        robot_exp_rewards[j].append(episode_rewards[i][j])
        robot_exp_rewards_avg[j].append(episode_rewards[i][j]/episode_time_steps[i])
        robot_exp_intentions[j].append(episode_intentions[i][j])
        robot_exp_intentions_avg[j].append(episode_intentions[i][j]/episode_time_steps[i])

print('\n. . . Live Plotting Full Experiment '+args.Episode+' . . .')

CB_color_cycle = ['#377eb8', '8e41e0', 'e04161', 'cf6730','c91818', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#dede00']

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.xlabel('Episodes')
plt.ylabel('Reward')
for i in range(len(episode_success_reward)):

fig = plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 22})

opaqueness = 5
robot_center_op = 0.5
prediction_op = 0.8
decrement = 0.05

cyl = plt.plot([],[], c=CB_color_cycle[0], linewidth=3)
pos_r = [[] for i in range(args.num_robots)]
pos_p = [[] for i in range(args.num_robots)]
for i in range(args.num_robots):
    for j in range(opaqueness):
        pos_r[i].append(plt.plot([],[], c=CB_color_cycle[(1 + 2*i)%len(CB_color_cycle)], alpha=robot_center_op - j*decrement)[0])
        pos_p[i].append(plt.plot([],[], c=CB_color_cycle[(2 + 2*i)%len(CB_color_cycle)], alpha=prediction_op - j*decrement)[0])


plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Positions and Predictions in relation to Center of Mass')

# metadata = dict(title='Center of Mass', artist='Chandler Garcia')
# writer = animation.PillowWriter(fps=15, metadata=metadata)

# with writer.saving(fig, "CenterOfMassPrediction.gif", len(pos_x_cyl)):
#     for i in range(len(pos_x_cyl)):

visible_interval = 600
opacity_change = int(visible_interval / opaqueness)

def animate(i):
    cyl[0].set_data(pos_x_cyl[max(i-visible_interval, 0):i],pos_y_cyl[max(i-visible_interval, 0):i])
    for j in range(args.num_robots):
        for k in range(opaqueness):
            start = max(i - (opacity_change * k), 0)
            end = max(i - (opacity_change * (1 + k)), 0)
            pos_r[j][k].set_data(pos_x_robots[end:start,j],pos_y_robots[end:start,j])
            pos_p[j][k].set_data(pred_x_robots[end:start,j],pred_y_robots[end:start,j])
    # print(list(chain.from_iterable(pos_r)))
    return cyl[0], *list(chain.from_iterable(pos_r)), *list(chain.from_iterable(pos_p))
            
ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=50, frames = len(pos_x_robots))

plt.axis([-1,1,-0.25,1.75])

writer = animation.PillowWriter(fps=15, metadata=dict(artist='Chandler Garcia'), bitrate=1800)

plt.show()

plt.savefig(CM_Plot_+'_'+str(episodes[j])+'.png')
plt.close()