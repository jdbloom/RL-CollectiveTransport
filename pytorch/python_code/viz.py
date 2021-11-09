import argparse
import pandas as pd
import numpy as np
import math
import json
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("figure_path")
parser.add_argument("figure_name")
parser.add_argument("--IL", default = False, action = "store_true")

args = parser.parse_args()

data_path = args.data_path + 'Data/'

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
episode_success_reward = []
episode_success_index = []
episode_failure_reward = []
episode_failure_index = []
terminals = []
last_10_axis = []
last_10_rewards = []

episode_run_times = []
cumulative_episode_run_times = [0]

print('. . . Consolodating Model Data')
IL_flag = False
for episode in df_list:
    rewards = []
    intention_reward = []
    terminal = 0
    for t in range(len(episode[1])):
        rewards.append(episode[1]['reward'][t].strip('][').split(','))
        intention_reward.append(episode[1]['intention_reward'][t].strip('][').split(','))
        terminal+=1
        run_t = episode[1]['run_time'][t]
    episode_run_times.append(run_t)
    cumulative_episode_run_times.append(cumulative_episode_run_times[-1] + run_t)
    terminals.append(terminal)

    robot_rewards = []
    robot_intentions = []
    for i in range(len(rewards[0])):
        tmp_r = 0
        tmp_i = 0
        for j in range(len(rewards)):
            tmp_r += float(rewards[j][i])
            import ipdb; ipdb.set_trace()
            tmp_i += float(intention_reward[j][i])
        robot_rewards.append(tmp_r)
        robot_intentions.append(tmp_i)
    episode_rewards.append(robot_rewards)
    episode_intentions.append(robot_intentions)

robot_exp_rewards = []
robot_exp_intentions = []
for j in range(len(episode_rewards[0])):
    robot_exp_rewards.append([])
    robot_exp_intentions.append([])
    episode_success_reward.append([])
    episode_success_index.append([])
    episode_failure_reward.append([])
    episode_failure_index.append([])

    for i in range(len(episode_rewards)):
        robot_exp_rewards[j].append(episode_rewards[i][j])
        robot_exp_intentions[j].append(episode_intentions[i][j])
        if terminals[i] < 4500:
            episode_success_reward[j].append(episode_rewards[i][j])
            episode_success_index[j].append(i)
        else:
            episode_failure_reward[j].append(episode_rewards[i][j])
            episode_failure_index[j].append(i)
    last_10_axis.append(np.arange(10, len(episode_rewards), 10))
    last_10_rewards.append([np.average(robot_exp_rewards[j][i:i+10]) for i in last_10_axis[j]])
cumulative_episode_run_times.pop(0)

print('. . . Plotting')

success_colors = ['darkturquoise', 'lightgreen', 'khaki', 'violet', 'lightcoral', 'sandybrown']
fail_colors = ['steelblue', 'yellowgreen', 'darkkhaki', 'darkviolet', 'tomato', 'chocolate']
avg_colors = ['blue', 'green', 'gold', 'purple', 'red', 'saddlebrown']

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.xlabel('Episodes')
plt.ylabel('Reward')
for i in range(len(episode_success_reward)):
    plt.scatter(episode_success_index[i], episode_success_reward[i], c = success_colors[i])#, label = 'Reached Goal')
    plt.scatter(episode_failure_index[i], episode_failure_reward[i], c = fail_colors[i], marker = 'x')#, label = 'Failure')
    plt.plot(last_10_axis[i], last_10_rewards[i], c=avg_colors[i], label = 'Robot '+str(i))
plt.ylim(-40000, 500)
plt.legend(loc = 1)
plt.title(args.figure_name)
plt.savefig(args.figure_path+args.figure_name+".png")

plt.close()

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.xlabel('Episodes')
plt.ylabel('Reward')
for i in range(len(robot_exp_intentions)):
    plt.plot(robot_exp_intentions[i], c=avg_colors[i], label = 'Robot '+str(i))
plt.legend(loc = 1)
plt.title(args.figure_name + ' Intention Reward')
plt.savefig(args.figure_path+args.figure_name+"_Intention.png")

plt.close()

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.xlabel('Run Time (s)')
plt.ylabel('Reward')
for i in range(len(episode_success_reward)):
    plt.scatter(cumulative_episode_run_times, robot_exp_rewards[i], c = success_colors[i])#, label = 'Reached Goal')
    #plt.plot(last_10_axis[i], last_10_rewards[i], c=avg_colors[i], label = 'Robot '+str(i))
plt.ylim(-40000, 500)
#plt.legend(loc = 1)
plt.title(args.figure_name+' Run Time')
plt.savefig(args.figure_path+args.figure_name+"_Experiment_Run_Time.png")

plt.close()

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.xlabel('Episodes')
plt.ylabel('Run Time (s)')
plt.scatter(np.arange(0, len(episode_run_times), 1), episode_run_times, c = success_colors[i])#, label = 'Reached Goal')
    #plt.plot(last_10_axis[i], last_10_rewards[i], c=avg_colors[i], label = 'Robot '+str(i))
#plt.legend(loc = 1)
plt.title(args.figure_name+' Episode Run Time')
plt.savefig(args.figure_path+args.figure_name+"_Episode_Run_Time.png")
