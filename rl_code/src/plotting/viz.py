import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("--gsp_log_scale", default=False, action="store_true")
parser.add_argument("--IL", default = False, action = "store_true")
parser.add_argument("--gate", default = False, action = "store_true")

args = parser.parse_args()

data_path = args.data_path + 'Data/'

print('. . . Loading Model Data')
file_names = []
for file in os.listdir(data_path):
    file_names.append(file)

episode_rewards = []
episode_gsps = []
episode_gsp_rewards = []
episode_success_reward = []
episode_success_index = []
episode_failure_reward = []
episode_failure_index = []
terminals = []
last_10_axis = []
last_10_rewards = []

episode_run_times = []
cumulative_episode_run_times = [0]
episode_time_steps = []

print('. . . Consolodating Model Data')
IL_flag = False
for i in range(len(file_names)-1):
    name = 'Data_Episode_'+str(i)+'.pkl'
    with open(data_path+name, 'rb') as f:
        data = pickle.load(f)

        rewards = np.array(data['reward'])
        gsp_rewrads = np.array(data['gsp_reward'])

        robot_rewards = np.array([rewards[:, i] for i in range(rewards.shape[-1])])
        robot_gsp_rewards = np.array([gsp_rewrads[:, i] for i in range(gsp_rewrads.shape[-1])])
        episode_rewards.append([np.sum(robot_rewards[i]) for i in range(robot_rewards.shape[0])])
        episode_gsp_rewards.append([np.sum(robot_gsp_rewards[i]) for i in range(robot_gsp_rewards.shape[0])])
        if rewards.shape[0] < 4500:
            terminals.append(1)
        else:
            terminals.append(0)

episode_rewards = np.array(episode_rewards)
terminals = np.array(terminals)
episode_gsp_rewards = np.array(episode_gsp_rewards)[:,0]
episode_std = np.std(episode_rewards, axis=1)
average_episode_rewards = np.average(episode_rewards, axis=1)
episode_index = np.arange(0, average_episode_rewards.shape[0], 1)
last_10_rewards = np.array([np.average(average_episode_rewards[i-10:i]) for i in range(10, average_episode_rewards.shape[0], 10)])
last_10_gsp_rewards = np.array([np.average(episode_gsp_rewards[i-10:i]) for i in range(10, episode_gsp_rewards.shape[0], 10)])
last_10_std = np.array([np.average(episode_std[i-10:i]) for i in range(10, episode_std.shape[0], 10)])
last_10_gsp_std = np.array([np.std(episode_gsp_rewards[i-10:i]) for i in range(10, episode_gsp_rewards.shape[0], 10)])
last_10_axis = np.arange(10, average_episode_rewards.shape[0], 10)
last_10_success_pct = np.array([np.average(terminals[i-10:i]) for i in range(10, terminals.shape[0], 10)])

for i, ep in enumerate(last_10_axis):
    print(f'Model {ep} had Average Reward: {last_10_rewards[i]:.2f}, GSP Reward: {last_10_gsp_rewards[i]:.2f}, and Success Rate: {last_10_success_pct[i]}')
# for i in range(episode_rewards.shape[1]):
#     plt.plot(episode_rewards[:, i])
# plt.plot(average_episode_rewards)

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 5))


ax1.set_ylabel('Reward', c='b')
ax1.tick_params(axis='y', labelcolor='b')
if np.average(last_10_gsp_rewards) < 0:
    ax = ax1.twinx()
    ax.fill_between(last_10_axis, last_10_gsp_rewards+last_10_gsp_std/2.0, last_10_gsp_rewards-last_10_gsp_std/2.0, color='salmon', label = 'GSP Reward STD', alpha = 0.2)
    ax.plot(last_10_axis, last_10_gsp_rewards, color = 'r', label = 'GSP Reward')
    if args.gsp_log_scale:
        ax.set_yscale("symlog")
    ax.set_ylabel('GSP Reward', c='r')
    # ax.set_ylim(1000, 100)
    ax.tick_params(axis='y', labelcolor='r')
ax1.fill_between(last_10_axis, last_10_rewards+last_10_std/2.0, last_10_rewards-last_10_std/2.0, color='lightblue', label='Reward STD', alpha=0.5)
ax1.plot(last_10_axis, last_10_rewards, c='b', label='Reward')

ax2.plot(last_10_axis, last_10_success_pct*100, c='k')
ax2.set_ylabel('Success (%)')
ax2.set_xlabel('Episodes')
plt.title('Training Metrics')
plt.savefig(args.data_path+'Training_Metrics.png')


    
    # rewards = []
    # gsp_reward = []

    # robot_x_pos = []
    # robot_y_pos = []
    # terminal = 0
    # episode_time_steps.append(len(data['reward']))

    # robot_rewards = []
    # robot_gsps = []
    # failures = np.zeros(len(data['reward']))
    # failure_x_pos = np.zeros(len(data['reward']))
    # failure_y_pos = np.zeros(len(data['reward']))
    # for i in range(len(data['reward'])):
    #     robot_rewards.append(np.sum(data['reward'][:][i]))
    #     robot_gsps.append(tmp_i)
    # episode_rewards.append(robot_rewards)
    # episode_gsps.append(robot_gsps)



# robot_exp_rewards = []
# robot_exp_rewards_avg = []
# robot_exp_gsps = []
# robot_exp_gsps_avg = []
# for j in range(len(episode_rewards[0])):
#     robot_exp_rewards.append([])
#     robot_exp_rewards_avg.append([])
#     robot_exp_gsps.append([])
#     robot_exp_gsps_avg.append([])
#     episode_success_reward.append([])
#     episode_success_index.append([])
#     episode_failure_reward.append([])
#     episode_failure_index.append([])
#     for i in range(len(episode_rewards)):
#         robot_exp_rewards[j].append(episode_rewards[i][j])
#         robot_exp_rewards_avg[j].append(episode_rewards[i][j]/episode_time_steps[i])
#         robot_exp_gsps[j].append(episode_gsps[i][j])
#         robot_exp_gsps_avg[j].append(episode_gsps[i][j]/episode_time_steps[i])
#         if terminals[i] < 4500:
#             episode_success_reward[j].append(episode_rewards[i][j])
#             episode_success_index[j].append(i)
#         else:
#             episode_failure_reward[j].append(episode_rewards[i][j])
#             episode_failure_index[j].append(i)
#     last_10_axis.append(np.arange(10, len(episode_rewards), 10))
#     last_10_rewards.append([np.average(robot_exp_rewards[j][i-10:i]) for i in last_10_axis[j]])
# cumulative_episode_run_times.pop(0)

# print('. . . Statistics')
# [print('[Models]', last_10_axis[0][i], np.average([last_10_rewards[j][i] for j in range(len(last_10_axis))])) for i in range(len(last_10_axis[0]))]
# print('\n[STATISTICS] Success Rate:', (len(episode_success_reward[0])/ len(episode_rewards)))
# print('[STATISTICS] Failures:', len(episode_failure_reward[0]))
# print('[STATISTICS] Avg Exp Reward:', np.average(robot_exp_rewards))
# print('[STATISTICS] Avg Exp gsp Reward:', np.average(robot_exp_gsps))
# #print('[STATISTICS] Best Model:', last_10_axis[0][np.argmax(last_10_rewards[0])])
# #print('\n[FAILING EPISODES]', episode_failure_index[0])


# print('. . . Plotting')

# success_colors = ['darkturquoise', 'lightgreen', 'khaki', 'violet', 'lightcoral', 'sandybrown', 'mediumpurple', 'plum']
# fail_colors = ['steelblue', 'yellowgreen', 'darkkhaki', 'darkviolet', 'tomato', 'chocolate', 'crimson', 'orangered']
# avg_colors = ['blue', 'green', 'gold', 'purple', 'red', 'saddlebrown', 'yellowgreen', 'darkcyan']

# plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# for i in range(len(episode_success_reward)):
#     plt.scatter(episode_success_index[i], episode_success_reward[i], c = success_colors[i])#, label = 'Reached Goal')
#     plt.scatter(episode_failure_index[i], episode_failure_reward[i], c = fail_colors[i], marker = 'x')#, label = 'Failure')
#     plt.plot(last_10_axis[i], last_10_rewards[i], c=avg_colors[i], label = 'Robot '+str(i))
# if args.gate:
#     plt.plot((599, 599), (-14000, 4000), c = 'salmon')
# plt.ylim(-15000, 3000)
# plt.legend(loc = 1)
# plt.title(args.figure_name)
# plt.savefig(args.figure_path+args.figure_name+".png")

# plt.close()

# plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# for i in range(len(robot_exp_gsps)):
#     plt.plot(robot_exp_gsps[i], c=avg_colors[i], label = 'Robot '+str(i))
# plt.legend(loc = 1)
# plt.title(args.figure_name + ' gsp Reward')
# plt.savefig(args.figure_path+args.figure_name+"_gsp.png")

# plt.close()

# plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.plot(robot_exp_rewards_avg[0], c=avg_colors[0], label = 'Action Reward')
# plt.plot(robot_exp_gsps_avg[0], c=avg_colors[1], label = 'gsp Reward')
# #plt.legend(loc = 1)
# plt.title(args.figure_name + ' Average Time Step Reward ')
# plt.savefig(args.figure_path+args.figure_name+"_average_time_step_reward.png")

# plt.close()

# plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.plot(robot_exp_gsps_avg[0], c=avg_colors[1], label = 'gsp Reward')
# #plt.legend(loc = 1)
# plt.title(args.figure_name + ' Average Time Step gsp Reward ')
# plt.savefig(args.figure_path+args.figure_name+"_average_time_step_gsp.png")

# plt.close()
# '''
# plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})
# plt.xlabel('Run Time (s)')
# plt.ylabel('Reward')
# for i in range(len(episode_success_reward)):
#     plt.scatter(cumulative_episode_run_times, robot_exp_rewards[i], c = success_colors[i])#, label = 'Reached Goal')
#     #plt.plot(last_10_axis[i], last_10_rewards[i], c=avg_colors[i], label = 'Robot '+str(i))
# plt.ylim(-40000, 500)
# #plt.legend(loc = 1)
# plt.title(args.figure_name+' Run Time')
# plt.savefig(args.figure_path+args.figure_name+"_Experiment_Run_Time.png")

# plt.close()

# plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})
# plt.xlabel('Episodes')
# plt.ylabel('Run Time (s)')
# plt.scatter(np.arange(0, len(episode_run_times), 1), episode_run_times, c = success_colors[i])#, label = 'Reached Goal')
#     #plt.plot(last_10_axis[i], last_10_rewards[i], c=avg_colors[i], label = 'Robot '+str(i))
# #plt.legend(loc = 1)
# plt.title(args.figure_name+' Episode Run Time')
# plt.savefig(args.figure_path+args.figure_name+"_Episode_Run_Time.png")
# '''