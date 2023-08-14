import argparse
import pandas as pd
import numpy as np
import math
import json
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("data_path_1")
parser.add_argument("data_path_2")
parser.add_argument("figure_path")
parser.add_argument("figure_name")
parser.add_argument("num_exps")
parser.add_argument("num_episodes")
parser.add_argument("--IL", default = False, action = "store_true")
parser.add_argument("--gate", default = False, action = "store_true")
plots =[]
args = parser.parse_args()
data_paths = [args.data_path_1, args.data_path_2]
labels = ['No Intention', 'DDPG-GNN Intention']
for path in range(2):
    exp_episode_status = [[] for ep in range(int(args.num_episodes))]
    for exp in range(int(args.num_exps)):
        data_path = data_paths[path] + str(exp) + '/Data/'

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

        print('. . . Consolodating Model Data')
        IL_flag = False
        for ep, episode in enumerate(df_list):
            rewards = []
            intention_reward = []

            robot_x_pos = []
            robot_y_pos = []
            terminal = 0
            for t in range(len(episode[1])):
                rewards.append(episode[1]['reward'][t].strip('][').split(','))
                intention_reward.append(episode[1]['intention_reward'][t].strip('][').split(','))
                terminal+=1
            terminals.append(terminal)

            robot_rewards = []
            robot_intentions = []

            for j in range(len(rewards)):
                if rewards[j][0] != 'reward':
                    robot_rewards.append(np.average([float(rewards[j][i]) for i in range(len(rewards[j]))]))
                    robot_intentions.append(np.average([float(intention_reward[j][i]) for i in range(len(rewards[j]))]))
            episode_rewards.append(robot_rewards)
            episode_intentions.append(robot_intentions)

            if terminals[ep] < 4500:
                exp_episode_status[ep].append(1)
            else:
                exp_episode_status[ep].append(0)

    exp_episode_scores = []
    for ep in range(int(args.num_episodes)):
        exp_episode_scores.append(np.average(exp_episode_status[ep]))
    plt.plot(exp_episode_scores, label = labels[path])

plt.title(args.figure_name)
plt.legend()
plt.xlabel('Episodes')
plt.xticks(np.arange(0, 20, 1))
plt.ylabel('Success Percentage (%)')
plt.savefig(args.figure_path+args.figure_name)


