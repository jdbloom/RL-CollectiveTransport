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
parser.add_argument("--second_path")
parser.add_argument("--non_convex", default = False, action = "store_true")
parser.add_argument("--test", default = False, action = "store_true")
parser.add_argument("--label_1")
parser.add_argument("--label_2")
args = parser.parse_args()

print('. . . Loading Model Data')
data_path = args.data_path + 'Data/'
file_names = []
for file in os.listdir(data_path):
    file_names.append(file)

df_list = []
for i in range(len(file_names)-1):
    name = 'Data_Episode_'+str(i)+'.csv'
    file_path = data_path+name
    df = pd.read_csv(file_path)
    df_list.append((i, df))

if args.second_path is not None:
    print('. . . Loading Second Model Data')
    second_data_path = args.second_path + 'Data/'
    second_file_names = []
    for file in os.listdir(second_data_path):
        second_file_names.append(file)

    second_df_list = []
    for i in range(len(second_file_names)-1):
        name = 'Data_Episode_'+str(i)+'.csv'
        file_path = second_data_path+name
        df = pd.read_csv(file_path)
        second_df_list.append((i, df))


print('. . . Plotting Model Data')
ep = 0

for ep in range(len(df_list)):
    episode = df_list[ep]
    episode_2 = None
    if args.second_path is not None:
        episode_2 = second_df_list[ep]
    cyl_x_pos = []
    cyl_y_pos = []
    cyl_x_pos_2 = []
    cyl_y_pos_2 = []
    pos_gate_x = 0
    pos_gate_length = 0
    neg_gate_x = 0
    neg_gate_length = 0

    for t in range(len(episode[1])):
        cyl_x_pos.append(episode[1]['cyl_x_pos'][t])
        cyl_y_pos.append(episode[1]['cyl_y_pos'][t])
    if episode_2 is not None:
        for t in range(len(episode_2[1])):
            cyl_x_pos_2.append(episode_2[1]['cyl_x_pos'][t])
            cyl_y_pos_2.append(episode_2[1]['cyl_y_pos'][t])
    gate = episode[1]['gate_stats'][0]
    obstacles = episode[1]['obstacle_stats'][0]

    if gate != 0:
        gate = gate.strip('][').split(',')
    if obstacles != 0:
        obstacles = obstacles.strip('][').split(',')

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rcParams.update({'font.size': 22})
    plt.plot((-10, 10, 10, -10, -10), (-5, -5, 5, 5, -5), c= 'k')

    if episode_2 is not None:
        plt.plot(cyl_x_pos, cyl_y_pos, 'r--', linewidth = '5', label = args.label_1)
        plt.plot(cyl_x_pos_2, cyl_y_pos_2, 'b-.',  linewidth = '5', label = args.label_2)
    else:
        plt.plot(cyl_x_pos, cyl_y_pos, 'r', linewidth = '5', label = args.label_1)
    if args.non_convex:
        plt.plot((-2, 0, -2), (2, 0, -2), c = 'k', linewidth = 10)
    if gate != 0:
        plt.plot((np.float64(gate[0]), np.float64(gate[0])), (-5, (-5 + np.float64(gate[1]))), c='k', linewidth = 5)
        plt.plot((np.float64(gate[0]), np.float64(gate[0])), (5, (5-np.float64(gate[3]))), c='k', linewidth = 5)
    if obstacles != 0:
        for i in range(int(len(obstacles)/2)):
            ax.add_patch(plt.Circle((np.float64(obstacles[i*2]), np.float64(obstacles[i*2+1])), 0.5, color = 'black'))
    ax.add_patch(plt.Circle((cyl_x_pos[0], cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
    ax.add_patch(plt.Circle((cyl_x_pos[-1], cyl_y_pos[-1]), 0.5, facecolor = 'lightgray', edgecolor='black'))
    if episode_2 is not None:
        ax.add_patch(plt.Circle((cyl_x_pos_2[-1], cyl_y_pos_2[-1]), 0.5, facecolor = 'lightgray', edgecolor='black'))
    ax.add_patch(plt.Circle((4.5, 0), 2, color='black', fill = False, linestyle = '--', linewidth='3'))
    plt.text(cyl_x_pos[0]-0.5, cyl_y_pos[0]+1, 'START', fontsize = 20)
    plt.text(4, -0.25, 'GOAL', fontsize=20)
    plt.xlim(-11, 11, 1)
    plt.xticks(np.arange(-11, 12, 1), fontsize = 18)
    plt.ylim(-6, 6, 1)
    plt.yticks(np.arange(-6, 7, 1), fontsize = 18)
    plt.axis('off')
    if args.label_1 is not None:
        plt.legend(loc = 'upper right')
    plt.savefig(args.figure_path+args.figure_name+'_'+str(ep)+'.png')
    plt.close()
    ep += 1
