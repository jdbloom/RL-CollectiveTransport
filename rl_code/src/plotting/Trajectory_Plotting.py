#OG Data: /media/josh/12c1249f-83c4-4056-874b-e4f34a6793bf/home/jbloom/Documents/Nest/Research/RL-CollectiveTransport/pytorch/python_code/Data
import argparse
import pandas as pd
import numpy as np
import math
import json
import os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("figure_path")
parser.add_argument("figure_name")
parser.add_argument("--source_path")
parser.add_argument("--Episode")
parser.add_argument("--max_episodes")
parser.add_argument("--DQN_path")
parser.add_argument("--DDQN_path")
parser.add_argument("--DDPG_path")
parser.add_argument("--TD3_path")
parser.add_argument("--DQN2_path")
parser.add_argument("--DDQN2_path")
parser.add_argument("--DDPG2_path")
parser.add_argument("--TD32_path")
parser.add_argument("--gate")


args = parser.parse_args()

if args.Episode is not None:
    episodes = [args.Episode]
elif args.max_episodes is not None:
    episodes = [i for i in range(int(args.max_episodes))]

for j in range(len(episodes)):
    print('\n. . . Episode %i . . .' % j)
    if args.DQN_path is not None:
        print('. . . Loading DQN Model Data')
        data_path = args.DQN_path
        file_names = []
        file_path = data_path+'Data/Data_Episode_'+str(episodes[j])+'.csv'
        if args.source_path is not None:
            DQN_data = pd.read_csv(args.source_path+file_path)
        else: DQN_data = pd.read_csv(file_path)

    if args.DQN2_path is not None:
        print('. . . Loading DQN 2 Model Data')
        data_path = args.DQN2_path
        file_names = []
        file_path = data_path+'Data/Data_Episode_'+str(episodes[j])+'.csv'
        if args.source_path is not None:
            DQN2_data = pd.read_csv(args.source_path+file_path)
        else:DQN2_data = pd.read_csv(file_path)

    if args.DDQN_path is not None:
        print('. . . Loading DDQN Model Data')
        data_path = args.DDQN_path
        file_names = []
        file_path = data_path+'Data/Data_Episode_'+str(episodes[j])+'.csv'
        if args.source_path is not None:
            DDQN_data = pd.read_csv(args.source_path + file_path)
        else: DDQN_data = pd.read_csv(file_path)

    if args.DDQN2_path is not None:
        print('. . . Loading DDQN 2 Model Data')
        data_path = args.DDQN2_path
        file_names = []
        file_path = data_path+'Data/Data_Episode_'+str(episodes[j])+'.csv'
        if args.source_path is not None:
            DDQN2_data = pd.read_csv(args.source_path + file_path)
        else: DDQN2_data = pd.read_csv(file_path)

    if args.DDPG_path is not None:
        print('. . . Loading DDPG Model Data')
        data_path = args.DDPG_path
        file_names = []
        file_path = data_path+'Data/Data_Episode_'+str(episodes[j])+'.csv'
        if args.source_path is not None:
            DDPG_data = pd.read_csv(args.source_path + file_path)
        else: DDPG_data = pd.read_csv(file_path)

    if args.DDPG2_path is not None:
        print('. . . Loading DDPG 2 Model Data')
        data_path = args.DDPG2_path
        file_names = []
        file_path = data_path+'Data/Data_Episode_'+str(episodes[j])+'.csv'
        if args.source_path is not None:
            DDPG2_data = pd.read_csv(args.source_path + file_path)
        else: DDPG2_data = pd.read_csv(file_path)

    if args.TD3_path is not None:
        print('. . . Loading TD3 Model Data')
        data_path = args.TD3_path
        file_names = []
        file_path = data_path+'Data/Data_Episode_'+str(episodes[j])+'.csv'
        if args.source_path is not None:
            TD3_data = pd.read_csv(args.source_path+file_path)
        else: TD3_data = pd.read_csv(file_path)
    if args.TD32_path is not None:
        print('. . . Loading TD3 2 Model Data')
        data_path = args.TD32_path
        file_names = []
        file_path = data_path+'Data/Data_Episode_'+str(episodes[j])+'.csv'
        if args.source_path is not None:
            TD32_data = pd.read_csv(args.source_path+file_path)
        else: TD32_data = pd.read_csv(file_path)


    if args.gate is None:
        DQN_label = 'DQN'
        DQN2_label = 'DQN 0'
        DDQN_label = 'DDQN'
        DDQN2_label = 'DDQN 0'
        DDPG_label = 'DDPG'
        DDPG2_label = 'DDPG 0'
        TD3_label = 'TD3'
        TD32_label = 'TD3 0'
    else:
        DQN_label = 'DQN'
        DQN2_label = 'DQN No Gate'
        DDQN_label = 'DDQN'
        DDQN2_label = 'DDQN No Gate'
        DDPG_label = 'DDPG'
        DDPG2_label = 'DDPG No Gate'
        TD3_label = 'TD3'
        TD32_label = 'TD3 No Gate'

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#dede00']
    DQN_color = CB_color_cycle[0]
    DDQN_color = CB_color_cycle[1]
    DDPG_color = CB_color_cycle[2]
    TD3_color = CB_color_cycle[3]
    DQN2_color = CB_color_cycle[4]
    DDQN2_color = CB_color_cycle[5]
    DDPG2_color = CB_color_cycle[6]
    TD3_color = CB_color_cycle[7]

    print('. . . Plotting Data')
    ep = 0
    Data = []
    pos_gate_x = 0
    pos_gate_length = 0
    neg_gate_x = 0
    neg_gate_length = 0
    DQN_cyl_x_pos = []
    DQN_cyl_y_pos = []
    DDQN_cyl_x_pos = []
    DDQN_cyl_y_pos = []
    DDPG_cyl_x_pos = []
    DDPG_cyl_y_pos = []
    TD3_cyl_x_pos = []
    TD3_cyl_y_pos = []
    DQN2_cyl_x_pos = []
    DQN2_cyl_y_pos = []
    DDQN2_cyl_x_pos = []
    DDQN2_cyl_y_pos = []
    DDPG2_cyl_x_pos = []
    DDPG2_cyl_y_pos = []
    TD32_cyl_x_pos = []
    TD32_cyl_y_pos = []


    if args.DQN_path is not None:
        general_episode_data = DQN_data
        for t in range(len(DQN_data)):
            DQN_cyl_x_pos.append(DQN_data['cyl_x_pos'][t])
            DQN_cyl_y_pos.append(DQN_data['cyl_y_pos'][t])
    if args.DDQN_path is not None:
        general_episode_data = DDQN_data
        for t in range(len(DDQN_data)):
            DDQN_cyl_x_pos.append(DDQN_data['cyl_x_pos'][t])
            DDQN_cyl_y_pos.append(DDQN_data['cyl_y_pos'][t])
    if args.DDPG_path is not None:
        general_episode_data = DDPG_data
        for t in range(len(DDPG_data)):
            DDPG_cyl_x_pos.append(DDPG_data['cyl_x_pos'][t])
            DDPG_cyl_y_pos.append(DDPG_data['cyl_y_pos'][t])
    if args.TD3_path is not None:
        general_episode_data = TD3_data
        for t in range(len(TD3_data)):
            TD3_cyl_x_pos.append(TD3_data['cyl_x_pos'][t])
            TD3_cyl_y_pos.append(TD3_data['cyl_y_pos'][t])
    if args.DQN2_path is not None:
        general_episode_data = DQN2_data
        for t in range(len(DQN2_data)):
            DQN2_cyl_x_pos.append(DQN2_data['cyl_x_pos'][t])
            DQN2_cyl_y_pos.append(DQN2_data['cyl_y_pos'][t])
    if args.DDQN2_path is not None:
        general_episode_data = DDQN2_data
        for t in range(len(DDQN2_data)):
            DDQN2_cyl_x_pos.append(DDQN2_data['cyl_x_pos'][t])
            DDQN2_cyl_y_pos.append(DDQN2_data['cyl_y_pos'][t])
    if args.DDPG2_path is not None:
        general_episode_data = DDPG2_data
        for t in range(len(DDPG2_data)):
            DDPG2_cyl_x_pos.append(DDPG2_data['cyl_x_pos'][t])
            DDPG2_cyl_y_pos.append(DDPG2_data['cyl_y_pos'][t])
    if args.TD32_path is not None:
        general_episode_data = TD32_data
        for t in range(len(TD32_data)):
            TD32_cyl_x_pos.append(TD32_data['cyl_x_pos'][t])
            TD32_cyl_y_pos.append(TD32_data['cyl_y_pos'][t])
    else:
        assert("ERROR: No Data To Pull From")

    gate = general_episode_data['gate_stats'][0]
    obstacles = general_episode_data['obstacle_stats'][0]

    if gate != 0:
        gate = gate.strip('][').split(',')
    if obstacles != 0:
        obstacles = obstacles.strip('][').split(',')

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rcParams.update({'font.size': 22})
    plt.plot((-10, 10, 10, -10, -10), (-5, -5, 5, 5, -5), c= 'k')

    if args.DQN_path is not None:
        plt.plot(DQN_cyl_x_pos, DQN_cyl_y_pos, c = DQN_color, linewidth = '5', label = DQN_label)
    if args.DQN2_path is not None:
        plt.plot(DQN2_cyl_x_pos, DQN2_cyl_y_pos, DQN2_color, linewidth = '5', label = DQN2_label)
    if args.DDQN_path is not None:
        plt.plot(DDQN_cyl_x_pos, DDQN_cyl_y_pos, c = DDQN_color, linewidth = '5', label = DDQN_label)
    if args.DDQN2_path is not None:
        plt.plot(DDQN2_cyl_x_pos, DDQN2_cyl_y_pos, c = DDQN2_color, linewidth = '5', label = DDQN2_label)
    if args.DDPG_path is not None:
        plt.plot(DDPG_cyl_x_pos, DDPG_cyl_y_pos, c = DDPG_color, linewidth = '5', label = DDPG_label)
    if args.DDPG2_path is not None:
        plt.plot(DDPG2_cyl_x_pos, DDPG2_cyl_y_pos, c = DDPG2_color, linewidth = '5', label = DDPG2_label)
    if args.TD3_path is not None:
        plt.plot(TD3_cyl_x_pos, TD3_cyl_y_pos, c = TD3_color, linewidth = '5', label = TD3_label)
    if args.TD32_path is not None:
        plt.plot(TD32_cyl_x_pos, TD32_cyl_y_pos, c = TD32_color, linewidth = '5', label = TD32_label)

    if gate != 0:
        plt.plot((np.float64(gate[0]), np.float64(gate[0])), (-5, (-5 + np.float64(gate[1]))), c='k', linewidth = 5)
        plt.plot((np.float64(gate[0]), np.float64(gate[0])), (5, (5-np.float64(gate[3]))), c='k', linewidth = 5)
    if obstacles != 0:
        for i in range(int(len(obstacles)/2)):
            ax.add_patch(plt.Circle((np.float64(obstacles[i*2]), np.float64(obstacles[i*2+1])), 0.5, color = 'black'))
    if args.DQN_path is not None:
        ax.add_patch(plt.Circle((DQN_cyl_x_pos[0], DQN_cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax.add_patch(plt.Circle((DQN_cyl_x_pos[-1], DQN_cyl_y_pos[-1]), 0.5, facecolor = DQN_color, edgecolor='black'))
    if args.DDQN_path is not None:
        ax.add_patch(plt.Circle((DDQN_cyl_x_pos[0], DDQN_cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax.add_patch(plt.Circle((DDQN_cyl_x_pos[-1], DDQN_cyl_y_pos[-1]), 0.5, facecolor = DDQN_color, edgecolor='black'))
    if args.DDPG_path is not None:
        ax.add_patch(plt.Circle((DDPG_cyl_x_pos[0], DDPG_cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax.add_patch(plt.Circle((DDPG_cyl_x_pos[-1], DDPG_cyl_y_pos[-1]), 0.5, facecolor = DDPG_color, edgecolor='black'))
    if args.TD3_path is not None:
        ax.add_patch(plt.Circle((TD3_cyl_x_pos[0], TD3_cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax.add_patch(plt.Circle((TD3_cyl_x_pos[-1], TD3_cyl_y_pos[-1]), 0.5, facecolor = TD3_color, edgecolor='black'))
    if args.DQN2_path is not None:
        ax.add_patch(plt.Circle((DQN2_cyl_x_pos[0], DQN2_cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax.add_patch(plt.Circle((DQN2_cyl_x_pos[-1], DQN2_cyl_y_pos[-1]), 0.5, facecolor = DQN2_color, edgecolor=black))
    if args.DDQN2_path is not None:
        ax.add_patch(plt.Circle((DDQN2_cyl_x_pos[0], DDQN2_cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax.add_patch(plt.Circle((DDQN2_cyl_x_pos[-1], DDQN2_cyl_y_pos[-1]), 0.5, facecolor = DDQN2_color, edgecolor='black'))
    if args.DDPG2_path is not None:
        ax.add_patch(plt.Circle((DDPG2_cyl_x_pos[0], DDPG2_cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax.add_patch(plt.Circle((DDPG2_cyl_x_pos[-1], DDPG2_cyl_y_pos[-1]), 0.5, facecolor = DDPG2_color, edgecolor='black'))
    if args.TD32_path is not None:
        ax.add_patch(plt.Circle((TD32_cyl_x_pos[0], TD32_cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax.add_patch(plt.Circle((TD32_cyl_x_pos[-1], TD32_cyl_y_pos[-1]), 0.5, facecolor = TD32_color, edgecolor='black'))

    ax.add_patch(plt.Circle((4.5, 0), 2, color='black', fill = False, linestyle = '--', linewidth='3'))
    plt.text(general_episode_data['cyl_x_pos'][0]-0.5, general_episode_data['cyl_y_pos'][0]+1, 'START', fontsize = 20)
    plt.text(4, -0.25, 'GOAL', fontsize=20)
    plt.xlim(-11, 11, 1)
    plt.xticks(np.arange(-11, 12, 1), fontsize = 18)
    plt.ylim(-6, 6, 1)
    plt.yticks(np.arange(-6, 7, 1), fontsize = 18)
    plt.axis('off')
    plt.legend(loc = 'upper right', facecolor = 'white', edgecolor = 'k', framealpha = 1.0)
    plt.savefig(args.figure_path+args.figure_name+'_'+str(episodes[j])+'.png')
    plt.close()
