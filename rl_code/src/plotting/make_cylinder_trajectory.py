import argparse
import pandas as pd
import numpy as np
import math
import json
import os
import matplotlib.pyplot as plt
import pickle

def angle_normalize_unsigned_deg(a):
  while a < 0: a += 360
  while a >= 360: a -= 360
  return a

def angle_normalize_signed_deg(a):
  while a < -180: a += 360
  while a >= 180: a -= 360
  return a


parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("--second_path")
parser.add_argument("--small", default = False, action = "store_true")
parser.add_argument("--non_convex", default = False, action = "store_true")
parser.add_argument("--test", default = False, action = "store_true")
parser.add_argument("--heading", default=False, action = 'store_true')
parser.add_argument("--orientation", default=False, action='store_true')
parser.add_argument("--intention", default=False, action='store_true')
parser.add_argument("--failures", default=False, action='store_true')
parser.add_argument("--plot_robots", default=False, action = 'store_true')
parser.add_argument("--label_1")
parser.add_argument("--label_2")

args = parser.parse_args()

data_path = args.data_path + 'Data/'

print('. . . Loading Model Data')

file_names = []
for file in os.listdir(data_path):
    file_names.append(file)

df_list = []
for ep in range(len(file_names)-1):
    name = 'Data_Episode_'+str(ep)+'.pkl'
    file_path = data_path+name
    with open(data_path+name, 'rb') as f:
        data = pickle.load(f)

    cyl_heading_diff = []
    cyl_angle = data['cyl_angle']
    gsp = data['gsp_heading']
    predicted_cyl_heading = []
    for i in range(len(data['cyl_angle'])-1):
        predicted_cyl_heading.append(cyl_angle[i] + math.degrees(gsp[i+1]/10))
        old_cyl_ang = angle_normalize_unsigned_deg(data['cyl_angle'][i])
        new_cyl_ang = angle_normalize_unsigned_deg(data['cyl_angle'][i+1])
        diff = angle_normalize_signed_deg(new_cyl_ang-old_cyl_ang)
        diff = math.radians(diff)
        # Max rotation is 0.09 rad/step so we can multiply by 10 to get within range of -1, 1
        diff = np.clip(diff*100, -1, 1)

        # normalize between -1 and 1
        cyl_heading_diff.append(diff)
        
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20), gridspec_kw={'height_ratios': [3, 1]})
    plt.rcParams.update({'font.size': 22})
    if args.small:
        ax1.plot((-3.048, 3.048, 3.048, -3.048, -3.048), (-3.048, -3.048, 3.048, 3.048, -3.048), c= 'k')
        ax1.add_patch(plt.Circle((1.0, 0), 0.424, color='black', fill = False, linestyle = '--', linewidth=3.0))
        ax1.text(1.0, -0.25, 'GOAL', fontsize=20)
        ax1.set_xlim(-4, 4)
        ax1.set_xticks(np.arange(-4, 5, 1))
        ax1.set_ylim(-4, 4)
        ax1.set_yticks(np.arange(-4, 5, 1))
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.axis('off')
        if data['gate_stats'][0] != 0:
            ax1.plot((np.float64(data['gate_stats'][0][0]), np.float64(data['gate_stats'][0][0])), (-5, (-5 + np.float64(data['gate_stats'][0][1]))), c='k', linewidth = 5)
            ax1.plot((np.float64(data['gate_stats'][0][0]), np.float64(data['gate_stats'][0][0])), (5, (5-np.float64(data['gate_stats'][0][3]))), c='k', linewidth = 5)
            ax1.plot((-10, 10), (0, 0), c='r', linestyle='--')
        if data['obstacle_stats'][0] != 0:
            for i in range(int(len(data['obstacle_stats'][0])/2)):
                ax1.add_patch(plt.Circle((np.float64(data['obstacle_stats'][0][i*2]), np.float64(data['obstacle_stats'][0][i*2+1])), 0.15, color = 'black'))
        ax1.add_patch(plt.Circle((data['cyl_x_pos'][0], data['cyl_y_pos'][0]), 0.12, facecolor = 'lightgray', edgecolor='black'))
        ax1.add_patch(plt.Circle((data['cyl_x_pos'][-1], data['cyl_y_pos'][-1]), 0.12, facecolor = 'lightgray', edgecolor='black'))

    else:
        ax1.plot((-10, 10, 10, -10, -10), (-5, -5, 5, 5, -5), c= 'k')
        ax1.add_patch(plt.Circle((4.5, 0), 2, color='black', fill = False, linestyle = '--', linewidth=3.0))
        ax1.text(4, -0.25, 'GOAL', fontsize=20)
        ax1.set_xlim(-11, 11)
        ax1.set_xticks(np.arange(-11, 12, 1))
        ax1.set_ylim(-6, 6)
        ax1.set_yticks(np.arange(-6, 7, 1))
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.axis('off')
        if data['gate_stats'][0] != 0:
            ax1.plot((np.float64(data['gate_stats'][0][0]), np.float64(data['gate_stats'][0][0])), (-5, (-5 + np.float64(data['gate_stats'][0][1]))), c='k', linewidth = 5)
            ax1.plot((np.float64(data['gate_stats'][0][0]), np.float64(data['gate_stats'][0][0])), (5, (5-np.float64(data['gate_stats'][0][3]))), c='k', linewidth = 5)
            ax1.plot((-10, 10), (0, 0), c='r', linestyle='--')
        if data['obstacle_stats'][0] != 0:
            for i in range(int(len(data['obstacle_stats'][0])/2)):
                ax1.add_patch(plt.Circle((np.float64(data['obstacle_stats'][0][i*2]), np.float64(data['obstacle_stats'][0][i*2+1])), 0.5, color = 'black'))
        ax1.add_patch(plt.Circle((data['cyl_x_pos'][0], data['cyl_y_pos'][0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
        ax1.add_patch(plt.Circle((data['cyl_x_pos'][-1], data['cyl_y_pos'][-1]), 0.5, facecolor = 'lightgray', edgecolor='black'))

    ax1.plot(data['cyl_x_pos'], data['cyl_y_pos'], c='r', linewidth=5)
    
    if data['gate_stats'][0] != 0:
       ax1.plot((np.float64(data['gate_stats'][0][0]), np.float64(data['gate_stats'][0][0])), (-5, (-5 + np.float64(data['gate_stats'][0][1]))), c='k', linewidth = 5)
       ax1.plot((np.float64(data['gate_stats'][0][0]), np.float64(data['gate_stats'][0][0])), (5, (5-np.float64(data['gate_stats'][0][3]))), c='k', linewidth = 5)
       ax1.plot((-10, 10), (0, 0), c='r', linestyle='--')

    ax1.text(data['cyl_x_pos'][0]-0.5, data['cyl_y_pos'][0]+1, 'START', fontsize = 20.0)
    
    ax2.plot(data['cyl_angle'], c='b', label = 'Swarm Heading')
    ax2.plot(np.arange(10, len(predicted_cyl_heading), 10), np.array([np.average(predicted_cyl_heading[i-10:i]) for i in range(10, len(predicted_cyl_heading), 10)]), c='r', label = 'Predicted Swarm Heading')

    ax2.set_yticks(np.arange(-180, 185, 90))
    ax2.set_title("Swarm Heading")
    ax2.legend(loc='upper right')
    print('saving ... Trajectory'+'_'+str(ep)+'.png')
    plt.savefig(args.data_path+'/plots/Trajectory'+'_'+str(ep)+'.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rcParams.update({'font.size': 22})
    plt.plot(data['gsp_heading'], c= 'lightblue')
    last_10_gsp = np.array([np.average(data['gsp_heading'][i-10:i]) for i in range(10, len(data['gsp_heading']), 10)])
    last_10_gsp_index = np.arange(10, len(data['gsp_heading']), 10)
    plt.plot(last_10_gsp_index, last_10_gsp, c='b', label='GSP')
    plt.plot(cyl_heading_diff, c='r', label='Actual Heading Change')
    plt.legend()
    plt.savefig(args.data_path+'/plots/GSP_Heading_'+str(ep)+'.png')
    plt.close()