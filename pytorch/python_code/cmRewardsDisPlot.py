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
parser.add_argument("--episodes", default=999)
parser.add_argument("--num_robots", default = 4)
parser.add_argument("--intention", default=False, action="store_true")
parser.add_argument("--object_radius", default = 0.12)

args = parser.parse_args()

data_path = args.source_path

n_eps = int(args.episodes)
object_radius = float(args.object_radius)

print('. . . Loading Model Data')
file_names = []
for file in os.listdir(data_path):
    file_names.append(file)

df_list = []
for i in range(n_eps):
    name = '/Data_Episode_'+str(i)+'.csv'
    file_path = data_path+name
    df = pd.read_csv(file_path)
    df_list.append((i, df))

episode_rewards = []
episode_intentions = []
episode_intention_rewards = []

rewards = []
rewards_std = []
intention_reward = []
intention_std = []
prediction_offset = []
prediction_offset_std = []
i_prediction_offset = []
i_prediction_offset_std = []

n_robots = int(args.num_robots)

print('. . . Consolodating Model Data')
IL_flag = False

for episode in df_list:
    pos_mx_cyl = np.asarray(episode[1]['cyl_mod_x'])
    pos_my_cyl = np.asarray(episode[1]['cyl_mod_y'])
    cyl_radius = np.asarray(episode[1]['cyl_radius'])
    pred_x_robots = np.empty([len(episode[1]['agent_predictions_x']),n_robots])
    pred_y_robots = np.empty([len(episode[1]['agent_predictions_y']),n_robots])
    pos_x_robots = np.empty([len(episode[1]['robots_x_pos']),n_robots])
    pos_y_robots = np.empty([len(episode[1]['robots_y_pos']),n_robots])
    intention_pred_x = np.empty([len(episode[1]['agent_predictions_y']),n_robots])
    intention_pred_y = np.empty([len(episode[1]['agent_predictions_y']),n_robots])
    intention_error_cm = np.empty([len(episode[1]['agent_predictions_y'])])
    r = np.empty([len(episode[1]['agent_predictions_y']),n_robots])
    inte = np.empty([len(episode[1]['agent_predictions_y']),n_robots])
    for i in range(len(pos_mx_cyl)): # For each episode
        r[i] = np.fromstring(episode[1]['reward'][i].strip(']['), sep=',')
        inte[i] = np.fromstring(episode[1]['intention_reward'][i].strip(']['), sep=',')

        if args.intention:
            # intention_angle = np.fromstring(episode[1]['intention_angle'][i].strip(']['), sep=' ')
            intention_radius = np.fromstring(episode[1]['intention_radius'][i].strip(']['), sep=' ')[0]
            print(episode[1]['intention_error'][i].strip('[]'), "\n")
            intention_error_cm = np.fromstring(episode[1]['intention_error'][i].strip(']['), sep=' ')

            pos_x_robots[i] = np.fromstring(episode[1]['robots_x_pos'][i].strip('[]'), sep=',')
            pos_y_robots[i] = np.fromstring(episode[1]['robots_y_pos'][i].strip('[]'), sep=',')

            # rob_to_cyl_ang = np.arctan2(pos_my_cyl[i] - pos_y_robots[i], pos_mx_cyl[0] - pos_x_robots[i])
            # angle_modified = rob_to_cyl_ang + intention_angle * math.pi / 2
            # intention_pred_x[i] = np.cos(angle_modified) * intention_radius * cyl_radius[1] + pos_x_robots[i]
            # intention_pred_y[i] = np.sin(angle_modified) * intention_radius * cyl_radius[1] + pos_y_robots[i]


        pred_x_robots[i] = np.fromstring(episode[1]['agent_predictions_x'][i].strip('[]'), sep=',')
        pred_y_robots[i] = np.fromstring(episode[1]['agent_predictions_y'][i].strip('[]'), sep=',')
    
    error = np.sqrt(np.square(np.subtract(pred_x_robots,pos_mx_cyl[:,None])) + np.square(np.subtract(pred_y_robots,pos_my_cyl[:,None])))
    if args.intention:
        # ierror = np.sqrt(np.square(np.subtract(intention_pred_x,pos_mx_cyl[:,None])) + np.square(np.subtract(intention_pred_y,pos_my_cyl[:,None])))
        ierror = intention_error_cm
    else:
        ierror = 0

    rewards.append(np.mean(r))
    rewards_std.append(np.std(r))
    intention_reward.append(np.mean(inte))
    intention_std.append(np.std(inte))

    prediction_offset.append(np.mean(error))
    prediction_offset_std.append(np.std(error))

    i_prediction_offset.append(np.mean(ierror))
    i_prediction_offset_std.append(np.std(ierror))

rewards = np.asarray(rewards)
rewards_std = np.asarray(rewards_std)
intention_reward = np.asarray(intention_reward)
intention_std = np.asarray(intention_std)
prediction_offset = np.asarray(prediction_offset)
prediction_offset_std = np.asarray(prediction_offset_std) 
i_prediction_offset = np.asarray(i_prediction_offset)
print(np.argmin(i_prediction_offset))
i_prediction_offset_std = np.asarray(i_prediction_offset_std) 


print('\n. . . Live Plotting Full Experiment From Episodes 0 to '+ str(args.episodes) +' . . .')

success_colors = ['darkturquoise', 'lightgreen', 'khaki', 'violet', 'lightcoral', 'sandybrown', 'mediumpurple', 'plum']
fail_colors = ['steelblue', 'yellowgreen', 'darkkhaki', 'darkviolet', 'tomato', 'chocolate', 'crimson', 'orangered']
avg_colors = ['blue', 'green', 'gold', 'purple', 'red', 'saddlebrown', 'yellowgreen', 'darkcyan']

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title("Training Rewards")

plt.plot(np.linspace(0,n_eps,n_eps), rewards, c = success_colors[0], label = "Rewards")
plt.fill_between(np.linspace(0,n_eps,n_eps), rewards - rewards_std, rewards + rewards_std, color=success_colors[0], alpha = 0.25)
if args.intention:
    plt.plot(np.linspace(0,n_eps,n_eps), intention_reward, c = success_colors[1], label = "Intention Assisted Rewards")
    plt.fill_between(np.linspace(0,n_eps,n_eps), intention_reward - intention_std, intention_reward + intention_std, color=success_colors[1], alpha = 0.25)

plt.legend(loc='upper right')

plt.show()

plt.close()

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.xlabel('Episodes')
plt.ylabel('Prediction Accuracy (cm)')
plt.title('Robot Predictions Error')
plt.plot(np.linspace(0,n_eps, n_eps), prediction_offset, c=avg_colors[1], label = "Unassisted Prediction")
plt.fill_between(np.linspace(0,n_eps,n_eps), prediction_offset - prediction_offset_std, prediction_offset + prediction_offset_std, color=avg_colors[1], alpha = 0.25)

if args.intention:
    print("Also showing intention on graph")
    plt.plot(np.linspace(0,n_eps, n_eps), i_prediction_offset, c=avg_colors[3], label = "Intention Assisted Prediction")
    plt.fill_between(np.linspace(0,n_eps,n_eps), i_prediction_offset - i_prediction_offset_std, i_prediction_offset + i_prediction_offset_std, color=avg_colors[3], alpha = 0.25)

plt.legend(loc='upper right')

plt.show()

plt.close()