import argparse
import pandas as pd
import numpy as np
import math
import json
import os
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("--second_path")
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
        
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rcParams.update({'font.size': 22})
    plt.plot((-10, 10, 10, -10, -10), (-5, -5, 5, 5, -5), c= 'k')
    ax.add_patch(plt.Circle((4.5, 0), 2, color='black', fill = False, linestyle = '--', linewidth=3.0))
    ax.add_patch(plt.Circle((data['cyl_x_pos'][0], data['cyl_y_pos'][0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
    ax.add_patch(plt.Circle((data['cyl_x_pos'][-1], data['cyl_y_pos'][-1]), 0.5, facecolor = 'lightgray', edgecolor='black'))
    ax.plot(data['cyl_x_pos'], data['cyl_y_pos'], c='r', linewidth=5)
    

    if data['gate_stats'][0] != 0:
       plt.plot((np.float64(data['gate_stats'][0][0]), np.float64(data['gate_stats'][0][0])), (-5, (-5 + np.float64(data['gate_stats'][0][1]))), c='k', linewidth = 5)
       plt.plot((np.float64(data['gate_stats'][0][0]), np.float64(data['gate_stats'][0][0])), (5, (5-np.float64(data['gate_stats'][0][3]))), c='k', linewidth = 5)
       plt.plot((-10, 10), (0, 0), c='r', linestyle='--')
    if data['obstacle_stats'][0] != 0:
        for i in range(int(len(data['obstacle_stats'][0])/2)):
            ax.add_patch(plt.Circle((np.float64(data['obstacle_stats'][0][i*2]), np.float64(data['obstacle_stats'][0][i*2+1])), 0.5, color = 'black'))

    plt.text(data['cyl_x_pos'][0]-0.5, data['cyl_y_pos'][0]+1, 'START', fontsize = 20)
    plt.text(4, -0.25, 'GOAL', fontsize=20)
    plt.xlim(-11, 11)
    plt.xticks(np.arange(-11, 12, 1), fontsize = 18)
    plt.ylim(-6, 6)
    plt.yticks(np.arange(-6, 7, 1), fontsize = 18)
    plt.axis('off')
    print('saving ... Trajectory'+'_'+str(ep)+'.png')
    plt.savefig(args.data_path+'/plots/Trajectory'+'_'+str(ep)+'.png')
    plt.close()

    cyl_heading_diff = []
    for i in range(len(data['cyl_angle'])-1):
        # shift to get between 0 and 2 Pi
        old_cyl_ang = (math.radians(data['cyl_angle'][i]) + math.pi)%(2*math.pi)
        new_cyl_ang = (math.radians(data['cyl_angle'][i+1]) + math.pi)%(2*math.pi)
        # find the angle difference
        abs_diff = abs(old_cyl_ang-new_cyl_ang)
        diff = min(abs_diff, 2*math.pi-abs_diff)
        # find the directional difference
        if old_cyl_ang < new_cyl_ang:
            direction = 1 if (new_cyl_ang-old_cyl_ang) <= math.pi else -1
        else:
            direction = -1 if (old_cyl_ang-new_cyl_ang) <= math.pi else 1
        # shift back and apply direction
        diff *= direction
        # normalize between -1 and 1
        cyl_heading_diff.append((diff/math.pi)*100)

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rcParams.update({'font.size': 22})
    plt.plot(data['gsp_heading'], c= 'b')
    plt.plot(cyl_heading_diff, c='r')
    plt.savefig(args.data_path+'/plots/GSP_Heading_'+str(ep)+'.png')
    plt.close()




# if args.second_path is not None:
#     print('. . . Loading Second Model Data')
#     second_data_path = args.second_path + 'Data/'
#     second_file_names = []
#     for file in os.listdir(second_data_path):
#         second_file_names.append(file)

#     second_df_list = []
#     for i in range(len(second_file_names)-1):
#         name = 'Data_Episode_'+str(i)+'.csv'
#         file_path = second_data_path+name
#         df = pd.read_csv(file_path)
#         second_df_list.append((i, df))


# print('. . . Plotting Model Data')
# ep = 0

# for ep in range(len(df_list)):
#     episode = df_list[ep]
#     episode_2 = None
#     if args.second_path is not None:
#         episode_2 = second_df_list[ep]
#     cyl_x_pos = []
#     cyl_y_pos = []
#     cyl_angle = []
#     intention = []
#     cyl_x_pos_2 = []
#     cyl_y_pos_2 = []
#     robot_failures = []
#     robot_x_pos = []
#     robot_y_pos = []
#     pos_gate_x = 0
#     pos_gate_length = 0
#     neg_gate_x = 0
#     neg_gate_length = 0
#     for t in range(len(episode[1])):
#         cyl_x_pos.append(float(episode[1]['cyl_x_pos'][t]))
#         cyl_y_pos.append(float(episode[1]['cyl_y_pos'][t]))
#         if args.failures:
#             robot_failures.append(episode[1]['robot_failures'][t].strip('][').split(','))
#         robot_x_pos.append(episode[1]['robots_x_pos'][t].strip('][').split(','))
#         robot_y_pos.append(episode[1]['robots_y_pos'][t].strip('][').split(','))
#         if args.orientation and not args.intention:
#             cyl_angle.append(episode[1]['cyl_angle'][t])
#         if args.intention:
#             intention.append(episode[1]['intention_heading'][t])
#             cyl_angle.append(episode[1]['cyl_angle'][t])
#     if episode_2 is not None:
#         for t in range(len(episode_2[1])):
#             cyl_x_pos_2.append(episode_2[1]['cyl_x_pos'][t])
#             cyl_y_pos_2.append(episode_2[1]['cyl_y_pos'][t])
#     gate = episode[1]['gate_stats'][0]
#     obstacles = episode[1]['obstacle_stats'][0]

#     if gate != 0:
#         gate = gate.strip('][').split(',')
#     if obstacles != 0:
#         obstacles = obstacles.strip('][').split(',')
#     robot_plot_pos_x = [[] for i in range(len(robot_x_pos[0]))]
#     robot_plot_pos_y = [[] for i in range(len(robot_y_pos[0]))]
#     cylinder_robot_plot_pos_x = []
#     cylinder_robot_plot_pos_y = []
#     if args.plot_robots:
#         for i in range(len(robot_x_pos)):
#             if (i+1)%500==0:
#                 for j in range(len(robot_x_pos[i])):
#                     robot_plot_pos_x[j].append(float(robot_x_pos[i][j]))
#                     robot_plot_pos_y[j].append(float(robot_y_pos[i][j]))
#                 cylinder_robot_plot_pos_x.append(cyl_x_pos[i])
#                 cylinder_robot_plot_pos_y.append(cyl_y_pos[i])
#         for j in range(len(robot_x_pos[0])):
#             robot_plot_pos_x[j].append(float(robot_x_pos[0][j]))
#             robot_plot_pos_y[j].append(float(robot_y_pos[0][j]))
#             robot_plot_pos_x[j].append(float(robot_x_pos[-1][j]))
#             robot_plot_pos_y[j].append(float(robot_y_pos[-1][j]))
               



#     if args.failures:
#         failures = np.zeros(len(robot_failures[0]))
#         failure_x_pos = np.zeros(len(robot_failures[0]))
#         failure_y_pos = np.zeros(len(robot_failures[0]))
#         for i in range(len(robot_failures[0])):
#             for j in range(len(episode[1])):
#                 if int(robot_failures[j][i]) == 1 and failures[i] == 0:
#                     failures[i] = j
#                     failure_x_pos[i] = float(robot_x_pos[j][i])
#                     failure_y_pos[i] = float(robot_y_pos[j][i])

#         robot_failure_x_pos = []
#         robot_failure_y_pos = []
#         for i in range(failure_x_pos.shape[0]):
#             if failure_x_pos[i] != 0.0:
#                 robot_failure_x_pos.append(failure_x_pos[i])
#                 robot_failure_y_pos.append(failure_y_pos[i])

#     fig, ax = plt.subplots(figsize=(20, 10))
#     plt.rcParams.update({'font.size': 22})
#     plt.plot((-10, 10, 10, -10, -10), (-5, -5, 5, 5, -5), c= 'k')


#     if args.heading:
#         headings = []
#         for i in range(len(cyl_x_pos)-1):
#             x0 = cyl_x_pos[i]
#             x1 = cyl_x_pos[i+1]
#             y0 = cyl_y_pos[i]
#             y1 = cyl_y_pos[i+1]
#             h = math.atan2((y1-y0), (x1-x0))
#             headings.append(((x1, x1+math.cos(h)), (y1, y1+math.sin(h))))
#         heading_plot_freq = 3 #seconds
#         heading_index = np.arange(0, len(cyl_x_pos)-1, heading_plot_freq*10)
#         plot_headings = []
#         for i in range(len(heading_index)):
#             plot_headings.append(headings[heading_index[i]])
#     if args.orientation:
#         cyl_angle_freq = 3 #seconds
#         cyl_angle_index = np.arange(0, len(cyl_angle)-1, cyl_angle_freq*10)
#         plot_cyl_angles = []
#         for i in range(len(cyl_angle_index)):
#             x0 = cyl_x_pos[cyl_angle_index[i]]
#             y0 = cyl_y_pos[cyl_angle_index[i]]
#             x1 = x0 + math.cos(math.radians(cyl_angle[cyl_angle_index[i]]))
#             y1 = y0 + math.sin(math.radians(cyl_angle[cyl_angle_index[i]]))
#             plot_cyl_angles.append(((x0, x1), (y0, y1)))
#     if args.intention:
#         intention_freq = 3 #seconds
#         intention_index = np.arange(0, len(intention)-1, intention_freq*10)
#         plot_intention = []
#         for i in range(len(intention_index)):
#             x0 = cyl_x_pos[intention_index[i]]
#             y0 = cyl_y_pos[intention_index[i]]
#             x1 = x0 + math.cos(intention[intention_index[i]]*math.pi+math.radians(cyl_angle[intention_index[i]]))
#             y1 = y0 + math.sin(intention[intention_index[i]]*math.pi+math.radians(cyl_angle[intention_index[i]]))
#             plot_intention.append(((x0, x1), (y0, y1)))

#     if episode_2 is not None:
#         plt.plot(cyl_x_pos, cyl_y_pos, 'r--', linewidth = '5', label = args.label_1)
#         plt.plot(cyl_x_pos_2, cyl_y_pos_2, 'b-.',  linewidth = '5', label = args.label_2)
#     else:
#         plt.plot(cyl_x_pos, cyl_y_pos, 'r', linewidth = '5', label = args.label_1)
#     if args.non_convex:
#         plt.plot((-2, 0, -2), (2, 0, -2), c = 'k', linewidth = 10)
#     #if gate[0] != 0:
#     #    plt.plot((np.float64(gate[0]), np.float64(gate[0])), (-5, (-5 + np.float64(gate[1]))), c='k', linewidth = 5)
#     #    plt.plot((np.float64(gate[0]), np.float64(gate[0])), (5, (5-np.float64(gate[3]))), c='k', linewidth = 5)
#     #    plt.plot((-10, 10), (0, 0), c='r', linestyle='--')
#     if obstacles != 0:
#         for i in range(int(len(obstacles)/2)):
#             ax.add_patch(plt.Circle((np.float64(obstacles[i*2]), np.float64(obstacles[i*2+1])), 0.5, color = 'black'))
#     ax.add_patch(plt.Circle((cyl_x_pos[0], cyl_y_pos[0]), 0.5, facecolor = 'lightgray', edgecolor='black'))
#     ax.add_patch(plt.Circle((cyl_x_pos[-1], cyl_y_pos[-1]), 0.5, facecolor = 'lightgray', edgecolor='black'))
#     if episode_2 is not None:
#         ax.add_patch(plt.Circle((cyl_x_pos_2[-1], cyl_y_pos_2[-1]), 0.5, facecolor = 'lightgray', edgecolor='black'))
#     ax.add_patch(plt.Circle((4.5, 0), 2, color='black', fill = False, linestyle = '--', linewidth=3.0))
#     if args.heading:
#         for i in range(len(plot_headings)):
#             plt.plot(plot_headings[i][0], plot_headings[i][1], c='k')
#     if args.orientation:
#         for i in range(len(plot_cyl_angles)):
#             plt.plot(plot_cyl_angles[i][0], plot_cyl_angles[i][1], c='k')
#     if args.intention:
#         for i in range(len(plot_intention)):
#             plt.plot(plot_intention[i][0], plot_intention[i][1], c='green')
#     if args.plot_robots:
#         colors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#dede00']
#         labels = [f"Robot {i}" for i in range(len(robot_plot_pos_x))]
#         x_pos = [7.5 for i in range(len(robot_plot_pos_x))]
#         y_pos = [4-(i*.5) for i in range(len(robot_plot_pos_x))]
#         for j in range(len(x_pos)):
#             ax.add_patch(plt.Circle((x_pos[j], y_pos[j]), 0.1, facecolor = colors[j], edgecolor='black'))
#             ax.text(x_pos[j]+0.25, y_pos[j]-0.15, labels[j])
#         for i in range(len(robot_plot_pos_x)):
#             for j in range(len(robot_plot_pos_x[i])):
#                 ax.add_patch(plt.Circle((robot_plot_pos_x[i][j], robot_plot_pos_y[i][j]), 0.1, facecolor = colors[i], edgecolor='black'))
#         for i in range(len(cylinder_robot_plot_pos_x)):
#             ax.add_patch(plt.Circle((cylinder_robot_plot_pos_x[i], cylinder_robot_plot_pos_y[i]), 0.5, facecolor = 'lightgray', edgecolor='black'))
#     if args.failures:
#         plt.scatter(robot_failure_x_pos, robot_failure_y_pos, marker='x', c = 'b')
#     plt.text(cyl_x_pos[0]-0.5, cyl_y_pos[0]+1, 'START', fontsize = 20)
#     plt.text(4, -0.25, 'GOAL', fontsize=20)
#     plt.xlim(-11, 11, 1)
#     plt.xticks(np.arange(-11, 12, 1), fontsize = 18)
#     plt.ylim(-6, 6, 1)
#     plt.yticks(np.arange(-6, 7, 1), fontsize = 18)
#     plt.axis('off')
#     if args.label_1 is not None:
#         plt.legend(loc = 'upper right')
#     print('saving', args.figure_path+args.figure_name+'_'+str(ep)+'.png')
#     plt.savefig(args.figure_path+args.figure_name+'_'+str(ep)+'.png')
#     plt.close()
#     ep += 1
