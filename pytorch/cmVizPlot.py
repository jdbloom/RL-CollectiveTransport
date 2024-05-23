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
parser.add_argument("--Episode")
parser.add_argument("--num_robots", default = 4)

args = parser.parse_args()

episodes = args.Episode

CB_color_cycle = ['#377eb8', '#8e41e0', '#e04161', '#cf6730','#c91818', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#dede00']

print('\n. . . Live Plotting Episode '+args.Episode+' . . .')
if args.source_path is not None:
    print('. . . Loading Model Data')
    data_path = args.source_path
    file_names = []
    file_path = data_path+'Data/Data_Episode_'+str(args.Episode)+'.csv'
    print(file_path)
    data = pd.read_csv(file_path)
print('. . . Plotting Data')

pos_x_cyl = np.asarray(data['cyl_x_cm'])
pos_y_cyl = np.asarray(data['cyl_y_cm'])
pos_mx_cyl = np.asarray(data['cyl_mod_x'])
pos_my_cyl = np.asarray(data['cyl_mod_y'])
pos_ax_cyl = np.asarray(data['cyl_center_x'])
pos_ay_cyl = np.asarray(data['cyl_center_y'])
n_robots = int(args.num_robots)
pos_x_robots = np.empty([len(data['robots_x_pos']),n_robots])
pos_y_robots = np.empty([len(data['robots_y_pos']),n_robots])
pred_x_robots = np.empty([len(data['agent_predictions_x']),n_robots])
pred_y_robots = np.empty([len(data['agent_predictions_y']),n_robots])
vec_robot_force = np.empty([len(data['agent_predictions_y']),n_robots,2])
dis_x_t = np.empty([len(data['agent_predictions_y']),n_robots])
dis_y_t = np.empty([len(data['agent_predictions_y']),n_robots])
avg_forces = np.empty([len(data['agent_predictions_y'])])
std_forces = np.empty([len(data['agent_predictions_y'])])
mag_forces = np.empty([len(data['agent_predictions_y']),n_robots])
# vec_robot_dir = np.empty([len(data['agent_predictions_y']),n_robots])

for i in range(len(pos_x_robots)):
    # print(data['robots_x_pos'][i])
    # print(np.fromstring(data['robots_x_pos'][i].strip('[]'), sep=','))
    # print(data['robots_x_pos'][i].strip('[]'))
    pos_x_robots[i] = np.fromstring(data['robots_x_pos'][i].strip('[]'), sep=',')
    pos_y_robots[i] = np.fromstring(data['robots_y_pos'][i].strip('[]'), sep=',')
    pred_x_robots[i] = np.fromstring(data['agent_predictions_x'][i].strip('[]'), sep=',')
    pred_y_robots[i] = np.fromstring(data['agent_predictions_y'][i].strip('[]'), sep=',')
    ang = np.fromstring(data['force_angle'][i].strip('[]'), sep=',')
    mag = np.fromstring(data['force_magnitude'][i].strip('[]'), sep=',')
    # print("Mag", mag)
    # print("ang", ang)
    vrf_temp = []
    for j in range(n_robots):
        vrf_temp.append(np.array([math.cos(ang[j]), math.sin(ang[j])]) * mag[j])
        # print(vrf_temp[-1], mag[j])
    vec_robot_force[i] = vrf_temp
    avg_forces[i] = np.mean(mag)
    std_forces[i] = np.std(mag)
    mag_forces[i] = mag


std_x_t = np.absolute(pos_x_cyl[:,None] - pred_x_robots)
std_y_t = np.absolute(pos_y_cyl[:,None] - pred_y_robots)
dis_x_t = np.mean(std_x_t, axis=1)
dis_y_t = np.mean(std_y_t, axis=1)
# std_x_t = np.concatenate((np.amax(std_x_t, axis = 1)[:,None], np.amin(std_x_t, axis = 1)[:,None]), axis=1)
# std_y_t = np.concatenate((np.amax(std_y_t, axis = 1)[:,None], np.amin(std_y_t, axis = 1)[:,None]), axis=1)
std_x_t = np.std(std_x_t, axis=1)
std_y_t = np.std(std_y_t, axis=1)

# fig = plt.figure(figsize=(20,10))
fig, axs = plt.subplots(2,2)
axs0 = axs[0,0]
axs1 = axs[0,1]
axs2 = axs[1,0]

plt.rcParams.update({'font.size': 22})

opaqueness = 5
robot_center_op = 0.5
prediction_op = 0.8
decrement = 0.05
visible_interval = 100
opacity_change = int(visible_interval / opaqueness)


cyl = axs0.plot([],[], c=CB_color_cycle[0], linewidth=3)
cylm = axs0.plot([],[], c=CB_color_cycle[1], linewidth=3)
cyla = axs0.plot([],[], c=CB_color_cycle[2], linewidth=3)
pos_r = [[] for i in range(n_robots)]
pos_p = [[] for i in range(n_robots)]
dis_x = axs1.plot([],[], c='r')
dis_y = axs1.plot([],[], c='b')
std_dis_x = [axs1.fill_between([],[],[], color='r', alpha=0.25)]
std_dis_y = [axs1.fill_between([],[],[], color='r', alpha=0.25)]
force_sum = axs2.plot([],[],c='r')
force_std = [axs2.fill_between([],[],[], color='r', alpha = 0.25)]
forces_robots = []
vec_rob_f = []
for i in range(n_robots):
    vec_rob_f.append(axs0.quiver([],[],[],[], color=CB_color_cycle[1]))
    forces_robots.append(axs2.plot([],[],c=CB_color_cycle[i]))
    for j in range(opaqueness):
        pos_r[i].append(axs0.plot([],[], c=CB_color_cycle[(2 + 2*i)%len(CB_color_cycle)], **{'alpha':robot_center_op - j*decrement})[0])
        pos_p[i].append(axs0.plot([],[], c=CB_color_cycle[(2 + 2*i)%len(CB_color_cycle)], **{'alpha':robot_center_op - j*decrement})[0])
        
        # pos_r[i].append(axs0.plot([],[], c=CB_color_cycle[(2 + 2*i)%len(CB_color_cycle)], **{'alpha':robot_center_op - j*decrement, 'marker': 'o'}, markersize=0.1)[0])
        # pos_p[i].append(axs0.plot([],[], c=CB_color_cycle[(2 + 2*i)%len(CB_color_cycle)], **{'alpha':robot_center_op - j*decrement, 'marker': 'o'}, markersize=3)[0])


plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Positions and Predictions in relation to Center of Mass')

# metadata = dict(title='Center of Mass', artist='Chandler Garcia')
# writer = animation.PillowWriter(fps=15, metadata=metadata)

# with writer.saving(fig, "CenterOfMassPrediction.gif", len(pos_x_cyl)):
#     for i in range(len(pos_x_cyl)):


def animate(i):
    cyl[0].set_data(pos_x_cyl[max(i-visible_interval, 0):i],pos_y_cyl[max(i-visible_interval, 0):i])
    cylm[0].set_data(pos_mx_cyl[max(i-visible_interval, 0):i],pos_y_cyl[max(i-visible_interval, 0):i])
    cyla[0].set_data(pos_ax_cyl[max(i-visible_interval, 0):i],pos_y_cyl[max(i-visible_interval, 0):i])
    std_dis_x[0].remove()
    std_dis_y[0].remove()
    dis_x[0].set_data(np.linspace(0,i,i),dis_x_t[:i])
    dis_y[0].set_data(np.linspace(0,i,i),dis_y_t[:i])
    # std_dis_x[0] = axs1.fill_between(np.linspace(0,i,i),std_x_t[:i,0],std_x_t[:i,0], color='r', alpha=0.5)
    # std_dis_y[0] = axs1.fill_between(np.linspace(0,i,i),std_y_t[:i,0],std_y_t[:i,0], color='b', alpha=0.5)
    std_dis_x[0] = axs1.fill_between(np.linspace(0,i,i),dis_x_t[:i]-std_x_t[:i],dis_x_t[:i] + std_x_t[:i], color='r', alpha=0.25)
    std_dis_y[0] = axs1.fill_between(np.linspace(0,i,i),dis_y_t[:i]-std_y_t[:i],dis_y_t[:i] + std_y_t[:i], color='b', alpha=0.25)
    force_sum[0].set_data(np.linspace(0,i,i),avg_forces[:i])
    force_std[0].remove()
    force_std[0] = axs2.fill_between(np.linspace(0,i,i),avg_forces[:i]-std_forces[:i],avg_forces[:i]+std_forces[:i],color='r', alpha = 0.25)
    for j in range(n_robots):
        vec_rob_f[j].remove()
        forces_robots[j][0].set_data(np.linspace(0,i,i),mag_forces[:i,j])
        for k in range(opaqueness):
            start = max(i - (opacity_change * k), 0)
            end = max(i - (opacity_change * (1 + k)), 0)
            pos_r[j][k].set_data(pos_x_robots[end:start,j],pos_y_robots[end:start,j])
            pos_p[j][k].set_data(pred_x_robots[end:start,j],pred_y_robots[end:start,j])
        vec_rob_f[j] = axs0.quiver(pos_x_robots[i,j],pos_y_robots[i,j],*vec_robot_force[i,j], color=CB_color_cycle[4], width=0.005)
        # vec_rob_f[j] = axs0.quiver(pos_x_robots[i,j],pos_y_robots[i,j],*vec_robot_force[i,j], color=CB_color_cycle[4], width=0.005, scale_units='xy', scale=1000)
    return cyl[0], cylm[0],cyla[0], *list(chain.from_iterable(pos_r)), *list(chain.from_iterable(pos_p)), *vec_rob_f, dis_x[0], dis_y[0], std_dis_x[0], std_dis_y[0], force_sum[0], force_std[0], *list(chain.from_iterable(forces_robots))
            
ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, frames = len(pos_x_robots)-1, repeat = False)

axs0.set_xlim([-1,1])
axs0.set_ylim([-0.75,1.25])
axs1.set_xlim([0,len(pos_x_robots)])
axs1.set_ylim([0,0.5])
axs2.set_xlim([0,len(pos_x_robots)])
axs2.set_ylim([0,150])

# writer = animation.PillowWriter(fps=15, metadata=dict(artist='Chandler Garcia'), bitrate=1800)


# ani.save('CM_PLOT_1_'+str(episodes)+'.gif', writer=writer)

plt.show()

plt.close()


# TODO : 
# plot abs x and abs y errors with time
# standard devation accross robots at a single timestep, 
# plot the sum of forces?
# plot these with the reward plot
# 