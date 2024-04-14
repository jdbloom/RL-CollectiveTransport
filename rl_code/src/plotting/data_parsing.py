import pandas as pd
import numpy as np
import math
import json
import os
import sys
import matplotlib.pyplot as plt

episode = str(sys.argv[1])

path = 'Data/Failure/8_agents_6_failure/Data/'



name = 'Data_Episode_'+episode+'.csv'
file_path = path+name
df = pd.read_csv(file_path)

rewards = []
terminal = []
force_mags = []
force_angs = []
average_force_vec = []
for t in range(len(df)):
    rewards.append(df['reward'][t].strip('][').split(','))
    #epsilons.append(df['epsilon'][t].strip('][').split(','))
    #losses.append(df['loss'][t].strip('][').split(','))
    #terminal.append(df['termination'][t])
    force_mags.append(df['force magnitude'][t].strip('][').split(','))
    force_angs.append(df['force angle'][t].strip('][').split(','))
    average_force_vec.append(df['average force vector'][t].strip('][').split(','))
reward = []
for robot in range(len(rewards[0])):
    reward.append(sum(float(row[robot]) for row in rewards))

robot_forces = []
robot_angles = []
for i in range(len(force_mags[0])):
    forces = []
    angles = []
    for j in range(len(force_mags)):
        forces.append(float(force_mags[j][i]))
        angles.append(float(force_angs[j][i]))
    robot_forces.append(forces)
    robot_angles.append(angles)
mag = []
ang = []
for t in range(len(average_force_vec)):
    mag.append(float(average_force_vec[t][0]))
    ang.append(float(average_force_vec[t][1]))
axis = np.arange(0, len(mag))

#plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
fig, ax = plt.subplots(2, 1)
fig.set_figheight(12)
fig.set_figwidth(20)
fig.suptitle('Episode '+episode+' had Reward: %.2f' % reward[0])
for i in range(len(robot_forces)):
    l = 'Robot '+str(i+1) + ' Force'
    ax[0].plot(robot_forces[i], label = l)
    l = 'Robot '+str(i+1) + ' Heading'
    ax[1].plot(robot_angles[i], label = l)
ax[0].plot(mag, c='k', label = 'Cumulative Force')
ax[1].plot(ang, c = 'k', label = 'Average Heading')
plt.xlabel('Time')
ax[0].set_title('Force over Time for Episode '+episode)
ax[1].set_title('Heading over Time for Episode '+episode)
ax[0].set_ylabel('Force')
ax[1].set_ylabel('Heading (deg)')
ax[1].set_yticks(np.arange(-180, 181, 90))
ax[0].legend()
ax[1].legend()


plt.savefig('Data/Failure/8_agents_6_failure/Figures/8_agents_6_failure_Force_episode_'+episode+'.png')
