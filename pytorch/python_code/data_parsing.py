import pandas as pd
import numpy as np
import math
import json
import os
import matplotlib.pyplot as plt

episode = '35'

path = 'Data/Failure/2_agents_0_failure/Data/'



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
robot_forces = []
for i in range(len(force_mags[0])):
    forces = []
    for j in range(len(force_mags)):
        forces.append(float(force_mags[j][i]))
    robot_forces.append(forces)
mag = []
ang = []
for t in range(len(average_force_vec)):
    mag.append(float(average_force_vec[t][0]))
    ang.append(average_force_vec[t][1])
axis = np.arange(0, len(mag))

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
for i in range(len(robot_forces)):
    l = 'robot '+str(i+1) + 'force'
    plt.plot(robot_forces[i], label = l)
plt.plot(axis, mag, c='k', label = 'Cumulative Force')
plt.xlabel('Time')
plt.title('Force over Time for Episode '+episode)
plt.ylabel('Force')
plt.legend()
plt.savefig('Data/Figures/2_agents_0_failure_Force_episode_'+episode+'.png')
