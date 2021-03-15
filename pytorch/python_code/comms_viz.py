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
parser.add_argument("Episode")
args = parser.parse_args()

data_path = args.data_path + 'Data/'


name = 'Data_Episode_'+args.Episode+'.csv'
file_path = data_path+name
df = pd.read_csv(file_path)
messages = []
for t in range(len(df)):
    messages.append(df['messages'][t].strip('][').split(','))

fig, axs = plt.subplots(2, 2)
ax = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]

time = np.arange(0, len(messages), 1)
for r in range(len(messages[0])):
    robot_messages = []
    msg_1 = 0
    msg_2 = 0
    msg_3 = 0
    msg_4 = 0
    for i in range(len(messages)):
        msg = int(messages[i][r])
        robot_messages.append(msg)
        if msg == 0: msg_1 += 1
        if msg == 1: msg_2 += 1
        if msg == 2: msg_3 += 1
        if msg == 3: msg_4 += 1
    ax[r].plot(time, robot_messages, label = 'robot '+str(r))
    ax[r].legend(loc=1)
    print('------ ROBOT:', r, '--------')
    print('MSG 1 = %0.2f' % (msg_1/len(messages)))
    print('MSG 2 = %0.2f' % (msg_2/len(messages)))
    print('MSG 3 = %0.2f' % (msg_3/len(messages)))
    print('MSG 4 = %0.2f' % (msg_4/len(messages)))
plt.savefig(args.figure_path+'COMMS_CHART_'+args.figure_name)
plt.show()
