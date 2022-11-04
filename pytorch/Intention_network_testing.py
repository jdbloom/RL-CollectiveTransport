from urllib.parse import uses_relative
import python_code.Agent as Agent

import argparse
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import math
import copy
import csv
import os
import time
import torch as T
import matplotlib.pyplot as plt

containing_folder = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--learning_scheme")
parser.add_argument("--test", default = False, action = "store_true")
parser.add_argument("--model_path")
parser.add_argument("--trained_num_robots")                                          # if we are testing a model trained on a different number of robots. This should be set to the training number of robots so that the network is built properly.
parser.add_argument("--no_print", default = False, action = "store_true")
parser.add_argument("--port", default = "55555")
parser.add_argument("--independent_learning", default = False, action = "store_true")
parser.add_argument("--global_knowledge", default = False, action = "store_true")   # append knowledge of other agents to the observation space
parser.add_argument("--intention", default=False, action = "store_true")
parser.add_argument("--recurrent", default= False, action= 'store_true')
parser.add_argument("--attention", default= False, action= 'store_true')
parser.add_argument("--recurrent-rl", default=False, action = 'store_true')
parser.add_argument("--attention-rl", default=False, action="store_true")
parser.add_argument("--meta_param_size", default=0, type=int)
parser.add_argument("--share_prox_values", default=False, action = 'store_true')    # Robots will share their averaged prox values with eachother

args = parser.parse_args()

recording_path = os.path.join(containing_folder, args.recording_path)
if args.model_path is not None:
    model_file_path = os.path.join(containing_folder, args.model_path)
learning_scheme = args.learning_scheme
port = args.port
test_mode = args.test
train_mode = not test_mode

data_file_path = recording_path 


agent_args = {'n_agents':4,
              'n_obs':31, 
              'n_actions': 2,
              'learning_scheme': args.learning_scheme,
              'options_per_action':3,
              'n_chars':1,
              'meta_param_size':1, 
              'use_intention':args.intention, 
              'use_recurrent':args.recurrent,
              'attention':args.attention, 
              'intention_look_back':2}
model = Agent.Agent(id = 0, **agent_args)
model.load_model(model_file_path)


f1 = [(1+math.sin(i))/2.0 for i in np.arange(0, 2*math.pi, 0.01)]
f2 = [(1+math.sin(2*i))/2.0 for i in np.arange(0, 2*math.pi, 0.01)]
f3 = [(1+math.sin(4*i))/2.0 for i in np.arange(0, 2*math.pi, 0.01)]
f4 = [(1+math.sin(8*i))/2.0 for i in np.arange(0, 2*math.pi, 0.01)]
f1_z = np.zeros(len(f1))
f2_z = np.zeros(len(f2))
f3_z = np.zeros(len(f3))
f4_z = np.zeros(len(f4))


f1_intention = [math.degrees(model.choose_object_intention([f1[i], f2_z[i], f3_z[i], f4_z[i]], True)*math.pi) for i in range(len(f1))]
f2_intention = [math.degrees(model.choose_object_intention([f1_z[i], f2[i], f3_z[i], f4_z[i]], True)*math.pi) for i in range(len(f1))]
f3_intention = [math.degrees(model.choose_object_intention([f1_z[i], f2_z[i], f3[i], f4_z[i]], True)*math.pi) for i in range(len(f1))]
f4_intention = [math.degrees(model.choose_object_intention([f1_z[i], f2_z[i], f3_z[i], f4[i]], True)*math.pi) for i in range(len(f1))]

intention = [math.degrees(model.choose_object_intention([f1[i], f2[i], f3[i], f4[i]], True)*math.pi) for i in range(len(f1))]


fig, axs = plt.subplots((2), figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
labels = ["P1", "P2", "P3", "P4", "All"]
#axs[0].set_title('Global State Prediction Network Probe')
axs[0].plot(f1, linewidth=3,label='P1', c='orange')
axs[0].plot(f2, linewidth=3,label = 'P2', c='green')
axs[0].plot(f3, linewidth=3,label = 'P3', c='red')
axs[0].plot(f4, linewidth=3,label = 'P4', c='purple')
axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[0].tick_params(axis='both', which='minor', labelsize=10)
axs[0].set_ylabel('Proximity Readings\n(Avg)', fontsize=30)
axs[0].plot([0, 400],[10, 10] , linewidth=4, label = 'All', c='b')
axs[0].set_ylim(0, 1)
axs[0].set_xticks([])
#axs[0].legend(loc=1,fontsize=20)
#ax2 = ax.twinx()


axs[1].plot(f1_intention, linewidth=3, label = 'P1', c='orange')#, linestyle = (0, (5, 5)))
axs[1].plot(f2_intention, linewidth=3, label = 'P2', c='green')#, linestyle = (0, (5, 5)))
axs[1].plot(f3_intention, linewidth=3, label = 'P3', c='red')#, linestyle = (0, (5, 5)))
axs[1].plot(f4_intention, linewidth=3, label = 'P4', c='purple')#, linestyle = (0, (5, 5)))
axs[1].plot(intention, linewidth=4, label = 'All', c='b')
axs[1].set_ylabel('Predicted Angle Delta\n(Deg)', fontsize=30)
axs[1].plot([0, len(f1)], [0, 0], c='r', linestyle = '--')
axs[1].set_yticks(np.arange(-35, 15, 5))
axs[1].tick_params(axis='both', which='major', labelsize=20)
axs[1].tick_params(axis='both', which='minor', labelsize=10)
#axs[1].legend(loc=1, fontsize=20)
axs[1].set_xticks([])

fig.legend(labels=labels, loc='right', fontsize=30)


plt.savefig(data_file_path + '/function_plot.png')