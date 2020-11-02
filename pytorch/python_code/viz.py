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
args = parser.parse_args()

path = args.data_path

file_names = []
for file in os.listdir(path):
    file_names.append(file)

df_list = []
for i in range(len(file_names)-1):
    name = 'Data_Episode_'+str(i)+'.csv'
    file_path = path+name
    df = pd.read_csv(file_path)
    df_list.append((i, df))

episode_rewards = []
losses = []
epsilons = []
terminals = []
for episode in df_list:
    rewards = []
    terminal = []
    for t in range(len(episode[1])):
        rewards.append(episode[1]['reward'][t].strip('][').split(','))
        #epsilons.append(episode[1]['epsilon'][t].strip('][').split(','))
        #losses.append(episode[1]['loss'][t].strip('][').split(','))
        #terminal.append(episode[1]['termination'][t])
    reward = []
    for robot in range(len(rewards[0])):
        reward.append(sum(float(row[robot]) for row in rewards))
    episode_rewards.append(reward)
    print(episode[0], reward[0])

reward = [row[0] for row in episode_rewards]
last_10_axis = np.arange(0, len(reward), 10)
last_10_reward = [sum(reward[i:i + 10])/10
          for i in last_10_axis[0:len(last_10_axis)-1]]
if len(last_10_axis) > 150:
    print('The Best Model is:', 150+np.argmax(last_10_reward[150:]))

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.title('4 Agent Double Deep Q-Learning\n with Obstacle and No Communication')
plt.xlabel('Episodes')
plt.ylabel('Reward')
#s = ['1.9', '1.8', '1.7', '1.6', '1.5', '1.4', '1.3', '1.2', '1.1', '1.0', '0.9', '0.8', '0.7', '0.6', '0.5' ]
#x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
#[250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750]
#y = [-9200, -9200, -9200, -9200, -9200, -9200, -9200, -9200, -9200, -9200, -9200, -9200, -9200, -9200, -9200]
#for i in range(0, min(math.floor(len(reward)/100), len(s))):
#    plt.text(x[i], y[i], s[i], c='gray')
plt.plot(reward, c = 'lightsteelblue')
plt.plot(last_10_axis[1:len(last_10_axis)], last_10_reward, c = 'b')
plt.savefig(args.figure_path)
