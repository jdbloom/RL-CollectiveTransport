import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

path = 'Data/1_agent_DDQN_1_threshold/Data/'
num_robots = 2

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
    print(episode[0])
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

reward = [row[0] for row in episode_rewards]
last_10_axis = np.arange(0, len(reward), 10)
last_10_reward = [sum(reward[i:i + 10])/10
          for i in last_10_axis[0:len(last_10_axis)-1]]

plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
plt.title('Single Agent Double Deep Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.plot(reward, c = 'lightsteelblue')
plt.plot(last_10_axis[1:len(last_10_axis)], last_10_reward, c = 'b')
plt.savefig('Data/Figures/1_agent_DDQN_1_threshold.png')
