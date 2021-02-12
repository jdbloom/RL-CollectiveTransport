import argparse
import pandas as pd
import numpy as np
import math
import json
import os
import matplotlib.pyplot as plt

# Base Model Stats over 1000 eps:
base_model_average = -1110.54
base_model_score_std = 93.48
base_model_pct_std = 8.44
base_model_path = 'Data/BaseModel/Data/'

# Base Model with 1 obstacle over 1000 eps:
base_model_1_obstacle_average = -1621.16
base_model_1_obstacle_score_std = 1791.59
base_model_1_obstacle_pct_std = 161.32
base_model_1_obstacle_path = 'Data/BaseModel_1_obstacle/Data/'

# Base Model with 2 obstacle over 1000 eps:
base_model_2_obstacle_average = -1983.07
base_model_2_obstacle_score_std = 2292.86
base_model_2_obstacle_pct_std = 206.46
base_model_2_obstacle_path = 'Data/BaseModel_2_obstacle/Data/'

# Base Model with 3 obstacle over 1000 eps:
base_model_3_obstacle_average = -2478.83
base_model_3_obstacle_score_std = 2747.92
base_model_3_obstacle_pct_std = 247.44
base_model_1_obstacle_path = 'Data/BaseModel_2_obstacle/Data/'

# Base Models with 8 aganets
# Base Model with 0 obstacles over 1000 eps
base_model_8_agents_average = -1106.49
base_model_8_agents_score_std = 92.38
base_model_8_agents_pct_std = 0
base_model_8_agents_path = 'Data/BaseModel_8_agents/Data/'

# Base Model with 2 obstacles over 1000 eps
base_model_8_agents_2_obstacles_average = -2136.96
base_model_8_agents_2_obstacles_score_std = 2491.55
base_model_8_agents_2_obstacles_pct_std = 0
base_model_8_agents_2_obstacles_path = 'Data/BaseModel_8_agents_2_obstacles/Data/'

base_avg = base_model_2_obstacle_average
base_score_std = base_model_2_obstacle_score_std
base_pct_std = base_model_2_obstacle_pct_std
base_path = base_model_2_obstacle_path


parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("figure_path")
parser.add_argument("figure_name")
parser.add_argument("--test", default = False, action = "store_true")
args = parser.parse_args()

data_path = args.data_path + 'Data/'

print('. . . Loading Model Data')
file_names = []
for file in os.listdir(data_path):
    file_names.append(file)

df_list = []
for i in range(len(file_names)-1):
    name = 'Data_Episode_'+str(i)+'.csv'
    file_path = data_path+name
    df = pd.read_csv(file_path)
    df_list.append((i, df))

episode_rewards = []
losses = []
epsilons = []
terminals = 0
print('. . . Consolodating Model Data')
for episode in df_list:
    rewards = []
    terminal = []
    for t in range(len(episode[1])):
        rewards.append(episode[1]['reward'][t].strip('][').split(','))
        #epsilons.append(episode[1]['epsilon'][t].strip('][').split(','))
        #losses.append(episode[1]['loss'][t].strip('][').split(','))
        #terminals += episode[1]['termination'][t]
    reward = []
    for robot in range(len(rewards[0])):
        reward.append(sum(float(row[robot]) for row in rewards))
    episode_rewards.append(reward)
    #print(episode[0], reward[0])

reward = [row[0] for row in episode_rewards]
AVERAGE = np.average(reward)
last_10_axis = np.arange(0, len(reward), 10)
last_10_reward = [sum(reward[i:i + 10])/10
                  for i in last_10_axis[0:len(last_10_axis)-1]]
if args.test:
    print('. . . Loading Baseline Data')
    file_names = []
    for file in os.listdir(base_path):
        file_names.append(file)

    df_list = []
    for i in range(len(file_names)-1):
        name = 'Data_Episode_'+str(i)+'.csv'
        file_path = base_path+name
        df = pd.read_csv(file_path)
        df_list.append((i, df))

    episode_rewards = []
    losses = []
    epsilons = []
    base_terminals = 0
    print('. . . Consolodating Baseline Data')
    for episode in df_list:
        rewards = []
        terminal = []
        for t in range(len(episode[1])):
            rewards.append(episode[1]['reward'][t].strip('][').split(','))
            #epsilons.append(episode[1]['epsilon'][t].strip('][').split(','))
            #losses.append(episode[1]['loss'][t].strip('][').split(','))
            #base_terminals += episode[1]['termination'][t]
        r = []
        for robot in range(len(rewards[0])):
            r.append(sum(float(row[robot]) for row in rewards))
        episode_rewards.append(r)
        #print(episode[0], r[0])

    base_reward = [row[0] for row in episode_rewards]
    base_AVERAGE = np.average(base_reward)
    base_last_10_axis = np.arange(0, len(base_reward), 10)
    base_last_10_reward = [sum(base_reward[i:i + 10])/10
                      for i in base_last_10_axis[0:len(base_last_10_axis)-1]]
    base_fail = 0
    for i in range(len(base_reward)):
        if base_reward[i] < -6500:
            base_fail += 1

episode_fail = 0
episode_struggle = 0
for i in range(len(reward)):
    if reward[i] < -6500:
        episode_fail += 1
    elif reward[i] < -2500:
        episode_struggle += 1

print('. . . Plotting')

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.title(args.figure_name)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.scatter(np.arange(0, len(reward), 1), reward, c = 'darkturquoise', label = 'Episode Scores')
#plt.plot(reward, c = 'lightsteelblue', label = 'Episode Scores')
plt.plot((0, len(reward)), (base_avg, base_avg), c = 'r', label = 'Base Model Average')
plt.plot(last_10_axis[1:len(last_10_axis)], last_10_reward, c = 'b', label = 'Running Average')
plt.ylim(-10000, 500)
plt.legend(loc = 1)
plt.savefig(args.figure_path+args.figure_name+".png")
if args.test:
    plt.clf()
    plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title(args.figure_name)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.scatter(np.arange(0, len(base_reward), 1), base_reward, c = 'lightsalmon', label = 'Baseline Raw Scores')
    plt.plot(base_last_10_axis[1:len(base_last_10_axis)], base_last_10_reward, c = 'r', label = 'Baseline Running Average')
    plt.scatter(np.arange(0, len(reward), 1), reward, c = 'lightsteelblue', label = 'Model Raw Scores')
    plt.plot(last_10_axis[1:len(last_10_axis)], last_10_reward, c = 'b', label = 'Model Running Average')
    plt.ylim(-10000, 500)
    plt.legend(loc = 1)
    plt.savefig(args.figure_path+args.figure_name+"_overlay.png")

    plt.clf()
    score_vs_baseline = []
    for i in range(len(reward)):
        score_vs_baseline.append(reward[i] - base_avg)
    score_std = np.std(score_vs_baseline)
    score_average = np.average(score_vs_baseline)
    last_10_axis = np.arange(0, len(score_vs_baseline), 10)
    last_10_reward = [sum(score_vs_baseline[i:i + 10])/10
              for i in last_10_axis[0:len(last_10_axis)-1]]
    plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title(args.figure_name)
    plt.xlabel('Episodes')
    plt.ylabel('Score vs. Baseline')
    plt.plot(score_vs_baseline, c = 'lightsteelblue')
    plt.plot(last_10_axis[1:len(last_10_axis)], last_10_reward, c = 'b')
    plt.plot([0, len(score_vs_baseline)], [0, 0], c='r')
    plt.plot([0, len(score_vs_baseline)], [base_score_std, base_score_std], c = 'lightsalmon', linestyle = 'dashed')
    plt.plot([0, len(score_vs_baseline)], [-base_score_std, -base_score_std], c = 'lightsalmon', linestyle = 'dashed')
    plt.savefig(args.figure_path+args.figure_name+"_BASE_MODEL_COMPARE_SCORES.png")


    plt.clf()
    pct_vs_baseline = []
    for i in range(len(reward)):
        pct_vs_baseline.append((reward[i] - base_avg)*100/-base_avg)
    pct_std = np.std(pct_vs_baseline)
    pct_average = np.average(pct_vs_baseline)
    last_10_axis = np.arange(0, len(pct_vs_baseline), 10)
    last_10_reward = [sum(pct_vs_baseline[i:i + 10])/10
              for i in last_10_axis[0:len(last_10_axis)-1]]
    plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title(args.figure_name)
    plt.xlabel('Episodes')
    plt.ylabel('% vs. Baseline')
    plt.plot(pct_vs_baseline, c = 'lightsteelblue')
    plt.plot(last_10_axis[1:len(last_10_axis)], last_10_reward, c = 'b')
    plt.plot([0, len(pct_vs_baseline)], [0, 0], c='r')
    plt.plot([0, len(pct_vs_baseline)], [base_pct_std, base_pct_std], c = 'lightsalmon', linestyle = 'dashed')
    plt.plot([0, len(pct_vs_baseline)], [-base_pct_std, -base_pct_std], c = 'lightsalmon', linestyle = 'dashed')
    plt.savefig(args.figure_path+args.figure_name+"_BASE_MODEL_COMPARE_PERCENTAGE.png")

print('Average Reward for Experiment:', AVERAGE)
if not args.test:
    print('Best Model:', (np.argmax(last_10_reward)+1)*10, last_10_reward[np.argmax(last_10_reward)])
print('Score STD:', np.std(reward))
if args.test:
    print('Adjusted Score STD:', score_std)
    print('PCT STD:', pct_std)
print('Failure Episodes:', episode_fail)
print('Struggling Episodes:', episode_struggle)
print('Percentage Failure: %.2f' % (episode_fail*100/len(reward)))
if args.test:
    print('Base Failure Episodes:', base_fail)
    print('Base Percentage Failure: %.2f' % (base_fail*100/len(base_reward)))
print('BASELINE----------  0        1        2        3')
print('Baseline score avg: %.2f %.2f %.2f %.2f' % (base_model_average, base_model_1_obstacle_average, base_model_2_obstacle_average, base_model_3_obstacle_average))
print('Baseline score std:  %.2f    %.2f  %.2f  %.2f' % (base_model_score_std, base_model_1_obstacle_score_std, base_model_2_obstacle_score_std, base_model_3_obstacle_score_std))
print('Baseline pct std:    %.2f     %.2f   %.2f   %.2f' % (base_model_pct_std, base_model_1_obstacle_pct_std, base_model_2_obstacle_pct_std, base_model_3_obstacle_pct_std))
