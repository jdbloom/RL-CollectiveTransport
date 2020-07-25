import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_to_json = 'Data/test1'
path_to_Model_Info_json = 'Data/test1/Model_Info/'
path_to_Loss_Info_json = 'Data/test1/Loss_Info/'

# Load all robot data
json_Model_file_names = [pos_json for pos_json in os.listdir(path_to_Model_Info_json) if pos_json.endswith('.json')]
json_Loss_file_names = [pos_json for pos_json in os.listdir(path_to_Loss_Info_json) if pos_json.endswith('.json')]
fig, (ax1, ax2, ax3) = plt.subplots(3)
while True:
    modelData = []
    lossData = []
    for file in json_Model_file_names:
        with open(path_to_Model_Info_json+file) as f:
            modelData.append(json.load(f))
    for file in json_Loss_file_names:
        with open(path_to_Loss_Info_json+file) as f:
            lossData.append(json.load(f))
    epsilons = []
    rewards = []
    losses = []
    for robot in range(len(modelData)):
        epsilon = []
        reward = []
        old_j = 0
        for episode_num in modelData[robot]:
            epsilon.append(modelData[robot][episode_num][0])
            reward.append(modelData[robot][episode_num][1])
        epsilons.append(epsilon)
        rewards.append(reward)
    for robot in range(len(lossData)):
        loss = []
        for episode_num in range(len(lossData[robot])):
            loss.append(lossData[robot][episode_num])
        losses.append(loss)

    fig.suptitle('Single Agent')
    ax1.set_ylabel('Epsilon')
    ax2.set_ylabel('Reward')
    ax3.set_ylabel('Loss')
    for robot in range(len(modelData)):
        ax1.plot(epsilons[robot], label = 'model '+str(robot+1))
        ax2.plot(rewards[robot], label = 'model '+str(robot+1))
    for robot in range(len(lossData)):
        ax3.plot(losses[robot], label = 'model '+str(robot+1))
    ax1.axes.get_xaxis().set_visible(False)
    #plt.xticks(np.arange(0, len(epsilons[0])+1, 50.0))

    ax1.legend()
    plt.xlabel('Episodes/ Time Steps')
    plt.pause(15)
    print('Updating plot')
    plt.savefig(path_to_json+'charts.png')
    ax1.cla()
    ax2.cla()
    ax3.cla()
