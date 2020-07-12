import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_to_json = 'Data/test'
path_to_Model_Info_json = 'Data/test/Model_Info/'
path_to_Loss_Info_json = 'Data/test/Loss_Info/'

json_Model_files = [pos_json for pos_json in os.listdir(path_to_Model_Info_json) if pos_json.endswith('.json')]
json_Loss_files = [pos_json for pos_json in os.listdir(path_to_Loss_Info_json) if pos_json.endswith('.json')]
fig, (ax1, ax2, ax3) = plt.subplots(3)
while True:
    modelData = []
    lossData = []
    for file in json_Model_files:
        with open(path_to_Model_Info_json+file) as f:
            modelData.append(json.load(f))
    for file in json_Loss_files:
        with open(path_to_Loss_Info_json+file) as f:
            lossData.append(json.load(f))
    epsilons = []
    rewards = []
    losses = []
    for i in range(len(modelData)):
        epsilon = []
        reward = []
        old_j = 0
        for j in modelData[i]:
            epsilon.append(modelData[i][j][0])
            reward.append(modelData[i][j][1])
        epsilons.append(epsilon)
        rewards.append(reward)
    for i in range(len(lossData)):
        loss = []
        for j in range(len(lossData[i])):
            loss.append(lossData[i][j])
        losses.append(loss)

    fig.suptitle('Single Agent')
    ax1.set_ylabel('Epsilon')
    ax2.set_ylabel('Reward')
    ax3.set_ylabel('Loss')
    for i in range(len(modelData)):
        ax1.plot(epsilons[i], label = 'model '+str(i+1))
        ax2.plot(rewards[i], label = 'model '+str(i+1))
    for i in range(len(lossData)):
        ax3.plot(losses[i], label = 'model '+str(i+1))
    ax1.axes.get_xaxis().set_visible(False)
    #plt.xticks(np.arange(0, len(epsilons[0])+1, 50.0))

    ax1.legend()

    plt.xlabel('Episodes/ Time Steps')
    plt.pause(30)
    print('Updating plot')
    plt.savefig(path_to_json+'charts.png')
    ax1.cla()
    ax2.cla()
    ax3.cla()
