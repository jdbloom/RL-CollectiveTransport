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

gradients = []
listener_loss =[]
print('. . . Consolodating Model Data')
for episode in df_list:

    for t in range(len(episode[1])):
        listener_loss.append(episode[1]['listener_loss'][t])
        gradients.append(episode[1]['var_grad'][t])


plt.plot(gradients)
plt.savefig(args.figure_path+args.figure_name+'_Variance.png')
plt.clf()
plt.plot(listener_loss)
plt.savefig(args.figure_path+args.figure_name+'_Loss.png')
