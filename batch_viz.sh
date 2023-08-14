#!/bin/bash

set -e

# 2 Obstacles
#---------------------------------------------------------------------------
#                  DQN

learning_scheme=TD3
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="DQN_Gate"
recording_path="Data/MRS/Training/Gate/DQN_2/"
port="55555"


echo "$experiment_name Viz Started"
cd pytorch/python_code
python viz.py $recording_path $recording_path $experiment_name > "$recording_path""scores.txt"


learning_scheme=TD3
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="TD3_Gate"
recording_path="Data/MRS/Training/Gate/TD3_1/"
port="55555"


echo "$experiment_name Viz Started"

python viz.py $recording_path $recording_path $experiment_name > "$recording_path""scores.txt"