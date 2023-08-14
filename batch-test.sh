#!/bin/bash

set -e
max_failures=0

num_episodes=1000
seed=123




# Gate Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="DQN_Gate"
recording_path="python_code/Data/MRS/Training/Gate/DQN"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention --neighbors&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

learning_scheme=DQN
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="DQN_Gate"
recording_path="python_code/Data/MRS/Training/Gate/DQN_2"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention --neighbors&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

learning_scheme=TD3
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="TD3_Gate"
recording_path="python_code/Data/MRS/Training/Gate/TD3_1"
port="55555"


echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention --neighbors&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
