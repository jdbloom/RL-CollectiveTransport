#!/bin/bash

set -e

num_episodes=1



#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
num_robots_test=4
max_failures=0
experiment_name="DQN_Shared_Prox_2_Obstacles"
recording_path="python_code/Data/Shared_Prox/Training/2_Obstacles/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --share_prox_values &
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


#---------------------------------------------------------------------------
#                  DDQN
learning_scheme=DDQN
num_robots_test=4
max_failures=0
experiment_name="DDQN_Shared_Prox_2_Obstacles"
recording_path="python_code/Data/Shared_Prox/Training/2_Obstacles/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --share_prox_values &
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

#---------------------------------------------------------------------------
#                  DDPG
learning_scheme=DDPG
num_robots_test=4
max_failures=0
experiment_name="DDPG_Shared_Prox_2_Obstacles"
recording_path="python_code/Data/Shared_Prox/Training/2_Obstacles/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --share_prox_values &
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

#---------------------------------------------------------------------------
#                  TD3
learning_scheme=TD3
num_robots_test=4
max_failures=0
experiment_name="TD3_Shared_Prox_2_Obstacles"
recording_path="python_code/Data/Shared_Prox/Training/2_Obstacles/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --share_prox_values &
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
