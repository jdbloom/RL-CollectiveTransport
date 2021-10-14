#!/bin/bash

set -e

num_episodes=500
seed=124

#---------------------------------------------------------------------------
#                 4 Agent DQN Trained
#---------------------------------------------------------------------------


model_path='python_code/Data/Obstacles/Paper/4_agent_2_obs_DDQN_train_1/Models/Episode_200'
learning_scheme=DDQN
comms_scheme=None



#---------------------------------------------------------------------------
#                  2 Obstacle Test
num_robots_test=4
max_failures=0
num_obstacles=2
experiment_name="4_agent_2_obs_DDQN_2_obs_test_1"
recording_path="python_code/Data/Obstacles/Paper/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

#---------------------------------------------------------------------------
#                  4 Obstacle Test
num_robots_test=4
max_failures=0
num_obstacles=4
experiment_name="4_agent_2_obs_DDQN_4_obs_test_1"
recording_path="python_code/Data/Obstacles/Paper/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
