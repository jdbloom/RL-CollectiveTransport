#!/bin/bash

set -e

num_episodes=1


#---------------------------------------------------------------------------
#                 4 Agent DQN
#---------------------------------------------------------------------------



learning_scheme=DQN
comms_scheme=None


#---------------------------------------------------------------------------
#                  No Intention
num_robots_test=4
max_failures=0
experiment_name="DQN_4_A_2_OB_NO_INTENTION"
recording_path="python_code/Data/Intention/CTDE/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


#---------------------------------------------------------------------------
#                  Intention
num_robots_test=4
max_failures=0
experiment_name="DQN_4_A_2_OB_INTENTION"
recording_path="python_code/Data/Intention/CTDE/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --use_intention &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
