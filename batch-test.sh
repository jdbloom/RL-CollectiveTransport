#!/bin/bash

set -e

num_episodes=100

seed=123
max_failures=0

# Gate
#---------------------------------------------------------------------------
#                  TD3
learning_scheme=TD3
num_robots_test=4
num_obstacles=0
gate=1
curriculum=0
experiment_name="TD3_Attention_Gate_Model_450"
recording_path="python_code/Data/Attention_Testing/Gate/TD3/$experiment_name"
model_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/Gate/Training/TD3_Gate_Attention/Models/Episode_450"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


# Gate
#---------------------------------------------------------------------------
#                  TD3
learning_scheme=TD3
num_robots_test=4
num_obstacles=0
gate=1
curriculum=0
experiment_name="TD3_Attention_Gate_Model_920"
recording_path="python_code/Data/Attention_Testing/Gate/TD3/$experiment_name"
model_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/Gate/Training/TD3_Gate_Attention/Models/Episode_920"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# Gate
#---------------------------------------------------------------------------
#                  TD3
learning_scheme=TD3
num_robots_test=4
num_obstacles=0
gate=1
curriculum=0
experiment_name="TD3_Attention_Gate_Model_470"
recording_path="python_code/Data/Attention_Testing/Gate/TD3/$experiment_name"
model_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/Gate/Training/TD3_Gate_Attention/Models/Episode_470"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# Gate
#---------------------------------------------------------------------------
#                  TD3
learning_scheme=TD3
num_robots_test=4
num_obstacles=0
gate=1
curriculum=0
experiment_name="TD3_Attention_Gate_Model_490"
recording_path="python_code/Data/Attention_Testing/Gate/TD3/$experiment_name"
model_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/Gate/Training/TD3_Gate_Attention/Models/Episode_490"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


num_episodes=1000
seed=125


# Gate
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="DQN_Gate_Attention"
recording_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/Gate/Training/$experiment_name"
#model_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/2_obstacles/Training/TD3_Attention_2_Obstacles/Models/Episode_940"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# Gate
#---------------------------------------------------------------------------
#                  DDQN
learning_scheme=DDQN
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="DDQN_Gate_Attention"
recording_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/Gate/Training/$experiment_name"
#model_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/2_obstacles/Training/TD3_Attention_2_Obstacles/Models/Episode_940"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# Gate
#---------------------------------------------------------------------------
#                  DDPG
learning_scheme=DDPG
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="DDPG_Gate_Attention"
recording_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/Gate/Training/$experiment_name"
#model_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/2_obstacles/Training/TD3_Attention_2_Obstacles/Models/Episode_940"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --attention&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

