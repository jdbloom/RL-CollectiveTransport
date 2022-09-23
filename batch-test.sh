#!/bin/bash

set -e
max_failures=0





num_episodes=100
seed=123



# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=810
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=820
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=870
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=880
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=890
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=900
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=910
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=920
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=930
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=940
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=950
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=960
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=970
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=980
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

# 4 Obstacles
#---------------------------------------------------------------------------
#                  DQN
learning_scheme=DQN
model=990
num_robots_test=4
num_obstacles=4
gate=0
curriculum=0
experiment_name="DQN_4_Obstacles_Intention_Model_$model"
recording_path="python_code/Data/Intention_Testing/4_Obstacles/DQN/$experiment_name"
model_path="python_code/Data/Intention_Training/2_Obstacles/DQN/DQN_2_Obstacles_Intention/Models/Episode_$model"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"



num_episodes=0

seed=125



# Gate
#---------------------------------------------------------------------------
#                  DDPG
learning_scheme=DDPG
num_robots_test=4
num_obstacles=0
gate=1
curriculum=1
experiment_name="DDPG_Gate_Intention"
recording_path="python_code/Data/Intention_Training/Gate/Training/DDPG/$experiment_name"
#model_path="/media/jbloom/91604956-fbc5-4201-b849-19898324f7ff/Attention_Training/Gate/Training/DDPG_Gate_Attention/Models/Episode_780"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --intention&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"







