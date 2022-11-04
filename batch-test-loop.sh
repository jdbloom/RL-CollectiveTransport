#!/bin/bash

set -e

num_episodes=100

seed=123
max_failures=0

# #---------------------------------------------------------------------------
# #                  DQN
# 370 (), 380 (), 410 (), 420 (), ​
# 430 (), 440 ()
# declare -a episodes=(370 380 410 420 430 440)

# for e in "${episodes[@]}"
# do
# learning_scheme=DQN
# num_robots_test=4
# num_obstacles=4
# gate=0
# curriculum=0
# experiment_name="${learning_scheme}_Recurrent_4Obs_Model_${e}"
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/4_obstacles/$experiment_name"
# sleep 1
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/4_obstacles/$experiment_name/Data"
# sleep 1
# recording_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/4_obstacles/$experiment_name"
# model_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/2_obstacles/data/testing/Models/Episode_${e}"
# port="55555"

# echo "$experiment_name Experiment Started"
# cd argos
# echo "Generating Argos File"
# python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
# cd ../pytorch
# echo "Starting Python"
# python Main.py $recording_path --learning_scheme $learning_scheme --intention --recurrent --test --model_path $model_path&
# sleep 5
# cd ..
# echo "Starting Argos"
# argos3 -c argos/collectiveRlTransport.argos &
# wait
# echo "$experiment_name Experiment Finished"
# done

# # #---------------------------------------------------------------------------
# # #                  DDQN Gate
# # 740 (), 970(), 560 (), 580 ()​
# declare -a episodes=(740 970 560 580)

# for e in "${episodes[@]}"
# do
# learning_scheme=DDQN
# num_robots_test=4
# num_obstacles=0
# gate=1
# curriculum=0
# experiment_name="${learning_scheme}_Recurrent_Gate_Model_${e}"
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name"
# sleep 1
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name/Data"
# sleep 1
# recording_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name"
# model_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/Models/Episode_${e}"
# port="55555"

# echo "$experiment_name Experiment Started"
# cd argos
# echo "Generating Argos File"
# python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
# cd ../pytorch
# echo "Starting Python"
# python Main.py $recording_path --learning_scheme $learning_scheme --intention --recurrent --test --model_path $model_path&
# sleep 5
# cd ..
# echo "Starting Argos"
# argos3 -c argos/collectiveRlTransport.argos &
# wait
# echo "$experiment_name Experiment Finished"
# done
# # #---------------------------------------------------------------------------
# # #                  DQN
# # 480 (), 500 ()​
# declare -a episodes=(480 500)

# for e in "${episodes[@]}"
# do
# learning_scheme=DQN
# num_robots_test=4
# num_obstacles=0
# gate=1
# curriculum=0
# experiment_name="${learning_scheme}_Recurrent_Gate_Model_${e}"
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name"
# sleep 1
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name/Data"
# sleep 1
# recording_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name"
# model_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/Models/Episode_${e}"
# port="55555"

# echo "$experiment_name Experiment Started"
# cd argos
# echo "Generating Argos File"
# python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
# cd ../pytorch
# echo "Starting Python"
# python Main.py $recording_path --learning_scheme $learning_scheme --intention --recurrent --test --model_path $model_path&
# sleep 5
# cd ..
# echo "Starting Argos"
# argos3 -c argos/collectiveRlTransport.argos &
# wait
# echo "$experiment_name Experiment Finished"
# done
# # #---------------------------------------------------------------------------
# # # #                  TD3
# # 590 (), 600 (), 620 (), 630 (), ​

# # 640 (), 660 (), 670 (), 680 (), ​

# # 690 (), 960 (), 970 ()
declare -a episodes=(420 440 460 740 750 760 900 910 920)

for e in "${episodes[@]}"
do
learning_scheme=TD3
num_robots_test=4
num_obstacles=0
gate=0
curriculum=0
experiment_name="${learning_scheme}_Global_Model-${e}"

mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/no-obstacle/$experiment_name"
sleep 1
mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/no-obstacle/$experiment_name/Data"
sleep 1
recording_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/no-obstacle/$experiment_name"
model_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/no-obstacle/Models/Episode_${e}"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --global_knowledge --test --model_path $model_path&
sleep 5
cd ..
echo "Starting Argos"
argos3 -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
done
# ##############################################################################################################################
# declare -a episodes=(150 420 340 320 450 540)

# for e in "${episodes[@]}"
# do
# learning_scheme=DDPG
# num_robots_test=4
# num_obstacles=4
# gate=0
# curriculum=0
# experiment_name="${learning_scheme}_Global_Model_4Obs-${e}"

# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/$experiment_name"
# sleep 1
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/$experiment_name/Data"
# sleep 1
# recording_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/$experiment_name"
# model_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/Models/Episode_${e}"
# port="55555"

# echo "$experiment_name Experiment Started"
# cd argos
# echo "Generating Argos File"
# python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
# cd ../pytorch
# echo "Starting Python"
# python Main.py $recording_path --learning_scheme $learning_scheme --global_knowledge --test --model_path $model_path&
# sleep 5
# cd ..
# echo "Starting Argos"
# argos3 -c argos/collectiveRlTransport.argos &
# wait
# echo "$experiment_name Experiment Finished"
# done
# # #---------------------------------------------------------------------------
# # #                  DDPG
# # 440 (), 470 (), 480 (), 490 (),​

# # 500 (), 790 (), 930 (), 330 (), ​

# # 340 ()
# declare -a episodes=(440 470 480 490 500 790 930 330 340)

# for e in "${episodes[@]}"
# do
# learning_scheme=DDPG
# num_robots_test=4
# num_obstacles=0
# gate=1
# curriculum=0
# experiment_name="${learning_scheme}_Recurrent_Gate_Model_${e}"
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name"
# sleep 1
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name/Data"
# sleep 1
# recording_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name"
# model_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/Models/Episode_${e}"
# port="55555"

# echo "$experiment_name Experiment Started"
# cd argos
# echo "Generating Argos File"
# python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port --num_obstacles $num_obstacles --seed $seed --use_gate $gate --gate_curriculum $curriculum
# cd ../pytorch
# echo "Starting Python"
# python Main.py $recording_path --learning_scheme $learning_scheme --intention --recurrent --test --model_path $model_path&
# sleep 5
# cd ..
# echo "Starting Argos"
# argos3 -c argos/collectiveRlTransport.argos &
# wait
# echo "$experiment_name Experiment Finished"
# done

#===========================================Plotting Gate========================================================================================================

# declare -a episodes=(480 500)
# cd pytorch
# cd python_code
# for e in "${episodes[@]}"
# do
# pwd
# learning_scheme=DQN
# num_robots_test=4
# num_obstacles=0
# gate=1
# curriculum=0
# experiment_name="${learning_scheme}_Recurrent_Gate_Model_${e}"
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name/Plots"
# sleep 1
# # recording_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name"
# # model_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/Models/Episode_${e}"
# # port="55555"

# echo "$experiment_name Plotting ...."
# python3 -u viz.py /home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name/ /home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name/Plots/ ${learning_scheme}_recurrent_gate_${e}

# echo "$experiment_name Experiment Finished"
# done

#===========================================Plotting 4 obstacles========================================================================================================
#TD3-2obs (500-33 510-90 520-95 890-93 900-66 910-83)
#DDPG-2obs (150-67 420-22 340-85 320-63 450-87 540-56)
# declare -a episodes=(420 440 460 740 750 760 900 910 920)
# cd pytorch
# cd python_code
# for e in "${episodes[@]}"
# do
# pwd
# learning_scheme=TD3
# num_robots_test=4
# num_obstacles=0
# gate=1
# curriculum=0
# experiment_name="${learning_scheme}_Global_Model-${e}"
# mkdir "/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/no-obstacle/$experiment_name/Plots"
# sleep 1
# # recording_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/$experiment_name"
# # model_path="/home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/recurrent/gate/Models/Episode_${e}"
# # port="55555"

# echo "$experiment_name Plotting ...."
# python3 -u viz.py /home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/no-obstacle/$experiment_name/ /home/endurance/RDDPG/Testing_Data/${learning_scheme}_Intention/Global_No_Intention/gate/$experiment_name/Plots/ ${learning_scheme}_recurrent_4_obstacles_${e}

# echo "$experiment_name Experiment Finished"
# done