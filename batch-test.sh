#!/bin/bash

set -e

num_episodes=500


#---------------------------------------------------------------------------
#                 8 Agent DQN Trained
#---------------------------------------------------------------------------


model_path='python_code/Data/Failure/8_agent_0_failure_DQN_Train/Models/Episode_800'
learning_scheme=DQN
comms_scheme=None


#---------------------------------------------------------------------------
#                  10 Scalability Test
num_robots_test=10
max_failures=0
experiment_name="8_agent_DQN_trained_10_agent_test"
recording_path="python_code/Data/Scalability/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port --scaled_robots 8&
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"


#---------------------------------------------------------------------------
#                 4 Agent DDQN Trained
#---------------------------------------------------------------------------

model_path='python_code/Data/Failure/4_agent_0_failure_DDQN_Train/Models/Episode_930'
learning_scheme=DDQN
comms_scheme=None


#---------------------------------------------------------------------------
#                  8 Scalability Test
num_robots_test=8
max_failures=0
experiment_name="4_agent_DDQN_trained_8_agent_test"
recording_path="python_code/Data/Scalability/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port --scaled_robots 4 &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

#---------------------------------------------------------------------------
#                  10 Scalability Test
num_robots_test=10
max_failures=0
experiment_name="4_agent_DDQN_trained_10_agent_test"
recording_path="python_code/Data/Scalability/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port --scaled_robots 4 &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"

#---------------------------------------------------------------------------
#                 8 Agent DDQN Trained
#--------------------------------------------------------------------------

model_path='python_code/Data/Scalability/8_agent_DDQN_Train/Models/Episode_680'
learning_scheme=DDQN
comms_scheme=None


#---------------------------------------------------------------------------
#                  10 Scalability Test
num_robots_test=10
max_failures=0
experiment_name="8_agent_DDQN_trained_10_agent_test"
recording_path="python_code/Data/Scalability/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port --scaled_robots 8 &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"





#---------------------------------------------------------------------------
#                 4 Agent DQN 2 Failures Trained
#---------------------------------------------------------------------------
num_robots_test=4
model_path='python_code/Data/Failure/4_agent_2_failure_DQN_Train/Models/Episode_860'
learning_scheme=DQN
comms_scheme=None
#---------------------------------------------------------------------------
#                  1 Failure Test
max_failures=1
experiment_name="4_agent_2_failure_train_DQN_1_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55575"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#                  2 Failure Test
max_failures=2
experiment_name="4_agent_2_failure_train_DQN_2_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55576"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#                  3 Failure Test
max_failures=3
experiment_name="4_agent_2_failure_train_DQN_1_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55577"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#                 8 Agents 0 Failures Trained
num_robots_test=8
model_path='python_code/Data/Failure/8_agent_0_failure_DQN_Train/Models/Episode_800'
learning_scheme=DQN
comms_scheme=None
#---------------------------------------------------------------------------
#                  2 Failure Test
max_failures=2
experiment_name="8_agent_0_failure_train_DQN_2_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55555"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10
#---------------------------------------------------------------------------
#                 4 Failure Test
max_failures=4
experiment_name="8_agent_0_failure_train_DQN_4_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55556"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#                  6 Failure Test
max_failures=6
experiment_name="8_agent_0_failure_train_DQN_6_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55557"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#                 8 Agents 4 Failures Trained
num_robots_test=8
model_path='python_code/Data/Failure/8_agent_4_failure_DQN_Train/Models/Episode_940'
learning_scheme=DQN
comms_scheme=None

#---------------------------------------------------------------------------
#                  0 Failure Test
max_failures=0
experiment_name="8_agent_4_failure_train_DQN_0_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55558"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#                  2 Failure Test
max_failures=2
experiment_name="8_agent_4_failure_train_DQN_2_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55559"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#                 4 Failure Test
max_failures=4
experiment_name="8_agent_4_failure_train_DQN_4_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55560"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#                  6 Failure Test
max_failures=6
experiment_name="8_agent_4_failure_train_DQN_6_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55561"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
sleep 10

#---------------------------------------------------------------------------
#                 4 Agent DDQN 0 Failure Trained
#---------------------------------------------------------------------------

num_robots_test=4
model_path='python_code/Data/Failure/4_agent_0_failure_DDQN_Train/Models/Episode_930'
learning_scheme=DDQN
comms_scheme=None

#---------------------------------------------------------------------------
#                  1 Failure Test
max_failures=1
experiment_name="4_agent_0_failure_train_DDQN_1_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55562"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  2 Failure Test
max_failures=2
experiment_name="4_agent_0_failure_train_DDQN_2_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55563"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  3 Failure Test
max_failures=3
experiment_name="4_agent_0_failure_train_DDQN_3_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55564"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                 4 Agent DDQN 2 Failure Trained
#---------------------------------------------------------------------------

num_robots_test=4
model_path='python_code/Data/Failure/4_agent_2_failure_DDQN_Train/Models/Episode_500'
learning_scheme=DDQN
comms_scheme=None



#---------------------------------------------------------------------------
#                  1 Failure Test
max_failures=1
experiment_name="4_agent_2_failure_train_DDQN_1_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55565"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  2 Failure Test
max_failures=2
experiment_name="4_agent_2_failure_train_DDQN_2_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55566"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  3 Failure Test
max_failures=3
experiment_name="4_agent_2_failure_train_DDQN_3_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55567"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                 8 Agent DDQN 0 Failure Trained
#---------------------------------------------------------------------------

num_robots_test=8
model_path='python_code/Data/Scalability/8_agent_DDQN_Train/Models/Episode_680'
learning_scheme=DDQN
comms_scheme=None


#---------------------------------------------------------------------------
#                  2 Failure Test
max_failures=2
experiment_name="8_agent_0_failure_train_DDQN_2_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55568"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  4 Failure Test
max_failures=4
experiment_name="8_agent_0_failure_train_DDQN_4_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55569"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  6 Failure Test
max_failures=6
experiment_name="8_agent_0_failure_train_DDQN_6_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55570"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                 8 Agent DDQN 4 Failure Trained
#---------------------------------------------------------------------------

num_robots_test=8
model_path='python_code/Data/Failure/8_agent_4_failure_DDQN_Train/Models/Episode_840'
learning_scheme=DDQN
comms_scheme=None


#---------------------------------------------------------------------------
#                  0 Failure Test
max_failures=0
experiment_name="8_agent_4_failure_train_DDQN_0_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55571"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  2 Failure Test
max_failures=2
experiment_name="8_agent_4_failure_train_DDQN_2_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55572"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  4 Failure Test
max_failures=4
experiment_name="8_agent_4_failure_train_DDQN_4_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55573"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
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
#                  6 Failure Test
max_failures=6
experiment_name="8_agent_4_failure_train_DDQN_6_failure_test"
recording_path="python_code/Data/Failure/Test/$experiment_name"
port="55574"

echo "$experiment_name Experiment Started"
cd argos
echo "Generating Argos File"
python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $max_failures --num_episodes $num_episodes --pytorch_port $port
cd ../pytorch
echo "Starting Python"
python Main.py $recording_path --learning_scheme $learning_scheme --comms_scheme $comms_scheme --test --model_path $model_path --no_print --port $port &
sleep 5
cd ..
echo "Starting Argos"
argos3 -z -c argos/collectiveRlTransport.argos &
wait
echo "$experiment_name Experiment Finished"
