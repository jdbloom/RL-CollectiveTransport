file_name=$(awk '/EXP_NAME/{print $2}' exp_config.yml)
num_obstacles=$(awk '/NUM_OBSTACLES/{print $2}' exp_config.yml)
num_robots=$(awk '/NUM_ROBOTS/{print $2}' exp_config.yml)
max_num_robot_failures=$(awk '/MAX_NUM_ROBOT_FAILURES/{print $2}' exp_config.yml)
chance_failure=$(awk '/CHANCE_FAILURE/{print $2}' exp_config.yml)
num_episodes=$(awk '/NUM_EPISODES/{print $2}' exp_config.yml)
port=$(awk '/PORT/{print $2}' exp_config.yml)
gate=$(awk '/USE_GATE/{print $2}' exp_config.yml)
gate_curriculum=$(awk '/GATE_CURRICULUM/{print $2}' exp_config.yml)
seed=$(awk '/SEED/{print $2}' exp_config.yml)
argos_filename=$(awk '/ARGOS_FILE_NAME/{print $2}' exp_config.yml)
gate_minimum=$(awk '/GATE_MIN/{print $2}' exp_config.yml)
model_num=$(awk '/MODEL_NUM/{print $2}' exp_config.yml)
# gate_curriculum_step=$(awk '/GATE_CURRICULUM_STEP/{print $2}' exp_config.yml)
# gate_curriculum_update_freq=$(awk '/GATE_CURRICULUM_UPDATE_FREQ/{print $2}' exp_config.yml)

cd argos
python generate_argos.py --num_obstacles $num_obstacles --num_robots $num_robots --max_num_robot_failures $max_num_robot_failures --chance_failure $chance_failure --num_episodes $num_episodes --pytorch_port $port --use_gate $gate --gate_curriculum $gate_curriculum --seed $seed --argos_filename $argos_filename --gate_minimum $gate_minimum

cd ..
mkdir rl_code/Data/$file_name/testing_model_$model_num
mkdir rl_code/Data/$file_name/testing_model_$model_num/Data
echo rl_code/Data/$file_name/testing_model_$model_num
cp exp_config.yml rl_code/Data/$file_name/testing_model_$model_num/agent_config.yml
cp argos/$argos_filename rl_code/Data/$file_name/testing_model_$model_num/$argos_filename

argos3 -c argos/$argos_filename &
cd rl_code 
python Main.py Data/$file_name/testing_model_$model_num --test --model_path Data/$file_name/Models/Episode_$model_num
wait $!
cd src/plotting
python viz.py ../../Data/$file_name/testing_model_$model_num/ >> ../../Data/$file_name/testing_model_$model_num/testing_data.txt