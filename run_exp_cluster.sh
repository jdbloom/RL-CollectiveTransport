CONFIG_FILE=$1
file_name=$(awk '/EXP_NAME/{print $2}' $CONFIG_FILE)

num_obstacles=$(awk '/NUM_OBSTACLES/{print $2}' $CONFIG_FILE)
num_robots=$(awk '/NUM_ROBOTS/{print $2}' $CONFIG_FILE)
max_num_robot_failures=$(awk '/MAX_NUM_ROBOT_FAILURES/{print $2}' $CONFIG_FILE)
chance_failure=$(awk '/CHANCE_FAILURE/{print $2}' $CONFIG_FILE)
num_episodes=$(awk '/NUM_EPISODES/{print $2}' $CONFIG_FILE)
port=$(awk '/PORT/{print $2}' $CONFIG_FILE)
gate=$(awk '/USE_GATE/{print $2}' $CONFIG_FILE)
gate_curriculum=$(awk '/GATE_CURRICULUM/{print $2}' $CONFIG_FILE)
seed=$(awk '/SEED/{print $2}' $CONFIG_FILE)
argos_filename=$(awk '/ARGOS_FILE_NAME/{print $2}' $CONFIG_FILE)
gate_minimum=$(awk '/GATE_MIN/{print $2}' $CONFIG_FILE)
# gate_curriculum_step=$(awk '/GATE_CURRICULUM_STEP/{print $2}' $CONFIG_FILE)
# gate_curriculum_update_freq=$(awk '/GATE_CURRICULUM_UPDATE_FREQ/{print $2}' $CONFIG_FILE)

cd argos
python generate_argos.py --num_obstacles $num_obstacles --num_robots $num_robots --max_num_robot_failures $max_num_robot_failures --chance_failure $chance_failure --num_episodes $num_episodes --pytorch_port $port --use_gate $gate --gate_curriculum $gate_curriculum --seed $seed --argos_filename $argos_filename --gate_minimum $gate_minimum

tmpdir=$(mktemp -d)

cd ..
mkdir $tmpdir/Data/$file_name
mkdir $tmpdir/Data/$file_name/Data/
mkdir $tmpdir/Data/$file_name/Models/
mkdir $tmpdir/Data/$file_name/plots
cp $CONFIG_FILE $tmpdir/Data/$file_name/agent_config.yml
cp argos/$argos_filename $tmpdir/Data/$file_name/$argos_filename

argos3 -c argos/$argos_filename &
cd rl_code 
python Main.py Data/$file_name
wait $!
cd src/plotting
python viz.py ../../Data/$file_name/ --name $file_name >> ../../Data/$file_name/training_data.txt