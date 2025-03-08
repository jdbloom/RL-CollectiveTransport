file_name=$(awk '/EXP_NAME/{print $2}' test_config.yml)
num_obstacles=$(awk '/NUM_OBSTACLES/{print $2}' test_config.yml)
num_robots=$(awk '/NUM_ROBOTS/{print $2}' test_config.yml)
max_num_robot_failures=$(awk '/MAX_NUM_ROBOT_FAILURES/{print $2}' test_config.yml)
chance_failure=$(awk '/CHANCE_FAILURE/{print $2}' test_config.yml)
num_episodes=$(awk '/NUM_EPISODES/{print $2}' test_config.yml)
port=$(awk '/PORT/{print $2}' test_config.yml)
gate=$(awk '/USE_GATE/{print $2}' test_config.yml)
gate_curriculum=$(awk '/GATE_CURRICULUM/{print $2}' test_config.yml)
seed=$(awk '/SEED/{print $2}' test_config.yml)
argos_filename=$(awk '/ARGOS_FILE_NAME/{print $2}' test_config.yml)
gate_minimum=$(awk '/GATE_MIN/{print $2}' test_config.yml)
model_nums=$(awk -F': ' '/MODEL_NUMS/{print $2}' test_config.yml)

# Store original directory
BASE_DIR=$(pwd)
MODEL_DIR=/home/jbloom/Documents/CTRL/RL-CollectiveTransport


# Strip quotes and leading/trailing spaces from model_nums
model_nums=${model_nums#\"}
model_nums=${model_nums%\"}
model_nums=$(echo $model_nums | xargs)

echo "Processing models: $model_nums"

for model_num in $model_nums; do
    test_file_path="testing_model_${model_num}_num_obstacles_${num_obstacles}_gate_${gate}_Non_Uniform"
    
    # Create directories with -p to avoid errors if they exist
    mkdir -p "$MODEL_DIR/rl_code/Data/$file_name/$test_file_path/"{Data,plots}
    
    echo "Created directory: rl_code/Data/$file_name/$test_file_path"
    
    cp test_config.yml "$MODEL_DIR/rl_code/Data/$file_name/$test_file_path/agent_config.yml"

done

for model_num in $model_nums; do
    echo "Starting processing for model: $model_num"
    test_file_path="testing_model_${model_num}_num_obstacles_${num_obstacles}_gate_${gate}_Non_Uniform"
    cd "$BASE_DIR/argos" || exit 1
    python generate_argos.py \
        --num_obstacles "$num_obstacles" \
        --num_robots "$num_robots" \
        --max_num_robot_failures "$max_num_robot_failures" \
        --chance_failure "$chance_failure" \
        --num_episodes "$num_episodes" \
        --pytorch_port "$port" \
        --use_gate "$gate" \
        --gate_curriculum "$gate_curriculum" \
        --seed "$seed" \
        --argos_filename "$argos_filename" \
        --gate_minimum "$gate_minimum"

    cd "$BASE_DIR" || exit 1
    
    cp "argos/$argos_filename" "$MODEL_DIR/rl_code/Data/$file_name/$test_file_path/$argos_filename"

    # Start argos in background and capture PID
    argos3 -c "argos/$argos_filename" &
    ARGOS_PID=$!

    cd "rl_code" || exit 1
    python Main.py "$MODEL_DIR/rl_code/Data/$file_name/$test_file_path" --test --model_path "$MODEL_DIR/rl_code/Data/$file_name/Models/Episode_$model_num"
    
    # Wait for argos to finish
    if ! wait $ARGOS_PID; then
        echo "Warning: argos3 process failed for model $model_num"
    fi

    cd "src/plotting" || exit 1
    python viz.py "$MODEL_DIR/rl_code/Data/$file_name/$test_file_path/" >> "$MODEL_DIR/rl_code/Data/$file_name/$test_file_path/testing_data.txt"
    python make_cylinder_trajectory.py "$MODEL_DIR/rl_code/Data/$file_name/$test_file_path/"
    
    # Return to base directory
    cd "$BASE_DIR" || exit 1

    echo "Completed processing for model: $model_num"
done