function train_model() {
    # learning_scheme - 1
    # comm_scheme - 2
    # num_robots - 3
    # prop_failures - 4
    # num_obstacles - 5

    # Computes the maximum number of robot failures from prop_failures and num_robots
    local max_num_robot_failures=0
    # Each worker uses one lower port than the last
    local port_number=55555
    local argos_filename="collectiveRlTransport${4}.argos"
    local experiment_name="${1}_${3}_agent_${5}_obstacles_${4}_failure_${2}_comm_scheme"
    local recording_path="python_code/Data/train/$experiment_name/"
    local figure_path="pytorch/python_code/Data/Figures/"

    # Create the relevant records of the training
    cp -r pytorch/python_code/Data/template pytorch/$recording_path
    # Start the python server
    python pytorch/Main.py $recording_path --learning_scheme ${1} --comm_scheme ${2} --port $port_number &
    echo $argos_filename
    # Generate a new argos file
    python argos/generate_argos.py --num_obstacles ${5} --num_robots ${3} --max_num_robot_failures $max_num_robot_failures --pytorch_port $port_number --argos_filename $argos_filename
    echo "Generated argos file"
    # Start argos
    argos3 -c argos/$argos_filename &

    # Wait on argos and pytorch server to finish
    wait

    # Visualize your creation
    python pytorch/python_code/viz.py Data/train/$experiment_name/ $figure_path $experiment_name
}


train_model "DQN" "None" 4 0 2
