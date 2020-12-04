function train_model() {
    # comm_scheme - 1
    # num_robots - 2
    # prop_failures - 3
    # worker_number - 4

    # Computes the maximum number of robot failures from prop_failures and num_robots
    local max_num_robot_failures=0
    # Each worker uses one lower port than the last
    local port_number=55555
    local argos_filename="collectiveRlTransport${4}.argos"
    local experiment_name="comm_scheme-${1}-num_robots-${2}-prop_failures-${3}"
    local recording_path="python_code/Data/train/$experiment_name/"
    local figure_path="pytorch/python_code/Data/Figures/$experiment_name.png"

    # Create the relevant records of the training
    cp -r pytorch/python_code/Data/template pytorch/$recording_path
    # Start the python server
    python pytorch/pytorch_server.py $recording_path --comm_scheme ${1} --port $port_number &
    echo $argos_filename
    # Generate a new argos file
    python argos/generate_argos.py --num_robots ${2} --max_num_robot_failures $max_num_robot_failures --pytorch_port $port_number --argos_filename $argos_filename
    echo "Generated argos file"
    # Start argos
    argos3 -c argos/$argos_filename &

    # Wait on argos and pytorch server to finish
    wait

    # Visualize your creation
    python pytorch/python_code/viz.py Data/train/$experiment_name/Data/ $figure_path
    python pytorch/python_code/viz.py pytorch/$recording_path/Data/ $figure_path
}


train_model "neighbors" 4 0 0
