#!/bin/bash

set -e

num_robots=10

until [ $num_robots -gt 16 ]; do
    max_num_robot_failures=0
    until [ $max_num_robot_failures -gt $(($num_robots-2)) ]; do
	experiment_name="num_robots-$num_robots-max_num_robot_failures-$max_num_robot_failures"
	recording_path="python_code/Data/train/$experiment_name"
	figure_path="Data/Figures/$experiment_name.png"
	cd pytorch
	cp -r python_code/Data/template $recording_path
	python pytorch_server.py $recording_path &
	sleep 1
	cd ../argos
	python generate_argos.py --num_robots $num_robots --max_num_robot_failures $max_num_robot_failures
	echo "Generated argos file"
	cd ..
	argos3 -c argos/collectiveRlTransport.argos
	cd pytorch/python_code/
	python viz.py Data/train/$experiment_name/Data/ $figure_path
	let max_num_robot_failures=max_num_robot_failures+1
	
    done
    let num_robots=num_robots+1
done
