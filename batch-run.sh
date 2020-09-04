#!/bin/bash

set -e

num_robots=4
max_num_robot_failures=1
chance_failure=0.5

until [ $num_robots -gt 16 ]; do
    max_num_robot_failures=0
    until [ $max_num_robot_failures -gt $(($num_robots-2)) ]; do
	experiment_name="num_robots-$num_robots-max_num_robot_failures-$max_num_robot_failures-chance_failure"
	recording_path="python_code/Data/train/$experiment_name"
	figure_path="Data/Figures/$experiment_name.png"
	cd pytorch
	cp -r python_code/Data/template $recording_path
	python pytorch_server.py $recording_path &
	cd ../argos
	python generate_argos.py --num_robots $num_robots --max_num_robot_failures $max_num_robot_failures --chance_failure $chance_failure
	cd ..
	argos3 -c argos/collectiveRlTransport.argos &
	wait
	cd pytorch/python_code/
	python viz.py $recording_path $figure_path
	let max_num_robot_failures=max_num_robot_failures+1
	
    done
    let num_robots=num_robots+1
done
