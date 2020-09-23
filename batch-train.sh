#!/bin/bash

set -e

num_robots=12

until [ $num_robots -gt 12 ]; do
    fourths_robot_failures=2
    until [ $fourths_robot_failures -gt 3 ]; do
	# Bash doesn't play nice with fractions/floats
	max_num_robot_failures=$(($fourths_robot_failures*$num_robots/4))
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
	cd ../..
	let fourths_robot_failures+=1
	
    done
    let num_robots=num_robots*2
done
