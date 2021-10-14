#!/bin/bash

set -e

num_robots_test=4
counter=1
until [ $num_robots_test -gt 12 ]; do
    num_robots_model=4
    max_test_failures=$(($num_robots_test*3/4))
    until [ $num_robots_model -gt 12 ]; do
	num_failures_test=0
	max_failures_model=$(($num_robots_model*3/4))
	until [ $num_failures_test -gt $max_test_failures ]; do
	    num_failures_model=0
	    until [ $num_failures_model -gt $max_failures_model ]; do
		if [ $num_robots_model -ne $num_robots_test ] || [ $num_failures_model -ne $num_failures_test ]
		then	       
		    test_name="num_robots_model-$num_robots_model-num_failures_model-$num_failures_model-num_robots_test-$num_robots_test-num_failures_test-$num_failures_test"
		    recording_path="python_code/Data/test/$test_name"
		    figure_path="Data/Figures/$test_name"
		    model_path="python_code/Data/train/num_robots-$num_robots_model-max_num_robot_failures-$num_failures_model/Models/best"
		    cd pytorch
		    cp -r python_code/Data/template $recording_path
		    python pytorch_server.py $recording_path --test --model_path $model_path &
		    sleep 1
		    cd ../argos
		    python generate_argos.py --num_robots $num_robots_test --max_num_robot_failures $num_failures_test --num_episodes 30
		    echo "Generated argos file"
		    cd ..
		    argos3 -c argos/collectiveRlTransport.argos
		    cd pytorch/python_code/
		    python viz.py Data/test/$test_name/Data/ $figure_path
		    cd ../..
		fi
		let num_failures_model+=$num_robots_model/4
	    done
	    let num_failures_test+=$num_robots_test/4
	done
	let num_robots_model+=4
    done
    let num_robots_test+=4
done

	
