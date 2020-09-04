#!/bin/bash

set -e

num_robots=1
max_num_robot_failures=1
chance_failure=0.5
recording_path="python_code/Data/train/num_robots-$num_robots-max_num_robot_failures-$max_num_robot_failures-chance_failure-$chance_failure"

cd pytorch
cp -r python_code/Data/template $recording_path
python pytorch_server.py $recording_path &
cd ../argos
python generate_argos.py --num_robots $num_robots --max_num_robot_failures $max_num_robot_failures --chance_failure $chance_failure
cd ..
argos3 -c argos/collectiveRlTransport.argos &
