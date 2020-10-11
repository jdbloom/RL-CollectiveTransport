#!/bin/bash

set -e

cp pytorch/python_code/Data/train/num_robots-$1-max_num_robot_failures-$2/Models/Model_11_Episode_$3 pytorch/python_code/Data/train/num_robots-$1-max_num_robot_failures-$2/Models/best
