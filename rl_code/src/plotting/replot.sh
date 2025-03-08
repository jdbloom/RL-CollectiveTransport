#!/bin/bash

# First find all immediate subdirectories under Data
for subdir in ~/Documents/CTRL/Non-Uniform/RL-CollectiveTransport/rl_code/Data/*/; do
    # Then find all testing_ directories within each subdir
    find "$subdir" -type d -name "testing_*" | while read dir; do
        echo "Processing directory: $dir/"
        # Run the python script with the directory path as argument
        python3 make_cylinder_trajectory.py "$dir/"
    done
done