#!/bin/bash

# Constants
all_num_robots=(4 6 8 10 12)
all_proportion_failures=(0)

num_workers=${1}
# Array to keep track of worker processes. Value is either idle or a PID
worker_processes=( "idle" )
worker_tasks=( "none" )
tasks=( 0 )
# Expand the array to have an entry per process
count=1
until [ $count -eq $num_workers ]; do
    worker_processes+=( "idle" )
    worker_tasks+=( "none" )
    let count+=1
done

arr_size=${#worker_processes[@]}
echo "Num Workers ${num_workers} : ${arr_size}"

# Check if progress file exists. If it doesn't, make it
if [ ! -f train_progress.csv ]; then
    # Column headers
    echo "num_robots,proportion_failures,status" >> train_progress.csv
    for num_robots in ${all_num_robots[@]}; do
	for proportion_failures in ${all_proportion_failures[@]}; do
	    echo "${num_robots},${proportion_failures},not_started" >> train_progress.csv
	done
    done
fi

function train_model() {
    # num_robots - 1
    # prop_failures - 2
    # worker_number - 3

    # Computes the maximum number of robot failures from prop_failures and num_robots
    local max_num_robot_failures= echo "(${1}*${2})/1" | bc
    # Each worker uses one lower port than the last
    local port_number=$(expr 55555 - ${3})
    local argos_filename="collectiveRlTransport${3}.argos"
    local experiment_name="num_robots-${1}-prop_failures-${2}"
    local recording_path="python_code/Data/train/$experiment_name"
    local figure_path="pytorch/python_code/Data/Figures/$experiment_name.png"
    
    # Create the relevant records of the training
    cp -r pytorch/python_code/Data/template pytorch/$recording_path
    # Start the python server
    python pytorch/pytorch_server.py $recording_path --port $port_number &
    # Generate a new argos file
    python argos/generate_argos.py --num_robots ${1} --max_num_robot_failures ${2} --pytorch_port $port_number --argos_filename $argos_filename
    echo "Generated argos file"
    # Start argos
    argos3 -c argos/$argos_filename &

    # Wait on argos and pytorch server to finish
    wait
    
    # Visualize your creation
    python pytorch/python_code/viz.py Data/train/$experiment_name/Data/ $figure_path
    python pytorch/python_code/viz.py pytorch/$recording_path/Data/ $figure_path
}

# Load progress file into an associative array
declare -A progress_dict
row_count=0
while IFS=, read -r num_robots proportion_failures status
do
    if [ $row_count -gt 0 ]; then
	progress_dict["${num_robots},${proportion_failures}"]=$status
    else
	let row_count+=1
    fi
    
done < train_progress.csv

# Writes progress_dict to train_progress.csv
function update_progress() {
    #echo "Pre update"
    #cat train_progress.csv
    # Write column headers
    echo "num_robots, proportion_failures, status" > train_progress.csv
    while IFS=, read -r num_robots proportion_failures; do
	local_key="${num_robots},${proportion_failures}"
	echo "${local_key},${progress_dict[$local_key]}" >> train_progress.csv	
    done < <(printf "%s\n" "${!progress_dict[@]}")
}

function cleanup() {
    echo "attempting a graceful exit"
    update_progress
    echo "attempting to kill lingering tasks"
    for j in ${!worker_processes[@]}; do
	if [ "${worker_processes[$j]}" != "idle" ]; then
	    # Kill -0 checks if you can signal a process. If we can't signal our own processes then
	    # it should be because the process exited
	    #
	    exit_code=$(kill -0 ${worker_processes[$j]} >> /dev/null >&2 )
	    if [ "$?" -eq 1 ]; then
		progress_dict[${worker_tasks[$j]}]="done"
		update_progress		
	    fi
	fi
    done
    # Kill all dependent processes
    list_descendants ()
    {
	local children=$(ps -o pid= --ppid "$1")
	
	for pid in $children
	do
	    list_descendants "$pid"
	done

	echo "$children"
    }

    kill $(list_descendants $$)
    echo "graceful exit complete"
}

trap cleanup SIGINT

task_counter=0
while IFS=, read -r num_robots proportion_failures status;
do
    
    #for key in "${!progress_dict[@]}"; do
    let task_counter+=1
    # Get value of variables back from string...
    # echo $key | read -r communication_scheme num_robots proportion_failures
    key="${num_robots},${proportion_failures}"
    #status=${progress_dict[$key]}
    # The second check is to disuade weirdness with blank tasks
    if [ "$status" = "not_started" ] && [[ ! -z "$num_robots" ]]; then
	#echo "Task #${task_counter}"
	#echo $key
	task_assigned=0
	any_completed=0
	until [ $task_assigned -eq 1  ]; do
	    #echo "Task assigned ${task_assigned}"
	    # Schedule it to a worker
	    # Find a free worker...
	    echo "Finding worker"
	    for i in ${!worker_processes[@]}; do
		echo "Checking worker #${i}"
		if [ "${worker_processes[$i]}" == "idle" ]; then
		    #echo "Found worker, begining task"
		    train_model ${num_robots} ${proportion_failures} $i &
		    #sleep 3 &
		    # Store PID of task you just started
		    worker_processes[$i]=$!
		    #echo "Input == ${communication_scheme},${num_robots},${proportion_failures}"
		    worker_tasks[$i]="${num_robots},${proportion_failures}"
		    #echo "b"
		    progress_dict[$key]="started"
		    #echo "Task ${worker_tasks[$i]} assigned to worker #${i}, pid of ${worker_processes[$i]}"
		    #update_progress # Write our update to file
		    task_assigned=1
		    break
		fi
	    done
	    
	    #echo "Looking for finished tasks"
	    # Free workers with completed tasks
	    for j in ${!worker_processes[@]}; do
		if [ "${worker_processes[$j]}" != "idle" ]; then
		    # Kill -0 checks if you can signal a process. If we can't signal our own processes then
		    # it should be because the process exited
		    #
		    exit_code=$(kill -0 ${worker_processes[$j]} >> /dev/null >&2 )
		    if [ "$?" -eq 1 ]; then
			echo "Worker ${j} finished their task"
			worker_processes[$j]="idle"
			progress_dict[${worker_tasks[$j]}]="done"
			worker_tasks[$j]="none"
			#update_progress
			any_completed=1
		    fi
		fi
	    done
	    if [ $any_completed -eq 0 ] && [ $task_assigned -eq 0 ]; then
		# If nothing completed recently and there're no free workers, wait a bit
		sleep 1
	    fi	    
	done
    fi    
done < train_progress.csv

echo "-----------------------"


update_progress
# Mop up last few tasks
wait
for j in ${!worker_processes[@]}; do
    if [ "${worker_processes[$j]}" != "idle" ]; then
	# Kill -0 checks if you can signal a process. If we can't signal our own processes then
	# it should be because the process exited
	#
	exit_code=$(kill -0 ${worker_processes[$j]} >> /dev/null >&1 )
	if [ "$?" -eq 1 ]; then
	    echo "Worker ${j} finished their task"
	    worker_processes[$j]="idle"
	    progress_dict[${worker_tasks[$j]}]="done"
	    worker_tasks[$j]="none"
	    any_completed=1
	fi
    fi
done

update_progress
