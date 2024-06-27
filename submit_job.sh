#!/bin/bash
#SBATCH --output=./runs/slurm_%j_std.out
#SBATCH --error=./runs/slurm_%j_std.err
#SBATCH --partition=short
#SBATCH --mail-user jdbloom@wpi.edu
#SBATCH --mail-type=end
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

config_name=exp_config_$(date +%s).yml
cp exp_config.yml configs/$config_name

srun="srun -N1 -n1 -c ${SLURM_CPUS_PER_TASK}"

my_task="./run_exp_cluster.sh configs/$config_name"

$srun $my_task