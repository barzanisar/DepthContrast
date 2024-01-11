#!/bin/bash
#SBATCH --job-name=test_job    # Job name
#SBATCH --ntasks=1                    # Run on n CPUs
#SBATCH --mem=200gb                     # Job memory request
#SBATCH --time=24-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=./output/%x-%j.log   # Standard output and error log
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

while true
do 
    # print date
    date
    # which GPU - looking for DGX Display
    nvidia-smi -L
    # sleep for 5 days
    sleep 5d
done


