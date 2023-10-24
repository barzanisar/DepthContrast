#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1                     # Request 4 GPUs
#SBATCH --ntasks=1                          # total number of tasks
#SBATCH --ntasks-per-node=1                 # Number of gpus per node
#SBATCH --time=01:00:00                     # 1 hour
#SBATCH --job-name=cluster_waymo
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=16                  # CPU cores/threads
#SBATCH --mem=64000M                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
SPLIT='train'
PROCESSED_DATA_TAG='waymo_processed_data_v_1_2_0'

# Additional parameters
DATA_DIR=/home/$USER/scratch/Datasets/Waymo
INFOS_DIR=/home/$USER/scratch/Datasets/Waymo

SING_IMG=/home/$USER/scratch/singularity/ssl_cluster_waymo.sif


# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER tools/scripts/${0##*/} [-h]
cluster_waymo.py parameters
[--split SPLIT]
"
}

# Change default data_dir and infos_dir for different datasets

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    # train.py parameters
    -c|--split)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SPLIT=$2
            shift
        else
            die 'ERROR: "--split" requires a non-empty option argument.'
        fi
        ;;
    -p|--processed_data_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PROCESSED_DATA_TAG=$2
            shift
        else
            die 'ERROR: "--processed_data_tag" requires a non-empty option argument.'
        fi
        ;;
    # Additional parameters
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done


echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""


export SPLIT=$SPLIT
export SING_IMG=$SING_IMG
export DATA_DIR=$DATA_DIR
export INFOS_DIR=$INFOS_DIR
export PROCESSED_DATA_TAG=$PROCESSED_DATA_TAG


srun scripts/launch_cluster_waymo.sh #$MASTER_ADDR $TCP_PORT $CFG_FILE $SING_IMG $DATA_DIR

