#!/bin/bash
#SBATCH --nodes 2
#SBATCH --gres=gpu:t4:4                     # Request 4 GPUs
#SBATCH --tasks-per-node=1                  # Number of gpus per node
#SBATCH --time=01:00:00
#SBATCH --job-name=DepthContrast-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=8                   # CPU cores/threads
#SBATCH --mem=64000M                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
CFG_FILE=configs/waymo.yaml
TCP_PORT=18888

# Additional parameters
DATA_DIR=/home/$USER/projects/rrg-swasland/Datasets/Waymo_short
SING_IMG=/home/$USER/projects/rrg-swasland/singularity/ssl.sif
#WANDB_API_KEY=$WANDB_API_KEY
WANDB_MODE='offline'



# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER --gres=gpu:GPU_ID:NUM_GPUS tools/scripts/${0##*/} [-h]
main.py parameters
[--cfg_file CFG_FILE]
[--tcp_port TCP_PORT]
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
    -c|--cfg_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CFG_FILE=$2
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -o|--tcp_port)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TCP_PORT=$2
            shift
        else
            die 'ERROR: "--tcp_port" requires a non-empty option argument.'
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


# Extract Dataset
echo "Extracting Waymo data"
TMP_DATA_DIR=$SLURM_TMPDIR/data

# echo "Unzipping $DATA_DIR/waymo_processed_data_10.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/waymo_processed_data_10.zip -d $TMP_DATA_DIR

# echo "Unzipping $DATA_DIR/Infos/waymo_processed_data_10_infos.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/Infos/waymo_processed_data_10_infos.zip -d $TMP_DATA_DIR

# echo "Unzipping $DATA_DIR/Infos/waymo_infos.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/Infos/waymo_infos.zip -d $TMP_DATA_DIR

echo "Unzipping $DATA_DIR/waymo_processed_data_10_short.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/waymo_processed_data_10_short.zip -d $TMP_DATA_DIR

echo "Unzipping $DATA_DIR/waymo_processed_data_10_short_infos.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/waymo_processed_data_10_short_infos.zip -d $TMP_DATA_DIR

echo "Unzipping $DATA_DIR/waymo_processed_data_10_short_gt_database_train_sampled_1.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/waymo_processed_data_10_short_gt_database_train_sampled_1.zip -d $TMP_DATA_DIR

echo "Done extracting Waymo data"

export MASTER_ADDR=$(hostname)
export TCP_PORT=$TCP_PORT
export CFG_FILE=$CFG_FILE
export TMP_DATA_DIR=$TMP_DATA_DIR
export SING_IMG=$SING_IMG

srun scripts/launch_ddp.sh #$MASTER_ADDR $TCP_PORT $CFG_FILE $TMP_DATA_DIR $SING_IMG

