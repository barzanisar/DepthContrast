#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4                     # Request 4 GPUs
#SBATCH --ntasks=2 
#SBATCH --ntasks-per-node=1                 # Number of gpus per node
#SBATCH --time=01:00:00
#SBATCH --job-name=DepthContrast-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=16                  # CPU cores/threads
#SBATCH --mem=400G                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
CFG_FILE=configs/waymo.yaml
TCP_PORT=18888
DOWNSTREAM=false
PRETRAINED_CKPT="default"
LINEAR_PROBE_LAST_N_CKPTS=-1
BATCHSIZE_PER_GPU=-1
EPOCHS=-1
MODEL_NAME="default"


# Additional parameters
DATA_DIR=/home/$USER/scratch/Datasets/Waymo #/home/$USER/projects/rrg-swasland/Datasets/Waymo_short
SING_IMG=/home/$USER/scratch/singularity/ssl_minkunet.sif


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
    -d|--downstream)       # Takes an option argument; ensure it has been specified.
        DOWNSTREAM="true"
        ;;
    -l|--pretrained_ckpt)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_CKPT=$2
            shift
        else
            die 'ERROR: "--pretrained_ckpt" requires a non-empty option argument.'
        fi
        ;;
    -m|--model_name)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            MODEL_NAME=$2
            shift
        else
            die 'ERROR: "--pretrained_ckpt" requires a non-empty option argument.'
        fi
        ;;
    -p|--linear_probe_last_n_ckpts)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            LINEAR_PROBE_LAST_N_CKPTS=$2
            shift
        else
            die 'ERROR: "--linear_probe_last_n_ckpts" requires a non-empty option argument.'
        fi
        ;;
    -b|--batchsize_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            BATCHSIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--batchsize_per_gpu" requires a non-empty option argument.'
        fi
        ;;
    -e|--epochs)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EPOCHS=$2
            shift
        else
            die 'ERROR: "--epochs" requires a non-empty option argument.'
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


export MASTER_ADDR=$(hostname)
export TCP_PORT=$TCP_PORT
export CFG_FILE=$CFG_FILE
export SING_IMG=$SING_IMG
export DATA_DIR=$DATA_DIR
export DOWNSTREAM=$DOWNSTREAM
export PRETRAINED_CKPT=$PRETRAINED_CKPT
export LINEAR_PROBE_LAST_N_CKPTS=$LINEAR_PROBE_LAST_N_CKPTS
export BATCHSIZE_PER_GPU=$BATCHSIZE_PER_GPU
export EPOCHS=$EPOCHS
export MODEL_NAME=$MODEL_NAME

srun scripts/launch_ddp.sh #$MASTER_ADDR $TCP_PORT $CFG_FILE $SING_IMG $DATA_DIR

