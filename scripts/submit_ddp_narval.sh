#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2                     # Request 4 GPUs
#SBATCH --ntasks=1                          # set it equal to --nodes
#SBATCH --ntasks-per-node=1                 # Number of gpus per node
#SBATCH --time=01:00:00
#SBATCH --job-name=DepthContrast-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=16                  # 48 max CPU cores/threads
#SBATCH --mem=200G                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain

hostname
nvidia-smi

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
PRETRAIN_CFG_FILE=configs/waymo.yaml
LINEARPROBE_CFG_FILE=configs/linear_probe.yaml
FINETUNE_CFG_FILE=configs/finetune.yaml
SCRATCH_CFG_FILE=configs/scratch.yaml

TCP_PORT=18888
MODE=pretrain-finetune #scratch #linearprobe
BACKBONE=minkunet

PRETRAINED_CKPT="default"
LINEARPROBE_LAST_N_CKPTS=-1
PRETRAIN_BATCHSIZE_PER_GPU=16
LINEARPROBE_BATCHSIZE_PER_GPU=-1
FINETUNE_BATCHSIZE_PER_GPU=8
PRETRAIN_EPOCHS=-1
FINETUNE_EPOCHS=-1
LINEARPROBE_EPOCHS=-1
MODEL_NAME="default"
DOWNSTREAM_MODEL_DIR="default"

# Additional parameters
WAYMO_DATA_DIR=/home/$USER/scratch/Datasets/Waymo
NUSCENES_DATA_DIR=/home/$USER/projects/def-swasland-ab/datasets/nuscenes
KITTI_DATA_DIR=/home/$USER/scratch/Datasets/semantic_kitti
SING_IMG=/home/$USER/scratch/singularity/ssl_nuscenes_lidar_aug.sif


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
            PRETRAIN_CFG_FILE=$2

            # Get model
            echo "Checking BACKBONE"
            if [[ "$PRETRAIN_CFG_FILE"  == *"minkunet"* ]]; then
                BACKBONE=minkunet
                LINEARPROBE_CFG_FILE=configs/waymo_lpseg_minkunet.yaml
                FINETUNE_CFG_FILE=configs/waymo_fine1lr_minkunet.yaml
                SCRATCH_CFG_FILE=configs/waymo_scratch_minkunet.yaml
                # LINEARPROBE_CFG_FILE=configs/waymo_lpseg_minkunet_test.yaml
                # FINETUNE_CFG_FILE=configs/waymo_finetune_minkunet_test.yaml
                # SCRATCH_CFG_FILE=configs/waymo_scratch_minkunet_test.yaml
                echo "Backbone: minkunet"
            elif [[ "$PRETRAIN_CFG_FILE" == *"pointrcnn"* ]]; then
                BACKBONE=pointrcnn
                LINEARPROBE_CFG_FILE=configs/waymo_lpseg_pointrcnn.yaml
                FINETUNE_CFG_FILE=configs/waymo_fine1lr_pointrcnn.yaml
                SCRATCH_CFG_FILE=configs/waymo_scratch_pointrcnn.yaml
                echo "Backbone: pointrcnn"
            else
                die 'ERROR: Could not determine backbone from cfg_file path.'
            fi

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
    -d|--mode)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            MODE=$2
            shift
        else
            die 'ERROR: "--mode" requires a non-empty option argument.'
        fi
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
    -w|--downstream_model_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DOWNSTREAM_MODEL_DIR=$2
            shift
        else
            die 'ERROR: "--downstream_model_dir" requires a non-empty option argument.'
        fi
        ;;
    -p|--linearprobe_last_n_ckpts)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            LINEARPROBE_LAST_N_CKPTS=$2
            shift
        else
            die 'ERROR: "--linearprobe_last_n_ckpts" requires a non-empty option argument.'
        fi
        ;;
    -x|--pretrain_batchsize_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAIN_BATCHSIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--pretrain_batchsize_per_gpu" requires a non-empty option argument.'
        fi
        ;;
    -y|--linearprobe_batchsize_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            LINEARPROBE_BATCHSIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--linearprobe_batchsize_per_gpu" requires a non-empty option argument.'
        fi
        ;;
    -z|--finetune_batchsize_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            FINETUNE_BATCHSIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--finetune_batchsize_per_gpu" requires a non-empty option argument.'
        fi
        ;;
    -e|--finetune_epochs)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            FINETUNE_EPOCHS=$2
            shift
        else
            die 'ERROR: "--finetune_epochs" requires a non-empty option argument.'
        fi
        ;;
    -f|--pretrain_epochs)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAIN_EPOCHS=$2
            shift
        else
            die 'ERROR: "--pretrain_epochs" requires a non-empty option argument.'
        fi
        ;;
    -i|--linearprobe_epochs)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            LINEARPROBE_EPOCHS=$2
            shift
        else
            die 'ERROR: "--linearprobe_epochs" requires a non-empty option argument.'
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
export SING_IMG=$SING_IMG
export WAYMO_DATA_DIR=$WAYMO_DATA_DIR

export PRETRAIN_CFG_FILE=$PRETRAIN_CFG_FILE
export LINEARPROBE_CFG_FILE=$LINEARPROBE_CFG_FILE
export FINETUNE_CFG_FILE=$FINETUNE_CFG_FILE
export SCRATCH_CFG_FILE=$SCRATCH_CFG_FILE

export MODE=$MODE

export PRETRAINED_CKPT=$PRETRAINED_CKPT
export LINEARPROBE_LAST_N_CKPTS=$LINEARPROBE_LAST_N_CKPTS

export PRETRAIN_BATCHSIZE_PER_GPU=$PRETRAIN_BATCHSIZE_PER_GPU
export LINEARPROBE_BATCHSIZE_PER_GPU=$LINEARPROBE_BATCHSIZE_PER_GPU
export FINETUNE_BATCHSIZE_PER_GPU=$FINETUNE_BATCHSIZE_PER_GPU

export PRETRAIN_EPOCHS=$PRETRAIN_EPOCHS
export FINETUNE_EPOCHS=$FINETUNE_EPOCHS
export LINEARPROBE_EPOCHS=$LINEARPROBE_EPOCHS

export MODEL_NAME=$MODEL_NAME
export DOWNSTREAM_MODEL_DIR=$DOWNSTREAM_MODEL_DIR

srun scripts/launch_ddp.sh #$MASTER_ADDR $TCP_PORT $CFG_FILE $SING_IMG $DATA_DIR

