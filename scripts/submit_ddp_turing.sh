#!/bin/bash

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

SING_IMG=/raid/home/nisarbar/singularity/ssl_lidar_aug.sif
DATA_DIR=/raid/datasets/Waymo
NUM_GPUS=2
CUDA_VISIBLE_DEVICES=0,1
MASTER_ADDR=Turing
WORKERS_PER_GPU=8 # Turing has 48 cpus so use 10 cpus/gpu



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
    -n|--num_gpus)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_GPUS=$2
            shift
        else
            die 'ERROR: "--num_gpus" requires a non-empty option argument.'
        fi
        ;;
    -c|--cuda_visible_devices)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CUDA_VISIBLE_DEVICES=$2
            shift
        else
            die 'ERROR: "--cuda_visible_devices" requires a non-empty option argument.'
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


PROJ_DIR=$PWD
DEPTH_CONTRAST_BINDS=""
for entry in $PROJ_DIR/third_party/OpenPCDet/pcdet/*
do
    name=$(basename $entry)
    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        DEPTH_CONTRAST_BINDS+="--bind $entry:/DepthContrast/third_party/OpenPCDet/pcdet/$name
"
    fi
done

# Extra binds
DEPTH_CONTRAST_BINDS+="
    --bind $PROJ_DIR/third_party/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py:/DepthContrast/third_party/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py
    --bind $PROJ_DIR/third_party/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py:/DepthContrast/third_party/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py
"

BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
SINGULARITYENV_NCCL_BLOCKING_WAIT=1
singularity exec
--nv
--pwd /DepthContrast
--bind $PROJ_DIR/checkpoints:/DepthContrast/checkpoints
--bind $PROJ_DIR/configs:/DepthContrast/configs
--bind $PROJ_DIR/criterions:/DepthContrast/criterions
--bind $PROJ_DIR/output:/DepthContrast/output
--bind $PROJ_DIR/datasets:/DepthContrast/datasets
--bind $PROJ_DIR/tools:/DepthContrast/tools
--bind $PROJ_DIR/models:/DepthContrast/models
--bind $PROJ_DIR/scripts:/DepthContrast/scripts
--bind $PROJ_DIR/utils:/DepthContrast/utils
--bind $DATA_DIR:/DepthContrast/data/waymo
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"

PRETRAIN_CMD=$BASE_CMD
PRETRAIN_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/main_dist.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$PRETRAIN_CFG_FILE --world-size $NUM_GPUS 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $PRETRAIN_EPOCHS 
--batchsize_per_gpu $PRETRAIN_BATCHSIZE_PER_GPU 
--workers $WORKERS_PER_GPU
"


FINETUNE_CMD=$BASE_CMD

FINETUNE_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/downstream_segmentation.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$FINETUNE_CFG_FILE --world-size $NUM_GPUS 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $FINETUNE_EPOCHS
--batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
--downstream_model_dir $DOWNSTREAM_MODEL_DIR
--model_name $MODEL_NAME
--pretrained_ckpt $PRETRAINED_CKPT 
--workers $WORKERS_PER_GPU
"

SCRATCH_CMD=$BASE_CMD

SCRATCH_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/downstream_segmentation.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$SCRATCH_CFG_FILE --world-size $NUM_GPUS 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $FINETUNE_EPOCHS
--batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
--downstream_model_dir $DOWNSTREAM_MODEL_DIR
--pretrained_ckpt $PRETRAINED_CKPT 
--workers $WORKERS_PER_GPU
"

LINEARPROBE_CMD=$BASE_CMD

LINEARPROBE_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/downstream_segmentation.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$LINEARPROBE_CFG_FILE --world-size $NUM_GPUS 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $LINEARPROBE_EPOCHS
--batchsize_per_gpu $LINEARPROBE_BATCHSIZE_PER_GPU 
--downstream_model_dir $DOWNSTREAM_MODEL_DIR
--model_name $MODEL_NAME
--linear_probe_last_n_ckpts $LINEARPROBE_LAST_N_CKPTS 
--workers $WORKERS_PER_GPU
"

if [[ "$MODE" == "pretrain-finetune" ]]; then
    echo "Running Pretraining"
    echo "$PRETRAIN_CMD"
    eval $PRETRAIN_CMD
    echo "Done pretraining"

    echo "Running Finetuning"
    echo "$FINETUNE_CMD"
    eval $FINETUNE_CMD
    echo "Done Finetuning"

elif [[ "$MODE" == "linearprobe" ]]; then
    echo "Running Linear Probe Only"
    echo "$LINEARPROBE_CMD"
    eval $LINEARPROBE_CMD
    echo "Done linear probe"

elif [[ "$MODE" == "scratch" ]]; then
    echo "Running Scratch training"
    echo "$SCRATCH_CMD"
    eval $SCRATCH_CMD
    echo "Done scratch training"
    
fi


