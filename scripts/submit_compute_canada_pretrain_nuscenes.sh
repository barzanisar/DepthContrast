#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1                     # Request 4 GPUs
#SBATCH --ntasks=1                          # set it equal to --nodes
#SBATCH --ntasks-per-node=1                 # Number of gpus per node
#SBATCH --time=08:00:00
#SBATCH --job-name=DepthContrast-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=32                  # 48 max CPU cores/threads
#SBATCH --mem=200G                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain

hostname
nvidia-smi

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
FINETUNE_CFG_FILE=finetune

MODE=pfd #pretrain, finetune, scratch, debug

PRETRAINED_CKPT=checkpoint-ep99.pth.tar
PRETRAIN_BATCHSIZE_PER_GPU=32
FINETUNE_BATCHSIZE_PER_GPU=8
PRETRAIN_EPOCHS=100
# FINETUNE_EPOCHS=100
# DATA_SKIP_RATIO=1

MODEL_NAME="default"
PRETRAIN_EXTRA_TAG="100ep_try0"
EXTRA_TAG="try0"

SING_IMG=/home/$USER/scratch/singularity/ssl_proposal.sif
# DATA_DIR=/raid/datasets/Waymo
# KITTI_DATA_DIR=/raid/datasets/semantic_kitti
NUSCENES_DATA_DIR=/home/nisarbar/scratch/Datasets/nuscenes:/DepthContrast/data/nuscenes/v1.0-trainval


MASTER_ADDR=$(hostname)
TCP_PORT=18888
FINETUNE_CFG_OPTIONS=""


# Change default data_dir and infos_dir for different datasets

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    # train.py parameters
    -b|--mode)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            MODE=$2
            shift
        else
            die 'ERROR: "--mode" requires a non-empty option argument.'
        fi
        ;;
    -c|--finetune_cfg_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            FINETUNE_CFG_FILE=$2
            shift
        else
            die 'ERROR: "--finetune_cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -d|--pretrained_ckpt)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_CKPT=$2
            shift
        else
            die 'ERROR: "--pretrained_ckpt" requires a non-empty option argument.'
        fi
        ;;
    -e|--model_name)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            MODEL_NAME=$2
            shift
        else
            die 'ERROR: "--model_name" requires a non-empty option argument.'
        fi
        ;;
    -f|--pretrain_extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAIN_EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--pretrain_extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -g|--extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -i|--pretrain_epochs)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAIN_EPOCHS=$2
            shift
        else
            die 'ERROR: "--pretrain_epochs" requires a non-empty option argument.'
        fi
        ;;
    -a|--pretrain_bs_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAIN_BATCHSIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--pretrain_bs_per_gpu" requires a non-empty option argument.'
        fi
        ;;
    -j|--finetune_cfg_options)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            FINETUNE_CFG_OPTIONS=$2
            shift
        else
            die 'ERROR: "--finetune_cfg_options" requires a non-empty option argument.'
        fi
        ;;
    -m|--tcp_port)       # Takes an option argument; ensure it has been specified.
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


# Get last element in string and increment by 1
NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
NUM_GPUS=$(($NUM_GPUS + 1))
WORLD_SIZE=$((NUM_GPUS * SLURM_NNODES))
WORKERS_PER_GPU=$(($SLURM_CPUS_PER_TASK / $NUM_GPUS))


echo "NUM GPUS in Node $SLURM_NODEID: $NUM_GPUS"
echo "Node $SLURM_NODEID says: main node at $MASTER_ADDR:$MASTER_PORT"
echo "Node $SLURM_NODEID says: WORLD_SIZE=$WORLD_SIZE"
echo "Node $SLURM_NODEID says: WORKERS_PER_GPU=$SLURM_CPUS_PER_TASK / $NUM_GPUS=$WORKERS_PER_GPU"
echo "Node $SLURM_NODEID says: Loading Singularity Env..."

# Load Singularity
module load StdEnv/2020 
module load singularity/3.7

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
SINGULARITYENV_WANDB_MODE=offline
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
--bind $NUSCENES_DATA_DIR
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"
if [[ "$MODE" =~ d ]]; then
    CMD="$BASE_CMD bash"
    echo "Debugging..."
    echo "$CMD"
    eval $CMD
fi

if [[ "$MODE" =~ p ]]; then

    PRETRAIN_CMD=$BASE_CMD
    if [ "$NUM_GPUS" -gt 1 ]; then

        PRETRAIN_CMD+="python -m torch.distributed.launch
        --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
        /DepthContrast/tools/main_dist.py
        --launcher pytorch
        --multiprocessing-distributed --cfg /DepthContrast/configs/"$MODEL_NAME".yaml --world-size $NUM_GPUS 
        --dist-url tcp://$MASTER_ADDR:$TCP_PORT"
    else

        PRETRAIN_CMD+="python /DepthContrast/tools/main_dist.py --cfg /DepthContrast/configs/"$MODEL_NAME".yaml"
    fi

    PRETRAIN_CMD+=" --epochs $PRETRAIN_EPOCHS 
        --batchsize_per_gpu $PRETRAIN_BATCHSIZE_PER_GPU 
        --workers $WORKERS_PER_GPU 
        --model_name $MODEL_NAME 
        --extra_tag $PRETRAIN_EXTRA_TAG
        "

    echo "Running Pretraining"
    echo "$PRETRAIN_CMD"
    eval $PRETRAIN_CMD
    echo "Done pretraining"

fi

if [[ "$MODE" =~ f ]]; then

    FINETUNE_CMD=$BASE_CMD

    if [ "$NUM_GPUS" -gt 1 ]; then

        FINETUNE_CMD+="python -m torch.distributed.launch
        --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
        /DepthContrast/tools/downstream_segmentation.py
        --launcher pytorch
        --multiprocessing-distributed --world-size $NUM_GPUS 
        --dist-url tcp://$MASTER_ADDR:$TCP_PORT"
    else
        FINETUNE_CMD+="python /DepthContrast/tools/downstream_segmentation.py --cfg /DepthContrast/configs/"$FINETUNE_CFG_FILE".yaml"
    fi

    FINETUNE_CMD+=" --workers $WORKERS_PER_GPU 
        --pretrained_ckpt $PRETRAINED_CKPT 
        --model_name $MODEL_NAME
        --pretrain_extra_tag $PRETRAIN_EXTRA_TAG
        --extra_tag $EXTRA_TAG
        --wandb_dont_resume
        "
    FINETUNE_CMD+=$FINETUNE_CFG_OPTIONS

    echo "Running Finetuning"
    echo "$FINETUNE_CMD"
    eval $FINETUNE_CMD
    echo "Done Finetuning"

        # --epochs $FINETUNE_EPOCHS
        # --batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU
        # --data_skip_ratio $FRAME_SAMPLING_DIV
        #--job_type finetune_"$DATASET"_"$FRAME_SAMPLING_DIV"percent
        # --val_interval 

fi
