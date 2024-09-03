#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1                        # Request 1 GPUs
#SBATCH --ntasks=1                          # set it equal to --nodes
#SBATCH --ntasks-per-node=1                 # Number of gpus per node
#SBATCH --time=12:00:00
#SBATCH --job-name=DepthContrast-train
#SBATCH --cpus-per-task=32                  # CPU cores/threads
#SBATCH --mem=160G                          # 160G also works memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-10%1                       # 3 is the number of jobs in the chain

hostname
nvidia-smi

# die function
die() { echo "$*" 1>&2 ; exit 1; }


# Default Command line args
# main.py script parameters
PRETRAIN_CFG_FILE=configs/waymo.yaml
FINETUNE_CFG_FILE=configs/finetune.yaml
SCRATCH_CFG_FILE=configs/scratch.yaml

MODE=pfsd #pretrain, finetune, scratch, debug
DATASETS=wns
BACKBONE=minkunet

PRETRAINED_CKPT=checkpoint-ep199.pth.tar
PRETRAIN_BATCHSIZE_PER_GPU=32
FINETUNE_BATCHSIZE_PER_GPU=8
PRETRAIN_EPOCHS=200
FINETUNE_EPOCHS=15
FRAME_SAMPLING_DIV=1

MODEL_NAME="default"
PRETRAIN_EXTRA_TAG="try0"
EXTRA_TAG="try0"

SING_IMG=/scratch/ssd004/scratch/$USER/singularity/ssl_proposal.sif
DATA_DIR=/datasets/waymo-processed
# KITTI_DATA_DIR=/raid/datasets/semantic_kitti
# NUSCENES_DATA_DIR=/raid/datasets/nuscenes:/DepthContrast/data/nuscenes/v1.0-trainval


MASTER_ADDR=$(hostname)
TCP_PORT=18888

# Change default data_dir and infos_dir for different datasets

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    # train.py parameters
    -a|--cfg_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAIN_CFG_FILE=$2

            # Get model
            echo "Checking BACKBONE"
            if [[ "$PRETRAIN_CFG_FILE"  == *"minkunet"* ]]; then
                BACKBONE=minkunet
                FINETUNE_CFG_FILE=configs/waymo_fine1lr_minkunet.yaml
                SCRATCH_CFG_FILE=configs/waymo_scratch_minkunet.yaml
                echo "Backbone: minkunet"
            elif [[ "$PRETRAIN_CFG_FILE" == *"pointrcnn"* ]]; then
                BACKBONE=pointrcnn
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
    -b|--mode)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            MODE=$2
            shift
        else
            die 'ERROR: "--mode" requires a non-empty option argument.'
        fi
        ;;
    -c|--datasets)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATASETS=$2
            shift
        else
            die 'ERROR: "--datasets" requires a non-empty option argument.'
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
    -h|--finetune_epochs)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            FINETUNE_EPOCHS=$2
            shift
        else
            die 'ERROR: "--finetune_epochs" requires a non-empty option argument.'
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
    -k|--pretrain_bs_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAIN_BATCHSIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--pretrain_bs_per_gpu" requires a non-empty option argument.'
        fi
        ;;
    -j|--frame_sampling_div)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            FRAME_SAMPLING_DIV=$2
            shift
        else
            die 'ERROR: "--frame_sampling_div" requires a non-empty option argument.'
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
    -n|--workers_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            WORKERS_PER_GPU=$2
            shift
        else
            die 'ERROR: "--workers_per_gpu" requires a non-empty option argument.'
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
module load singularity-ce/3.8.2


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
SINGULARITYENV_WANDB_MODE=online
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
        --multiprocessing-distributed --cfg /DepthContrast/$PRETRAIN_CFG_FILE --world-size $NUM_GPUS 
        --dist-url tcp://$MASTER_ADDR:$TCP_PORT"
    else

        PRETRAIN_CMD+="python /DepthContrast/tools/main_dist.py --cfg /DepthContrast/$PRETRAIN_CFG_FILE"
    fi

    PRETRAIN_CMD+=" --epochs $PRETRAIN_EPOCHS 
    --batchsize_per_gpu $PRETRAIN_BATCHSIZE_PER_GPU 
    --workers $WORKERS_PER_GPU 
    --model_name $MODEL_NAME 
    --extra_tag "$PRETRAIN_EPOCHS"ep_"$PRETRAIN_EXTRA_TAG"
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
        FINETUNE_CMD+="python /DepthContrast/tools/downstream_segmentation.py"
    fi

    FINETUNE_CMD+=" --workers $WORKERS_PER_GPU 
    --epochs $FINETUNE_EPOCHS
    --batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU
    --pretrained_ckpt $PRETRAINED_CKPT 
    --model_name $MODEL_NAME
    --pretrain_extra_tag "$PRETRAIN_EPOCHS"ep_"$PRETRAIN_EXTRA_TAG"
    --extra_tag "$FINETUNE_EPOCHS"ep_"$EXTRA_TAG" 
    --frame_sampling_div $FRAME_SAMPLING_DIV
    "

    
    if [[ "$DATASETS" =~ w ]]; then
        DATASET=waymo

        FINETUNE_CFG_FILE=configs/"$DATASET"_fine1lr_$BACKBONE.yaml
        FINAL_FINETUNE_CMD=$FINETUNE_CMD
        FINAL_FINETUNE_CMD+=" --cfg /DepthContrast/$FINETUNE_CFG_FILE 
        --job_type finetune_"$DATASET"_"$FRAME_SAMPLING_DIV"percent 
        "
        echo "Running Finetuning"
        echo "$FINAL_FINETUNE_CMD"
        eval $FINAL_FINETUNE_CMD
        echo "Done Finetuning"
    fi
    if [[ "$DATASETS" =~ n ]]; then
        DATASET=nuscenes

        FINETUNE_CFG_FILE=configs/"$DATASET"_fine1lr_$BACKBONE.yaml
        FINAL_FINETUNE_CMD=$FINETUNE_CMD
        FINAL_FINETUNE_CMD+=" --cfg /DepthContrast/$FINETUNE_CFG_FILE 
        --job_type finetune_"$DATASET"_"$FRAME_SAMPLING_DIV"percent 
        "
        echo "Running Finetuning"
        echo "$FINAL_FINETUNE_CMD"
        eval $FINAL_FINETUNE_CMD
        echo "Done Finetuning"
    fi
    if [[ "$DATASETS" =~ s ]]; then
        DATASET=semantickitti

        FINETUNE_CFG_FILE=configs/"$DATASET"_fine1lr_$BACKBONE.yaml
        FINAL_FINETUNE_CMD=$FINETUNE_CMD
        FINAL_FINETUNE_CMD+=" --cfg /DepthContrast/$FINETUNE_CFG_FILE 
        --job_type finetune_"$DATASET"_"$FRAME_SAMPLING_DIV"percent 
        "
        echo "Running Finetuning"
        echo "$FINAL_FINETUNE_CMD"
        eval $FINAL_FINETUNE_CMD
        echo "Done Finetuning"
    fi

fi

if [[ "$MODE" =~ s ]]; then
    SCRATCH_CMD=$BASE_CMD
    SCRATCH_CMD+="python -m torch.distributed.launch
    --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
    /DepthContrast/tools/downstream_segmentation.py
    --launcher pytorch
    --multiprocessing-distributed --world-size $NUM_GPUS 
    --dist-url tcp://$MASTER_ADDR:$TCP_PORT 
    --epochs $FINETUNE_EPOCHS
    --batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
    --pretrained_ckpt checkpoint-ep0.pth.tar 
    --workers $WORKERS_PER_GPU 
    --frame_sampling_div $FRAME_SAMPLING_DIV 
    --extra_tag "$FINETUNE_EPOCHS"ep_"$EXTRA_TAG" 
    "
    if [[ "$DATASETS" =~ w ]]; then
        DATASET=waymo

        CFG_FILE=configs/"$DATASET"_scratch_$BACKBONE.yaml
        FINAL_CMD=$SCRATCH_CMD
        FINAL_CMD+=" --cfg /DepthContrast/$CFG_FILE 
        --job_type finetune_"$DATASET"_"$FRAME_SAMPLING_DIV"percent 
        "
        echo "Running Scratch"
        echo "$FINAL_CMD"
        eval $FINAL_CMD
        echo "Done Scratch"
    fi
    if [[ "$DATASETS" =~ n ]]; then
        DATASET=nuscenes

        CFG_FILE=configs/"$DATASET"_scratch_$BACKBONE.yaml
        FINAL_CMD=$SCRATCH_CMD
        FINAL_CMD+=" --cfg /DepthContrast/$CFG_FILE 
        --job_type finetune_"$DATASET"_"$FRAME_SAMPLING_DIV"percent 
        "
        echo "Running Scratch"
        echo "$FINAL_CMD"
        eval $FINAL_CMD
        echo "Done Scratch"
    fi
    if [[ "$DATASETS" =~ s ]]; then
        DATASET=semantickitti

        CFG_FILE=configs/"$DATASET"_scratch_$BACKBONE.yaml
        FINAL_CMD=$SCRATCH_CMD
        FINAL_CMD+=" --cfg /DepthContrast/$CFG_FILE 
        --job_type finetune_"$DATASET"_"$FRAME_SAMPLING_DIV"percent 
        "
        echo "Running Scratch"
        echo "$FINAL_CMD"
        eval $FINAL_CMD
        echo "Done Scratch"
    fi

fi