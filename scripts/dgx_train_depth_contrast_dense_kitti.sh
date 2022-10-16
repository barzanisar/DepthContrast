#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=DepthContrast-train
#SBATCH --cpus-per-task=8             # CPU cores/threads
#SBATCH --gres=gpu:2                # Number of GPUs (per node)
#SBATCH --mem=200gb                   # memory per node
#SBATCH --output=./output/log/%x-%j.out   # STDOUT
#SBATCH --mail-type=ALL
#SBATCH --array=1-3%1   # 4 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
CFG_FILE=configs/point_within_lidar_vdc_linear_probe.yaml
CLUSTER="false"
DIST="true"
LINEAR_PROBE="false"
TCP_PORT=18888
LAUNCHER='pytorch'

# Additional parameters
SING_IMG=/opt/singularity/depth_contrast_snow_sim.sif
DIST=true
WANDB_API_KEY=$WANDB_API_KEY
WANDB_MODE='offline' #'dryrun'

# Get last element in string and increment by 1
NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
NUM_GPUS=$(($NUM_GPUS + 1))

# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER --gres=gpu:GPU_ID:NUM_GPUS tools/scripts/${0##*/} [-h]
main.py parameters
[--cfg CFG_FILE]

additional parameters
[--data_dir DENSE_DATA_DIR]
[--infos_dir INFOS_DIR]
[--sing_img SING_IMG]
[--dist]

main.py parameters:
--cfg             CFG_FILE           Config file                         [default=$CFG_FILE]

additional parameters:
--sing_img             SING_IMG           Singularity image file              [default=$SING_IMG]
--dist                 DIST               Distributed training flag           [default=$DIST]
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
    -s|--cluster)       # Takes an option argument; ensure it has been specified.
        CLUSTER="true"
        ;;
    -l|--launcher)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            LAUNCHER=$2
            shift
        else
            die 'ERROR: "--launcher" requires a non-empty option argument.'
        fi
        ;;
    -2|--dist)       # Takes an option argument; ensure it has been specified.
        DIST="true"
        ;;
    -p|--linear_probe)       # Takes an option argument; ensure it has been specified.
        LINEAR_PROBE="true"
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

echo "Running with the following arguments:
main.py parameters:
CFG=$CFG_FILE

Additional parameters
SING_IMG=$SING_IMG
DIST=$DIST
NUM_GPUS=$NUM_GPUS
"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""


# Load Singularity
module load StdEnv/2020 
module load singularity/3.6

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
SINGULARITYENV_WANDB_MODE=$WANDB_MODE
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
--bind $PROJ_DIR/data/dense:/DepthContrast/data/dense
--bind $PROJ_DIR/data/kitti:/DepthContrast/data/kitti
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD
if [ $LINEAR_PROBE == "true" ]
then

    if [ $DIST != "true" ]
    then
        TRAIN_CMD+="python /DepthContrast/tools/linear_probe.py --cfg /DepthContrast/$CFG_FILE
    "
    else
        TRAIN_CMD+="python -m torch.distributed.launch
        --nproc_per_node=$NUM_GPUS
        /DepthContrast/tools/linear_probe.py
        --launcher pytorch
        --tcp_port $TCP_PORT --multiprocessing-distributed --cfg /DepthContrast/$CFG_FILE
        "
    fi

    echo "Running linear probe"
    echo "$TRAIN_CMD"
    eval $TRAIN_CMD
    echo "Done linear probe"
else

    if [ $DIST != "true" ]
    then
        TRAIN_CMD+="python /DepthContrast/tools/main_dist.py --cfg /DepthContrast/$CFG_FILE
    "
    else
        TRAIN_CMD+="python -m torch.distributed.launch
        --nproc_per_node=$NUM_GPUS
        /DepthContrast/tools/main_dist.py
        --launcher pytorch
        --tcp_port $TCP_PORT --multiprocessing-distributed --cfg /DepthContrast/$CFG_FILE
        "
    fi

    echo "Running training"
    echo "$TRAIN_CMD"
    eval $TRAIN_CMD
    echo "Done training"
fi



