#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                     # Request 4 GPUs
#SBATCH --ntasks=1                          # total number of tasks
#SBATCH --ntasks-per-node=1                 # Number of gpus per node
#SBATCH --time=01:00:00                     # 1 hour
#SBATCH --job-name=cluster_waymo
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=6                  # CPU cores/threads
#SBATCH --mem=64000M                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
SPLIT='train'
PROCESSED_DATA_TAG='waymo_processed_data_v_1_2_0'
FRAME_SAMPLING_INTERVAL=1
EPS=0.2

# Additional parameters
DATA_DIR=/home/$USER/scratch/Datasets/Waymo
INFOS_DIR=/home/$USER/scratch/Datasets/Waymo

SING_IMG=/home/$USER/scratch/singularity/ssl_cluster.sif


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
    -f|--frame_sampling_interval)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            FRAME_SAMPLING_INTERVAL=$2
            shift
        else
            die 'ERROR: "--frame_sampling_interval" requires a non-empty option argument.'
        fi
        ;;
    -e|--eps)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EPS=$2
            shift
        else
            die 'ERROR: "--eps" requires a non-empty option argument.'
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


# # Get last element in string and increment by 1
# NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
# NUM_GPUS=$(($NUM_GPUS + 1))
# WORLD_SIZE=$((NUM_GPUS * SLURM_NNODES))


# echo "NUM GPUS in Node $SLURM_NODEID: $NUM_GPUS"
# echo "Node $SLURM_NODEID says: split $SPLIT"
# echo "Node $SLURM_NODEID says: Loading Singularity Env..."


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

TRAIN_CMD=$BASE_CMD
TRAIN_CMD+="python /DepthContrast/tools/cluster_waymo.py --split $SPLIT --processed_data_tag $PROCESSED_DATA_TAG --frame_sampling_interval $FRAME_SAMPLING_INTERVAL --eps $EPS"

echo "Running training"
echo "$TRAIN_CMD"
eval $TRAIN_CMD
echo "Done training"



