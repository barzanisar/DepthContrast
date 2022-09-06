#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=DepthContrast-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=8             # CPU cores/threads
#SBATCH --gres=gpu:t4:1                # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --output=./output/log/%x-%j.out   # STDOUT
#SBATCH --mail-type=ALL
#SBATCH --array=1-3%1   # 4 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
SNOWFALL_RATE_INDEX=-1
SPLIT='None'
FOV=true


# Additional parameters
DATASET=dense
DATA_DIR=/home/$USER/projects/rrg-swasland/Datasets/Dense
INFOS_DIR=/home/$USER/projects/rrg-swasland/Datasets/Dense/Precompute_Infos
SING_IMG=/home/$USER/projects/rrg-swasland/singularity/depth_contrast_snow_sim.sif


# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER --gres=gpu:GPU_ID:NUM_GPUS tools/scripts/${0##*/} [-h]
main.py parameters
[--cfg CFG_FILE]

additional parameters
[--data_dir DATA_DIR]
[--infos_dir INFOS_DIR]
[--sing_img SING_IMG]
[--dist]
[--test_only]

main.py parameters:
--cfg             CFG_FILE           Config file                         [default=$CFG_FILE]

additional parameters:
--data_dir             DATA_DIR           Zipped data directory               [default=$DATA_DIR]
--infos_dir            INFOS_DIR          Infos directory                     [default=$INFOS_DIR]
--sing_img             SING_IMG           Singularity image file              [default=$SING_IMG]
--dist                 DIST               Distributed training flag           [default=$DIST]
--test_only            TEST_ONLY          Test only flag                      [default=$TEST_ONLY]
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
    -s|--split)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SPLIT=$2
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -o|--snowfall_rate_index)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SNOWFALL_RATE_INDEX=$2
            shift
        else
            die 'ERROR: "--tcp_port" requires a non-empty option argument.'
        fi
        ;;
    # Additional parameters
    -d|--data_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATA_DIR=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    -i|--infos_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            INFOS_DIR=$2
            shift
        else
            die 'ERROR: "--infos_dir" requires a non-empty option argument.'
        fi
        ;;
    -s|--sing_img)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SING_IMG=$2
            shift
        else
            die 'ERROR: "--sing_img" requires a non-empty option argument.'
        fi
        ;;
    -f|--fov)       # Takes an option argument; ensure it has been specified.
        FOV="true"
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
FOV=$FOV
SNOWFALL_RATE_INDEX=$SNOWFALL_RATE_INDEX
SPLIT=$SPLIT
"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""

# Extract Dataset
echo "Extracting data"
TMP_DATA_DIR=$SLURM_TMPDIR/data
for file in $DATA_DIR/*.zip; do
    echo "Unzipping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting data"

# Extract dataset infos
echo "Extracting dataset infos"
for file in $INFOS_DIR/*.zip; do
   echo "Unzipping $file to $TMP_DATA_DIR"
   unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting dataset infos"

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
--bind $TMP_DATA_DIR:/DepthContrast/data/$DATASET
--bind $PROJ_DIR/data/$DATASET/ImageSets:/DepthContrast/data/$DATASET/ImageSets
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD
TRAIN_CMD+="python -m lib.LiDAR_snow_sim.tools.snowfall.precompute --split $SPLIT --snowfall_rate_index $SNOWFALL_RATE_INDEX --fov"
echo "Precomputing snow sim on FOV"
echo "$TRAIN_CMD"
eval $TRAIN_CMD
echo "Done Snow Sim FOV"
# if [ $FOV == "true" ]
# then
#     TRAIN_CMD+=" --fov"

#     echo "Precomputing snow sim on FOV"
#     echo "$TRAIN_CMD"
#     eval $TRAIN_CMD
#     echo "Done Snow Sim FOV"
# else

#     echo "Precomputing snow sim 360"
#     echo "$TRAIN_CMD"
#     eval $TRAIN_CMD
#     echo "Done Snow Sim 360"
# fi



