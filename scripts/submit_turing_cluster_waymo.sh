#!/bin/bash

# die function
die() { echo "$*" 1>&2 ; exit 1; }

SING_IMG=/raid/home/nisarbar/singularity/ssl_cluster.sif
DATA_DIR=/raid/datasets/Waymo
CUDA_VISIBLE_DEVICES=0

# Default Command line args
# main.py script parameters
SPLIT='train'
PROCESSED_DATA_TAG='waymo_processed_data_v_1_2_0'
FRAME_SAMPLING_INTERVAL=1
EPS=0.2



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



