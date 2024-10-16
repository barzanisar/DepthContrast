#!/bin/bash

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
SING_IMG=/raid/home/nisarbar/singularity/ssl_cluster.sif
NUSCENES_DATA_DIR=/raid/datasets/nuscenes:/DepthContrast/data/nuscenes/v1.0-trainval

CUDA_VISIBLE_DEVICES=0
NUM_WORKERS=4
START_IDX=0
END_IDX=100
SWEEPS=1
EPS=0.7



# Change default data_dir and infos_dir for different datasets

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    # train.py parameters
    -n|--start_idx)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            START_IDX=$2
            shift
        else
            die 'ERROR: "--start_idx" requires a non-empty option argument.'
        fi
        ;;
    -c|--end_idx)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            END_IDX=$2
            shift
        else
            die 'ERROR: "--end_idx" requires a non-empty option argument.'
        fi
        ;;
    -d|--sweeps)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SWEEPS=$2
            shift
        else
            die 'ERROR: "--sweeps" requires a non-empty option argument.'
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
--bind $NUSCENES_DATA_DIR
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"

CMD=$BASE_CMD
CMD+="python /DepthContrast/tools/cluster_nuscenes.py --start_scene_idx $START_IDX --end_scene_idx $END_IDX --sweeps $SWEEPS --eps $EPS --num_workers $NUM_WORKERS"

echo "$CMD"
eval $CMD
echo "Done clustering"