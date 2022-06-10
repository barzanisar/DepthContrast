#!/bin/bash

# PARAMETERS
#KITTI_TRAIN=$(readlink -f ../data/kitti/training)
#KITTI_TEST=$(readlink -f ../data/kitti/testing)
#WAYMO_RAW=$(readlink -f ../data/waymo/raw_data)
WAYMO_PROCESSED=$(readlink -f ./data/waymo/waymo_processed_data_10)

# Setup volume linking (host link:container link)
CUR_DIR=$(pwd)
PROJ_DIR=$CUR_DIR
#KITTI_TRAIN=$KITTI_TRAIN:/DepthContrast/data/kitti/training
#KITTI_TEST=$KITTI_TEST:/DepthContrast/data/kitti/testing
#WAYMO_RAW=$WAYMO_RAW:/DepthContrast/data/waymo/raw_data
WAYMO_PROCESSED=$WAYMO_PROCESSED:/DepthContrast/data/waymo/waymo_processed_data_10

PCDET_VOLUMES=""
for entry in $PROJ_DIR/third_party/OpenPCDet/pcdet/*
do
    name=$(basename $entry)

    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        PCDET_VOLUMES+="--volume $entry:/DepthContrast/third_party/OpenPCDet/pcdet/$name "
    fi
done

docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$XAUTHORITY:/root/.Xauthority:rw" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --hostname="inside-DOCKER" \
        --name="DepthContrast" \
        --volume $WAYMO_PROCESSED \
        --volume $PROJ_DIR/data:/DepthContrast/data \
        --volume $PROJ_DIR/output:/DepthContrast/output \
        --volume $PROJ_DIR/tools:/DepthContrast/tools \
        --volume $PROJ_DIR/checkpoints:/DepthContrast/checkpoints \
        --volume $PROJ_DIR/configs:/DepthContrast/configs \
        --volume $PROJ_DIR/criterions:/DepthContrast/criterions \
        --volume $PROJ_DIR/datasets:/DepthContrast/datasets \
        --volume $PROJ_DIR/models:/DepthContrast/models \
        --volume $PROJ_DIR/scripts:/DepthContrast/scripts \
        --volume $PROJ_DIR/utils:/DepthContrast/utils \
        $PCDET_VOLUMES \
        --rm \
        depth_contrast_img:first_try bash