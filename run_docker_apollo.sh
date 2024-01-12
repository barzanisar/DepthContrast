#!/bin/bash

CUR_DIR=$(pwd)
PROJ_DIR=$CUR_DIR
DATA_DIR=/home/bnisar/datasets/Waymo


PCDET_VOLUMES=""
for entry in $PROJ_DIR/third_party/OpenPCDet/pcdet/*
do
    name=$(basename $entry)

    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        PCDET_VOLUMES+="--volume $entry:/DepthContrast/third_party/OpenPCDet/pcdet/$name "
    fi
done

echo "$PCDET_VOLUMES"
# Extra binds
PCDET_VOLUMES+="
    --volume $PROJ_DIR/third_party/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py:/DepthContrast/third_party/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py
    --volume $PROJ_DIR/third_party/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py:/DepthContrast/third_party/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py
"

docker run -it --env="WANDB_API_KEY=$WANDB_API_KEY" \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --hostname="inside-DOCKER" \
        --name="DepthContrast" \
        --volume $DATA_DIR/data:/DepthContrast/data \
        --volume $PROJ_DIR/output:/DepthContrast/output \
        --volume $PROJ_DIR/tools:/DepthContrast/tools \
        --volume $PROJ_DIR/checkpoints:/DepthContrast/checkpoints \
        --volume $PROJ_DIR/configs:/DepthContrast/configs \
        --volume $PROJ_DIR/criterions:/DepthContrast/criterions \
        --volume $PROJ_DIR/datasets:/DepthContrast/datasets \
        --volume $PROJ_DIR/models:/DepthContrast/models \
        --volume $PROJ_DIR/scripts:/DepthContrast/scripts \
        --volume $PROJ_DIR/utils:/DepthContrast/utils \
        --volume $PROJ_DIR/lib:/DepthContrast/lib \
        $PCDET_VOLUMES \
        --rm \
        ssl:minkunet_fixed bash
