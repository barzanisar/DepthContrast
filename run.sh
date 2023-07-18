#!/bin/bash

# PARAMETERS
#KITTI_TRAIN=$(readlink -f ../data/kitti/training)
#KITTI_TEST=$(readlink -f ../data/kitti/testing)
#WAYMO_RAW=$(readlink -f ../data/waymo/raw_data)
WAYMO_PROCESSED=$(readlink -f ./data/waymo/waymo_processed_data_10_short)
WAYMO_PROCESSED_INFO_TRAIN_PKL=$(readlink -f ./data/waymo/waymo_processed_data_10_short_infos_train_short.pkl)
WAYMO_GTDB=$(readlink -f ./data/waymo/waymo_processed_data_10_short_gt_database_train_sampled_1)
WAYMO_GTDB_PKL=$(readlink -f ./data/waymo/waymo_processed_data_10_short_waymo_dbinfos_train_sampled_1.pkl)

# DENSE_LIDAR=$(readlink -f ./data/dense/lidar_hdl64_strongest)
# SNOWFALL_LIDAR=$(readlink -f ./data/dense/snowfall_simulation)
# SNOWFALL_LIDAR_FOV=$(readlink -f ./data/dense/snowfall_simulation_FOV)
# SNOWFLAKES=$(readlink -f ./data/dense/snowflakes)
# DROR=$(readlink -f ./data/dense/DROR)

# Setup volume linking (host link:container link)
CUR_DIR=$(pwd)
PROJ_DIR=$CUR_DIR
#KITTI_TRAIN=$KITTI_TRAIN:/DepthContrast/data/kitti/training
#KITTI_TEST=$KITTI_TEST:/DepthContrast/data/kitti/testing
#WAYMO_RAW=$WAYMO_RAW:/DepthContrast/data/waymo/raw_data
WAYMO_PROCESSED=$WAYMO_PROCESSED:/DepthContrast/data/waymo/waymo_processed_data_10_short
WAYMO_PROCESSED_INFO_TRAIN_PKL=$WAYMO_PROCESSED_INFO_TRAIN_PKL:/DepthContrast/data/waymo/waymo_processed_data_10_short_infos_train_short.pkl
WAYMO_GTDB=$WAYMO_GTDB:/DepthContrast/data/waymo/waymo_processed_data_10_short_gt_database_train_sampled_1
WAYMO_GTDB_PKL=$WAYMO_GTDB_PKL:/DepthContrast/data/waymo/waymo_processed_data_10_short_waymo_dbinfos_train_sampled_1.pkl
# DENSE_LIDAR=$DENSE_LIDAR:/DepthContrast/data/dense/lidar_hdl64_strongest
# SNOWFALL_LIDAR=$SNOWFALL_LIDAR:/DepthContrast/data/dense/snowfall_simulation
# SNOWFALL_LIDAR_FOV=$SNOWFALL_LIDAR_FOV:/DepthContrast/data/dense/snowfall_simulation_FOV
# SNOWFLAKES=$SNOWFLAKES:/DepthContrast/data/dense/snowflakes
# DROR=$DROR:/DepthContrast/data/dense/DROR

PCDET_VOLUMES=""
for entry in $PROJ_DIR/third_party/OpenPCDet/pcdet/*
do
    name=$(basename $entry)

    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        PCDET_VOLUMES+="--volume $entry:/DepthContrast/third_party/OpenPCDet/pcdet/$name "
    fi
done

docker run -it --env="WANDB_API_KEY=$WANDB_API_KEY" \
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
        --volume $WAYMO_PROCESSED_INFO_TRAIN_PKL \
        --volume $WAYMO_GTDB \
        --volume $WAYMO_GTDB_PKL \
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
        --volume $PROJ_DIR/lib:/DepthContrast/lib \
        $PCDET_VOLUMES \
        --rm \
        ssl:free_invalid bash

#--volume $WAYMO_PROCESSED \
# --volume $DENSE_LIDAR \
# --volume $SNOWFALL_LIDAR \
# --volume $SNOWFLAKES \
# --volume $SNOWFALL_LIDAR_FOV \
# --volume $DROR \
