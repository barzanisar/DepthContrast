#!/bin/bash


# # $SLURM_NTASKS should be the same as $SLURM_NNODES
# echo "SLURM NTASKS: $SLURM_NTASKS"
# echo "SLURM_NNODES: $SLURM_NNODES" 

# echo "command line args for launch_ddp.sh"
# echo "$1 $2 $3 $4 $5"

# Extract Dataset
echo "Extracting Waymo data in Node: $SLURM_NODEID, SLURM_TMPDIR: $SLURM_TMPDIR"
TMP_DATA_DIR=$SLURM_TMPDIR/data

echo "Unzipping $DATA_DIR/$PROCESSED_DATA_TAG.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/$PROCESSED_DATA_TAG.zip -d $TMP_DATA_DIR

echo "Unzipping $INFOS_DIR/${PROCESSED_DATA_TAG}_infos_train_splits.zip to $TMP_DATA_DIR"
unzip -qq $INFOS_DIR/${PROCESSED_DATA_TAG}_infos_train_splits.zip -d $TMP_DATA_DIR

# echo "Unzipping $DATA_DIR/Infos/waymo_infos.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/Infos/waymo_infos.zip -d $TMP_DATA_DIR

# echo "Unzipping $DATA_DIR/waymo_processed_data_10_short_gt_database_train_sampled_1.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/waymo_processed_data_10_short_gt_database_train_sampled_1.zip -d $TMP_DATA_DIR

echo "Done extracting Waymo data"

# Get last element in string and increment by 1
NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
NUM_GPUS=$(($NUM_GPUS + 1))
WORLD_SIZE=$((NUM_GPUS * SLURM_NNODES))


echo "NUM GPUS in Node $SLURM_NODEID: $NUM_GPUS"
echo "Node $SLURM_NODEID says: split $SPLIT"
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
--bind $TMP_DATA_DIR:/DepthContrast/data/waymo
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD
TRAIN_CMD+="python /DepthContrast/tools/cluster_waymo.py --split $SPLIT --processed_data_tag $PROCESSED_DATA_TAG"

echo "Running training"
echo "Node $SLURM_NODEID says: Launching python script..."

echo "$TRAIN_CMD"
eval $TRAIN_CMD
echo "Done training"

