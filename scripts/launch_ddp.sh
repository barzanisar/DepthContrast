#!/bin/bash


# # $SLURM_NTASKS should be the same as $SLURM_NNODES
# echo "SLURM NTASKS: $SLURM_NTASKS"
# echo "SLURM_NNODES: $SLURM_NNODES" 

# echo "command line args for launch_ddp.sh"
# echo "$1 $2 $3 $4 $5"

# Extract Dataset
# echo "Extracting Waymo data in Node: $SLURM_NODEID, SLURM_TMPDIR: $SLURM_TMPDIR"
# TMP_DATA_DIR=$SLURM_TMPDIR/data

# echo "Unzipping $DATA_DIR/waymo_processed_data_10.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/waymo_processed_data_10.zip -d $TMP_DATA_DIR

# # echo "Unzipping $DATA_DIR/Infos/waymo_processed_data_10_infos.zip to $TMP_DATA_DIR"
# # unzip -qq $DATA_DIR/Infos/waymo_processed_data_10_infos.zip -d $TMP_DATA_DIR

# # echo "Unzipping $DATA_DIR/Infos/waymo_infos.zip to $TMP_DATA_DIR"
# # unzip -qq $DATA_DIR/Infos/waymo_infos.zip -d $TMP_DATA_DIR

# # echo "Unzipping $DATA_DIR/waymo_processed_data_10_short.zip to $TMP_DATA_DIR"
# # unzip -qq $DATA_DIR/waymo_processed_data_10_short.zip -d $TMP_DATA_DIR

# echo "Copying $DATA_DIR/ImageSets to $TMP_DATA_DIR"
# cp -r $DATA_DIR/ImageSets -d $TMP_DATA_DIR

# # echo "Unzipping $DATA_DIR/waymo_processed_data_10_short_infos.zip to $TMP_DATA_DIR"
# # unzip -qq $DATA_DIR/waymo_processed_data_10_short_infos.zip -d $TMP_DATA_DIR

# # echo "Unzipping $DATA_DIR/waymo_processed_data_10_short_gt_database_train_sampled_1.zip to $TMP_DATA_DIR"
# # unzip -qq $DATA_DIR/waymo_processed_data_10_short_gt_database_train_sampled_1.zip -d $TMP_DATA_DIR

# echo "Done extracting Waymo data"

# Get last element in string and increment by 1
NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
NUM_GPUS=$(($NUM_GPUS + 1))
WORLD_SIZE=$((NUM_GPUS * SLURM_NNODES))


echo "NUM GPUS in Node $SLURM_NODEID: $NUM_GPUS"
echo "Node $SLURM_NODEID says: main node at $MASTER_ADDR:$MASTER_PORT"
echo "Node $SLURM_NODEID says: WORLD_SIZE=$WORLD_SIZE"
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
--bind $DATA_DIR:/DepthContrast/data/waymo
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD
# TRAIN_CMD+="python /DepthContrast/tools/main_dist.py --cfg /DepthContrast/$CFG_FILE"
TRAIN_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/main_dist.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$CFG_FILE --world-size $WORLD_SIZE 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $EPOCHS 
--batchsize_per_gpu $BATCHSIZE_PER_GPU
"

TEST_CMD=$BASE_CMD

TEST_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/downstream_segmentation.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$CFG_FILE --world-size $WORLD_SIZE 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--linear_probe_last_n_ckpts $LINEAR_PROBE_LAST_N_CKPTS
--epochs $EPOCHS
--batchsize_per_gpu $BATCHSIZE_PER_GPU
"


if [ $DOWNSTREAM == "true" ]
then
    if [ "$PRETRAINED_CKPT" != "default" ]
    then
        TEST_CMD+=" --pretrained_ckpt $PRETRAINED_CKPT"
    fi

    if [ "$MODEL_NAME" != "default" ]
    then
        TEST_CMD+=" --model_name $MODEL_NAME"
    fi

    if [ "$DOWNSTREAM_MODEL_DIR" != "default" ]
    then
        TEST_CMD+=" --downstream_model_dir $DOWNSTREAM_MODEL_DIR"
    fi

    echo "Running ONLY downstream"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TEST_CMD"
    eval $TEST_CMD
    echo "Done evaluation"
else
    echo "Running training"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TRAIN_CMD"
    eval $TRAIN_CMD
    echo "Done training"
fi
