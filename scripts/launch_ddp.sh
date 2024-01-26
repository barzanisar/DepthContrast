#!/bin/bash

# echo "Inside extract data slurm_tmpdir: $SLURM_TMPDIR"
# echo "Inside extract data waymo data dir: $WAYMO_DATA_DIR"

# # Extract Waymo Dataset
# echo "Extracting Waymo data"
# TMP_DATA_DIR=$SLURM_TMPDIR/waymo_data

# for file in $WAYMO_DATA_DIR/*.zip; do
#    echo "Unzipping $file to $TMP_DATA_DIR"
#    unzip -qq $file -d $TMP_DATA_DIR
# done
# echo "Done extracting dataset Waymo infos"

# Get last element in string and increment by 1
NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
NUM_GPUS=$(($NUM_GPUS + 1))
WORLD_SIZE=$((NUM_GPUS * SLURM_NNODES))
WORKERS_PER_GPU=$(($SLURM_CPUS_PER_TASK / $NUM_GPUS))


echo "NUM GPUS in Node $SLURM_NODEID: $NUM_GPUS"
echo "Node $SLURM_NODEID says: main node at $MASTER_ADDR:$MASTER_PORT"
echo "Node $SLURM_NODEID says: WORLD_SIZE=$WORLD_SIZE"
echo "Node $SLURM_NODEID says: WORKERS_PER_GPU=$SLURM_CPUS_PER_TASK / $NUM_GPUS=$WORKERS_PER_GPU"
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
--bind $WAYMO_DATA_DIR:/DepthContrast/data/waymo
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"

PRETRAIN_CMD=$BASE_CMD
PRETRAIN_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/main_dist.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$PRETRAIN_CFG_FILE --world-size $WORLD_SIZE 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $PRETRAIN_EPOCHS 
--batchsize_per_gpu $PRETRAIN_BATCHSIZE_PER_GPU 
--workers $WORKERS_PER_GPU 
--model_name $MODEL_NAME
"


FINETUNE_CMD=$BASE_CMD

FINETUNE_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/downstream_segmentation.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$FINETUNE_CFG_FILE --world-size $WORLD_SIZE 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $FINETUNE_EPOCHS
--batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
--downstream_model_dir $DOWNSTREAM_MODEL_DIR
--model_name $MODEL_NAME
--pretrained_ckpt $PRETRAINED_CKPT 
--workers $WORKERS_PER_GPU
"

SCRATCH_CMD=$BASE_CMD

SCRATCH_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/downstream_segmentation.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$SCRATCH_CFG_FILE --world-size $WORLD_SIZE 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $FINETUNE_EPOCHS
--batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
--downstream_model_dir $DOWNSTREAM_MODEL_DIR
--pretrained_ckpt $PRETRAINED_CKPT 
--workers $WORKERS_PER_GPU
"

LINEARPROBE_CMD=$BASE_CMD

LINEARPROBE_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/DepthContrast/tools/downstream_segmentation.py
--launcher pytorch
--multiprocessing-distributed --cfg /DepthContrast/$LINEARPROBE_CFG_FILE --world-size $WORLD_SIZE 
--dist-url tcp://$MASTER_ADDR:$TCP_PORT 
--epochs $LINEARPROBE_EPOCHS
--batchsize_per_gpu $LINEARPROBE_BATCHSIZE_PER_GPU 
--downstream_model_dir $DOWNSTREAM_MODEL_DIR
--model_name $MODEL_NAME
--linear_probe_last_n_ckpts $LINEARPROBE_LAST_N_CKPTS 
--workers $WORKERS_PER_GPU
"

if [[ "$MODE" == "pretrain-finetune" ]]; then
    echo "Running Pretraining"
    echo "$PRETRAIN_CMD"
    eval $PRETRAIN_CMD
    echo "Done pretraining"

    echo "Running Finetuning"
    echo "$FINETUNE_CMD"
    eval $FINETUNE_CMD
    echo "Done Finetuning"

elif [[ "$MODE" == "linearprobe" ]]; then
    echo "Running Linear Probe Only"
    echo "$LINEARPROBE_CMD"
    eval $LINEARPROBE_CMD
    echo "Done linear probe"

elif [[ "$MODE" == "scratch" ]]; then
    echo "Running Scratch training"
    echo "$SCRATCH_CMD"
    eval $SCRATCH_CMD
    echo "Done scratch training"
    
fi




# if [[ "$MODE" == "pretrain-finetune" ]]; then
#     # Extract Semantic Kitti 
#     # echo "Remove waymo dataset"
#     # ls -l $TMP_DATA_DIR
#     # rm -r $TMP_DATA_DIR/*
#     # ls -l $TMP_DATA_DIR
#     # echo "Done removing waymo"
#     TMP_DATA_DIR=$SLURM_TMPDIR/semantickitti_data
#     FINETUNE_CFG_FILE=configs/semantickitti_fine1lr_$BACKBONE.yaml

#     echo "unzipping semantic kitti"
#     for file in $KITTI_DATA_DIR/*.zip; do
#         echo "Unzipping $file to $TMP_DATA_DIR"
#         unzip -qq $file -d $TMP_DATA_DIR
#     done
#     echo "Done extracting semantic kitti data"

#     BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
#     SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
#     SINGULARITYENV_WANDB_MODE=offline
#     SINGULARITYENV_NCCL_BLOCKING_WAIT=1
#     singularity exec
#     --nv
#     --pwd /DepthContrast
#     --bind $PROJ_DIR/checkpoints:/DepthContrast/checkpoints
#     --bind $PROJ_DIR/configs:/DepthContrast/configs
#     --bind $PROJ_DIR/criterions:/DepthContrast/criterions
#     --bind $PROJ_DIR/output:/DepthContrast/output
#     --bind $PROJ_DIR/datasets:/DepthContrast/datasets
#     --bind $PROJ_DIR/tools:/DepthContrast/tools
#     --bind $PROJ_DIR/models:/DepthContrast/models
#     --bind $PROJ_DIR/scripts:/DepthContrast/scripts
#     --bind $PROJ_DIR/utils:/DepthContrast/utils
#     --bind $TMP_DATA_DIR:/DepthContrast/data/semantic_kitti
#     --bind $PROJ_DIR/lib:/DepthContrast/lib
#     $DEPTH_CONTRAST_BINDS
#     $SING_IMG
#     "
#     FINETUNE_CMD=$BASE_CMD

#     FINETUNE_CMD+="python -m torch.distributed.launch
#     --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
#     /DepthContrast/tools/downstream_segmentation.py
#     --launcher pytorch
#     --multiprocessing-distributed --cfg /DepthContrast/$FINETUNE_CFG_FILE --world-size $WORLD_SIZE 
#     --dist-url tcp://$MASTER_ADDR:$TCP_PORT 
#     --epochs $FINETUNE_EPOCHS
#     --batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
#     --downstream_model_dir $DOWNSTREAM_MODEL_DIR
#     --model_name $MODEL_NAME
#     --pretrained_ckpt $PRETRAINED_CKPT 
#     --workers $WORKERS_PER_GPU
#     "
#     echo "Running Finetuning"
#     echo "$FINETUNE_CMD"
#     eval $FINETUNE_CMD
#     echo "Done Finetuning"

#     # Extract NuScenes  
#     # echo "Remove semantic kitti dataset"
#     # ls -l $TMP_DATA_DIR
#     # rm -r $TMP_DATA_DIR/*
#     # ls -l $TMP_DATA_DIR
#     # echo "Done removing semantic kitti"
#     TMP_DATA_DIR=$SLURM_TMPDIR/nuscenes_data
#     FINETUNE_CFG_FILE=configs/nuscenes_fine1lr_$BACKBONE.yaml


#     echo "unzipping nuscenes"
#     for file in $NUSCENES_DATA_DIR/*.zip; do
#         echo "Unzipping $file to $TMP_DATA_DIR"
#         unzip -qq $file -d $TMP_DATA_DIR
#     done
#     echo "Done extracting nuscenes data"

#     BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
#     SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
#     SINGULARITYENV_WANDB_MODE=offline
#     SINGULARITYENV_NCCL_BLOCKING_WAIT=1
#     singularity exec
#     --nv
#     --pwd /DepthContrast
#     --bind $PROJ_DIR/checkpoints:/DepthContrast/checkpoints
#     --bind $PROJ_DIR/configs:/DepthContrast/configs
#     --bind $PROJ_DIR/criterions:/DepthContrast/criterions
#     --bind $PROJ_DIR/output:/DepthContrast/output
#     --bind $PROJ_DIR/datasets:/DepthContrast/datasets
#     --bind $PROJ_DIR/tools:/DepthContrast/tools
#     --bind $PROJ_DIR/models:/DepthContrast/models
#     --bind $PROJ_DIR/scripts:/DepthContrast/scripts
#     --bind $PROJ_DIR/utils:/DepthContrast/utils
#     --bind $TMP_DATA_DIR:/DepthContrast/data/nuscenes
#     --bind $PROJ_DIR/lib:/DepthContrast/lib
#     $DEPTH_CONTRAST_BINDS
#     $SING_IMG
#     "
#     FINETUNE_CMD=$BASE_CMD

#     FINETUNE_CMD+="python -m torch.distributed.launch
#     --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
#     /DepthContrast/tools/downstream_segmentation.py
#     --launcher pytorch
#     --multiprocessing-distributed --cfg /DepthContrast/$FINETUNE_CFG_FILE --world-size $WORLD_SIZE 
#     --dist-url tcp://$MASTER_ADDR:$TCP_PORT 
#     --epochs $FINETUNE_EPOCHS
#     --batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
#     --downstream_model_dir $DOWNSTREAM_MODEL_DIR
#     --model_name $MODEL_NAME
#     --pretrained_ckpt $PRETRAINED_CKPT 
#     --workers $WORKERS_PER_GPU
#     "
#     echo "Running Finetuning"
#     echo "$FINETUNE_CMD"
#     eval $FINETUNE_CMD
#     echo "Done Finetuning"

# elif [[ "$MODE" == "scratch" ]]; then

#     TMP_DATA_DIR=$SLURM_TMPDIR/semantickitti_data
#     SCRATCH_CFG_FILE=configs/semantickitti_scratch_$BACKBONE.yaml

#     echo "unzipping semantic kitti"
#     for file in $KITTI_DATA_DIR/*.zip; do
#         echo "Unzipping $file to $TMP_DATA_DIR"
#         unzip -qq $file -d $TMP_DATA_DIR
#     done
#     echo "Done extracting semantic kitti data"

#     BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
#     SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
#     SINGULARITYENV_WANDB_MODE=offline
#     SINGULARITYENV_NCCL_BLOCKING_WAIT=1
#     singularity exec
#     --nv
#     --pwd /DepthContrast
#     --bind $PROJ_DIR/checkpoints:/DepthContrast/checkpoints
#     --bind $PROJ_DIR/configs:/DepthContrast/configs
#     --bind $PROJ_DIR/criterions:/DepthContrast/criterions
#     --bind $PROJ_DIR/output:/DepthContrast/output
#     --bind $PROJ_DIR/datasets:/DepthContrast/datasets
#     --bind $PROJ_DIR/tools:/DepthContrast/tools
#     --bind $PROJ_DIR/models:/DepthContrast/models
#     --bind $PROJ_DIR/scripts:/DepthContrast/scripts
#     --bind $PROJ_DIR/utils:/DepthContrast/utils
#     --bind $TMP_DATA_DIR:/DepthContrast/data/semantic_kitti
#     --bind $PROJ_DIR/lib:/DepthContrast/lib
#     $DEPTH_CONTRAST_BINDS
#     $SING_IMG
#     "
#     SCRATCH_CMD=$BASE_CMD

#     SCRATCH_CMD+="python -m torch.distributed.launch
#     --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
#     /DepthContrast/tools/downstream_segmentation.py
#     --launcher pytorch
#     --multiprocessing-distributed --cfg /DepthContrast/$SCRATCH_CFG_FILE --world-size $WORLD_SIZE 
#     --dist-url tcp://$MASTER_ADDR:$TCP_PORT 
#     --epochs $FINETUNE_EPOCHS
#     --batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
#     --downstream_model_dir $DOWNSTREAM_MODEL_DIR
#     --pretrained_ckpt $PRETRAINED_CKPT 
#     --workers $WORKERS_PER_GPU
#     "
#     echo "Running Scratch training"
#     echo "$SCRATCH_CMD"
#     eval $SCRATCH_CMD
#     echo "Done scratch training"

#     # Extract NuScenes  
#     # echo "Remove semantic kitti dataset"
#     # ls -l $TMP_DATA_DIR
#     # rm -r $TMP_DATA_DIR/*
#     # ls -l $TMP_DATA_DIR
#     # echo "Done removing semantic kitti"
#     TMP_DATA_DIR=$SLURM_TMPDIR/nuscenes_data
#     SCRATCH_CFG_FILE=configs/nuscenes_scratch_$BACKBONE.yaml


#     echo "unzipping nuscenes"
#     for file in $NUSCENES_DATA_DIR/*.zip; do
#         echo "Unzipping $file to $TMP_DATA_DIR"
#         unzip -qq $file -d $TMP_DATA_DIR
#     done
#     echo "Done extracting nuscenes data"

#     BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
#     SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
#     SINGULARITYENV_WANDB_MODE=offline
#     SINGULARITYENV_NCCL_BLOCKING_WAIT=1
#     singularity exec
#     --nv
#     --pwd /DepthContrast
#     --bind $PROJ_DIR/checkpoints:/DepthContrast/checkpoints
#     --bind $PROJ_DIR/configs:/DepthContrast/configs
#     --bind $PROJ_DIR/criterions:/DepthContrast/criterions
#     --bind $PROJ_DIR/output:/DepthContrast/output
#     --bind $PROJ_DIR/datasets:/DepthContrast/datasets
#     --bind $PROJ_DIR/tools:/DepthContrast/tools
#     --bind $PROJ_DIR/models:/DepthContrast/models
#     --bind $PROJ_DIR/scripts:/DepthContrast/scripts
#     --bind $PROJ_DIR/utils:/DepthContrast/utils
#     --bind $TMP_DATA_DIR:/DepthContrast/data/nuscenes
#     --bind $PROJ_DIR/lib:/DepthContrast/lib
#     $DEPTH_CONTRAST_BINDS
#     $SING_IMG
#     "
#     SCRATCH_CMD=$BASE_CMD

#     SCRATCH_CMD+="python -m torch.distributed.launch
#     --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
#     /DepthContrast/tools/downstream_segmentation.py
#     --launcher pytorch
#     --multiprocessing-distributed --cfg /DepthContrast/$SCRATCH_CFG_FILE --world-size $WORLD_SIZE 
#     --dist-url tcp://$MASTER_ADDR:$TCP_PORT 
#     --epochs $FINETUNE_EPOCHS
#     --batchsize_per_gpu $FINETUNE_BATCHSIZE_PER_GPU 
#     --downstream_model_dir $DOWNSTREAM_MODEL_DIR
#     --pretrained_ckpt $PRETRAINED_CKPT 
#     --workers $WORKERS_PER_GPU
#     "

#     echo "Running Scratch training"
#     echo "$SCRATCH_CMD"
#     eval $SCRATCH_CMD
#     echo "Done scratch training"
    
# fi