#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=DepthContrast-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=8             # CPU cores/threads
#SBATCH --gres=gpu:t4:4                # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --output=./output/log/%x-%j.out   # STDOUT
#SBATCH --mail-type=ALL
#SBATCH --array=1-3%1   # 4 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# main.py script parameters
CFG_FILE=configs/point_within_lidar_vdc_linear_probe.yaml
DIST="true"
TCP_PORT=18888

# Additional parameters
DENSE_DATA_DIR=/home/$USER/projects/rrg-swasland/Datasets/Dense
SING_IMG=/home/$USER/projects/rrg-swasland/singularity/depth_contrast_snow_sim.sif
DIST=true
WANDB_API_KEY=$WANDB_API_KEY
WANDB_MODE='offline' #'dryrun'

# Get last element in string and increment by 1
NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
NUM_GPUS=$(($NUM_GPUS + 1))


# Extract Semantic Kitti Dataset
echo "Extracting Semantic Kitti data"
TMP_DATA_DIR_SEM_KITTI=$SLURM_TMPDIR/semantic_kitti_data

echo "Unzipping $DENSE_DATA_DIR/semantic_kitti/dataset.zip to $TMP_DATA_DIR_SEM_KITTI"
unzip -qq $DENSE_DATA_DIR/semantic_kitti/dataset.zip -d $TMP_DATA_DIR_SEM_KITTI

echo "Done extracting Semantic Kitti data"

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
--bind $TMP_DATA_DIR_SEM_KITTI:/DepthContrast/data/semantic_kitti
--bind $PROJ_DIR/data/dense/ImageSets:/DepthContrast/data/dense/ImageSets
--bind $PROJ_DIR/lib:/DepthContrast/lib
$DEPTH_CONTRAST_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD

if [ $DIST != "true" ]
then
    TRAIN_CMD+="python /DepthContrast/tools/main_dist.py --cfg /DepthContrast/$CFG_FILE
"
else
    TRAIN_CMD+="python -m torch.distributed.launch
    --nproc_per_node=$NUM_GPUS
    /DepthContrast/tools/main_dist_demo.py
    --tcp_port $TCP_PORT --multiprocessing-distributed --cfg /DepthContrast/$CFG_FILE
    "
fi

echo "Running training"
echo "$TRAIN_CMD"
eval $TRAIN_CMD
echo "Done training"




