#!/bin/bash

# Extract Dataset
echo "Extracting Waymo data in Node: $SLURM_NODEID, SLURM_TMPDIR: $SLURM_TMPDIR"
TMP_DATA_DIR=$SLURM_TMPDIR/data

# echo "Unzipping $DATA_DIR/waymo_processed_data_10.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/waymo_processed_data_10.zip -d $TMP_DATA_DIR

# echo "Unzipping $DATA_DIR/Infos/waymo_processed_data_10_infos.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/Infos/waymo_processed_data_10_infos.zip -d $TMP_DATA_DIR

# echo "Unzipping $DATA_DIR/Infos/waymo_infos.zip to $TMP_DATA_DIR"
# unzip -qq $DATA_DIR/Infos/waymo_infos.zip -d $TMP_DATA_DIR

echo "Unzipping $DATA_DIR/waymo_processed_data_10_short.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/waymo_processed_data_10_short.zip -d $TMP_DATA_DIR

echo "Unzipping $DATA_DIR/waymo_processed_data_10_short_infos.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/waymo_processed_data_10_short_infos.zip -d $TMP_DATA_DIR

echo "Unzipping $DATA_DIR/waymo_processed_data_10_short_gt_database_train_sampled_1.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/waymo_processed_data_10_short_gt_database_train_sampled_1.zip -d $TMP_DATA_DIR

echo "Done extracting Waymo data"

#export TMP_DATA_DIR=$TMP_DATA_DIR