#!/bin/bash

sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-2%1 --job-name=seg_lp_1in2_cube_up_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_dense_linear_probe.sh --tcp_port 18900 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_lp_1in2_cube_up_dense_kitti.yaml
sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-2%1 --job-name=seg_lpSem_1in2_cube_up_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_semkitti_linear_probe.sh --tcp_port 18990 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_lpSem_1in2_cube_up_dense_kitti.yaml
