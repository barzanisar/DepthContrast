#!/bin/bash

#DC FOV3000
sbatch --time=03:00:00 --array=1-1%1 --job-name=pointnet-train-all-FOV3000-dc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos
--tcp_port 28800 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc.yaml

sbatch --time=03:00:00 --array=1-1%1 --job-name=pointnet-train-all-FOV3000-dc_snow_coupled_fog_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos
--tcp_port 28801 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fog_cube_upsample.yaml
