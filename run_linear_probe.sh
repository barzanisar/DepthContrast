#!/bin/bash

# dc_vdc
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointnet_lp_dc_vdc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18900 --linear_probe --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/linear_probe/dc_vdc_FOV3000.yaml
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointnet_lp_dc_vdc_snow1in2_wet_fog1in2_cubeF_upsample_FOV3000 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18900 --linear_probe --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/linear_probe/dc_vdc_snow1in2_wet_fog1in2_cubeF_upsample_FOV3000.yaml
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointnet_lp_dc_vdc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18901 --linear_probe --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/linear_probe/dc_vdc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000.yaml
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointnet_lp_dc_vdc_snow1in10_wet_fog1in10_cubeF_upsample_FOV3000 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18902 --linear_probe --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/linear_probe/dc_vdc_snow1in10_wet_fog1in10_cubeF_upsample_FOV3000.yaml
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointnet_lp_dc_vdc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18903 --linear_probe --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/linear_probe/dc_vdc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000.yaml
