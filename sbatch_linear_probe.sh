#!/bin/bash

# VDC objects
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_objects_1layer_gamma0_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18900 --linear_probe --cfg_file configs/vdc/vdc_lp_objects_1layer_gamma0_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_objects_1layer_gamma0_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18904 --linear_probe --cfg_file configs/vdc/vdc_lp_objects_1layer_gamma0_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_objects_1layer_gamma0_lr0p1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18910 --linear_probe --cfg_file configs/vdc/vdc_lp_objects_1layer_gamma0_lr0p1.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_objects_2layers_gamma0_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19900 --linear_probe --cfg_file configs/vdc/vdc_lp_objects_2layers_gamma0_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_objects_2layers_gamma0_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19700 --linear_probe --cfg_file configs/vdc/vdc_lp_objects_2layers_gamma0_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_objects_2layers_gamma0_lr0p1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19600 --linear_probe --cfg_file configs/vdc/vdc_lp_objects_2layers_gamma0_lr0p1.yaml

# DC objects
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_objects_1layer_gamma0_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18800 --linear_probe --cfg_file configs/dc/dc_lp_objects_1layer_gamma0_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_objects_1layer_gamma0_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18804 --linear_probe --cfg_file configs/dc/dc_lp_objects_1layer_gamma0_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_objects_1layer_gamma0_lr0p1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18810 --linear_probe --cfg_file configs/dc/dc_lp_objects_1layer_gamma0_lr0p1.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_objects_2layers_gamma0_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18900 --linear_probe --cfg_file configs/dc/dc_lp_objects_2layers_gamma0_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_objects_2layers_gamma0_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18700 --linear_probe --cfg_file configs/dc/dc_lp_objects_2layers_gamma0_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_objects_2layers_gamma0_lr0p1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18600 --linear_probe --cfg_file configs/dc/dc_lp_objects_2layers_gamma0_lr0p1.yaml


# VDC classes 1 layer
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_1layer_gamma0_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18901 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_1layer_gamma0_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_1layer_gamma0_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18902 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_1layer_gamma0_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_1layer_gamma2_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18903 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_1layer_gamma2_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_1layer_gamma2_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18905 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_1layer_gamma2_lr0p001.yaml

# DC classes 1 layer
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_1layer_gamma0_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18801 --linear_probe --cfg_file configs/dc/dc_lp_classes_1layer_gamma0_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_1layer_gamma0_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18802 --linear_probe --cfg_file configs/dc/dc_lp_classes_1layer_gamma0_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_1layer_gamma2_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18803 --linear_probe --cfg_file configs/dc/dc_lp_classes_1layer_gamma2_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_1layer_gamma2_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18805 --linear_probe --cfg_file configs/dc/dc_lp_classes_1layer_gamma2_lr0p001.yaml


# VDC classes 2 layers
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_2layers_gamma0_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19300 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_2layers_gamma0_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_2layers_gamma0_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19400 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_2layers_gamma0_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_2layers_gamma0_lr0p1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19500 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_2layers_gamma0_lr0p1.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_2layers_gamma2_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19200 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_2layers_gamma2_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_2layers_gamma2_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19100 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_2layers_gamma2_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=vdc_lp_classes_2layers_gamma2_lr0p1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 19110 --linear_probe --cfg_file configs/vdc/vdc_lp_classes_2layers_gamma2_lr0p1.yaml


# DC classes 2 layers
sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_2layers_gamma0_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18300 --linear_probe --cfg_file configs/dc/dc_lp_classes_2layers_gamma0_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_2layers_gamma0_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18400 --linear_probe --cfg_file configs/dc/dc_lp_classes_2layers_gamma0_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_2layers_gamma0_lr0p1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18500 --linear_probe --cfg_file configs/dc/dc_lp_classes_2layers_gamma0_lr0p1.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_2layers_gamma2_lr0p001 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18200 --linear_probe --cfg_file configs/dc/dc_lp_classes_2layers_gamma2_lr0p001.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_2layers_gamma2_lr0p01 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18100 --linear_probe --cfg_file configs/dc/dc_lp_classes_2layers_gamma2_lr0p01.yaml

sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=dc_lp_classes_2layers_gamma2_lr0p1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh --tcp_port 18110 --linear_probe --cfg_file configs/dc/dc_lp_classes_2layers_gamma2_lr0p1.yaml



