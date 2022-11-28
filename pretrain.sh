# FOV3000

# SEMANTIC-KITTI
#sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_semkitti-all --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_semkitti.sh --cluster --tcp_port 19569 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/semKitti/seg_semkitti_all.yaml

# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_semkitti.sh --cluster --tcp_port 19569 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/semKitti/seg_semkitti_syncbn.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_semkitti.sh --cluster --tcp_port 19369 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/semKitti/seg_semkitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_semkitti-segTransforms --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_semkitti.sh --cluster --tcp_port 19469 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/semKitti/seg_semkitti_segTransforms.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_semkitti-syncbn --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_semkitti.sh --cluster --tcp_port 19569 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/semKitti/seg_semkitti_syncbn.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_semkitti-segTransforms-b14 --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_semkitti.sh --cluster --tcp_port 19669 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/semKitti/seg_semkitti_segTransforms_b14.yaml

# #Dense-Kitti-SemKitti-Pretrain
#sbatch --time=05:00:00 --gres=gpu:t4:4 --array=1-20%1 --job-name=seg_dense_semkitti_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti_semkitti.sh --cluster --tcp_port 19569 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_dense_semkitti_kitti.yaml
sbatch --time=10:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=seg_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --cluster --tcp_port 19469 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_dense_semkitti.yaml

#Weather
#sbatch --time=05:00:00 --gres=gpu:t4:4 --array=1-20%1 --job-name=seg_dense_semkitti_kitti_weather --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti_semkitti.sh --cluster --tcp_port 19669 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_dense_semkitti_kitti_weather.yaml
sbatch --time=10:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=seg_dense_semkitti_weather --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --cluster --tcp_port 19869 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_dense_semkitti_weather.yaml

#SegContrast
#sbatch --time=10:00:00 --array=1-5%1 --job-name=seg_1in2_cube_up_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --cluster --tcp_port 18849 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_cube_up_dense_semkitti.yaml
#sbatch --time=10:00:00 --array=1-5%1 --job-name=seg_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --cluster --tcp_port 18840 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_dense_semkitti.yaml

#DepthContrast
#sbatch --time=24:00:00 --array=1-2%1 --job-name=dc_1in2_cube_up_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --tcp_port 18949 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_cube_up_dense_semkitti.yaml
#sbatch --time=24:00:00 --array=1-2%1 --job-name=dc_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --tcp_port 18640 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_dense_semkitti.yaml

#Dense-Kitti-Pretrain

#SegContrast
#--time=01:00:00 --array=1-1%1 --job-name=seg_1in2_cube_up_dense_kitti --mail-user=barzanisar93@gmail.com

#DGX
#sbatch --time=01:00:00 scripts/dgx_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 18969 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_cube_up_dense_kitti.yaml

#sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=seg_1in2_cube_up_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 18969 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_cube_up_dense_kitti.yaml
#sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=seg_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 18669 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_dense_kitti.yaml

#Weather Ablations on SEG
#--gres=gpu:v100:4
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_coupleF_upF_fog_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 19369 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_coupleF_upF_fog_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_coupleF_upF_snow_wet_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 19379 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_coupleF_upF_snow_wet_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_coupleF_upF_snow_wet_fog_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 19389 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_coupleF_upF_snow_wet_fog_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_coupleF_upF_snow_wetF_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 19399 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_coupleF_upF_snow_wetF_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_coupleF_upF_snow_wetF_fog_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 19329 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_coupleF_upF_snow_wetF_fog_dense_kitti.yaml


# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in10_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 19269 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in10_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_upF_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 19169 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_upF_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_coupleF_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 19369 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_coupleF_dense_kitti.yaml

# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in10_upF_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19369 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in10_upF_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_upF_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19169 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_upF_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_snow_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19469 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_snow_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_fog_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19569 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_fog_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_snow_fog_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19669 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_snow_fog_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=seg_1in2_up_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19769 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_up_dense_kitti.yaml

# Done pretraining upto 330 epochs
# #DepthContrast
# sbatch --time=24:00:00 --array=1-2%1 --job-name=dc_1in2_cube_up_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_cube_up_dense_kitti.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=dc_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19660 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_dense_kitti.yaml

# #Weather Ablations on DC
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=dc_1in2_upF_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 11969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_upF_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=dc_1in10_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 12969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in10_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=dc_1in10_upF_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 13969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in10_upF_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=dc_1in2_snow_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 14969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_snow_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=dc_1in2_fog_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 15969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_fog_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=dc_1in2_snow_fog_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 16969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_snow_fog_dense_kitti.yaml
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-30%1 --job-name=dc_1in2_up_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 17969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_up_dense_kitti.yaml



#____________________________REMOVE BELOW_____________________________________________________________________________________________
#TODO:
# Add ablation: seg-snow/fog sim, seg-adversetoclear

# #DepthContrast: only Dense-Kitti
# sbatch --time=00:30:00 --array=1-1%1 --job-name=dc_b14_lr0p15_red_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 18949 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/shortlist/dc_b14_lr0p15_red_dense_kitti.yaml
# sbatch --time=00:30:00 --array=1-1%1 --job-name=dc_1in2_cube_up_b14_lr0p15_red_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 18640 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/shortlist/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti.yaml
