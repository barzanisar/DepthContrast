# FOV3000

# #Dense-Kitti-SemKitti-Pretrain

#SegContrast
#sbatch --time=10:00:00 --array=1-5%1 --job-name=seg_1in2_cube_up_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --cluster --tcp_port 18849 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_cube_up_dense_semkitti.yaml
#sbatch --time=10:00:00 --array=1-5%1 --job-name=seg_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --cluster --tcp_port 18840 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_dense_semkitti.yaml

#DepthContrast
#sbatch --time=24:00:00 --array=1-2%1 --job-name=dc_1in2_cube_up_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --tcp_port 18949 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_cube_up_dense_semkitti.yaml
#sbatch --time=24:00:00 --array=1-2%1 --job-name=dc_dense_semkitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_semkitti.sh --tcp_port 18640 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_dense_semkitti.yaml

#Dense-Kitti-Pretrain

#SegContrast
sbatch --time=15:00:00 --gres=gpu:t4:2 --array=1-3%1 --job-name=seg_1in2_cube_up_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 18969 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_1in2_cube_up_dense_kitti.yaml
sbatch --time=15:00:00 --gres=gpu:t4:2 --array=1-3%1 --job-name=seg_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --cluster --tcp_port 18660 --cfg_file configs/pointnet_train_all_FOV3000_60/seg/seg_dense_kitti.yaml

# Done pretraining upto 330 epochs
# #DepthContrast
# sbatch --time=24:00:00 --array=1-2%1 --job-name=dc_1in2_cube_up_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19969 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_1in2_cube_up_dense_kitti.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=dc_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 19660 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_dense_kitti.yaml

#TODO:
# Add ablation: seg-snow/fog sim, seg-adversetoclear

# #DepthContrast: only Dense-Kitti
# sbatch --time=00:30:00 --array=1-1%1 --job-name=dc_b14_lr0p15_red_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 18949 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/shortlist/dc_b14_lr0p15_red_dense_kitti.yaml
# sbatch --time=00:30:00 --array=1-1%1 --job-name=dc_1in2_cube_up_b14_lr0p15_red_dense_kitti --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense_kitti.sh --tcp_port 18640 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/shortlist/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti.yaml
