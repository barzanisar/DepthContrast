sbatch --time=1:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pretrain_waymo_short scripts/compute_canada_train_depth_contrast_waymo.sh --tcp_port 19469 --cfg_file configs/waymo.yaml
