# sbatch --time=1:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pretrain_waymo_short scripts/compute_canada_train_depth_contrast_waymo.sh --tcp_port 19469 --cfg_file configs/waymo.yaml
sbatch --time=01:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pretrain_waymo_short_multinode scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19664 --cfg_file configs/waymo.yaml
#sbatch --time=00:30:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pretrain_waymo_short_multinode scripts/submit_slurm_compute_canada_waymo_multinode.sh --tcp_port 19554 --cfg_file configs/waymo.yaml

#sbatch --time=00:30:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_short

# needs 1 hr 14 min for each split (199 sequences) of waymo_10% -> should take around 12-13 hours for each 200 seq split of full waymo
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_0 --processed_data_tag waymo_processed_data_10
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_1 --processed_data_tag waymo_processed_data_10
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_2 --processed_data_tag waymo_processed_data_10
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_3 --processed_data_tag waymo_processed_data_10
