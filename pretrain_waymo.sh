sbatch --time=01:00:00 --array=1-1%1 --job-name=pretrain_waymo_short_multinode scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19664 --cfg_file configs/waymo_pointrcnn.yaml
sbatch --time=01:00:00 --array=1-1%1 --job-name=pretrain_waymo_short_multinode scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19667 --cfg_file configs/waymo_centerpoint.yaml

# needs 1 hr 14 min for each split (199 sequences) of waymo_10% -> should take around 12-13 hours for each 200 seq split of full waymo
# check if --processed_data_tag waymo_processed_data_v_1_2_0 in /home/barza/DepthContrast/scripts/submit_compute_canada_cluster_waymo.sh


################# Clustering #############################################################################################
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_0
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_1
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_2
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_3

# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_4
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_5
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_6
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_7

# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_8
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_9
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_10
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_11

# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_12
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_13
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_14
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_15

# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_0
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_1
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_2
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_3
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_4
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_5
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_6
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_7


