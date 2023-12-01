################# Clustering ###########################################################################################

sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_0 scripts/submit_compute_canada_cluster_waymo.sh --split train_0
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_1 scripts/submit_compute_canada_cluster_waymo.sh --split train_1
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_2 scripts/submit_compute_canada_cluster_waymo.sh --split train_2
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_3 scripts/submit_compute_canada_cluster_waymo.sh --split train_3

sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_4 scripts/submit_compute_canada_cluster_waymo.sh --split train_4
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_5 scripts/submit_compute_canada_cluster_waymo.sh --split train_5
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_6 scripts/submit_compute_canada_cluster_waymo.sh --split train_6
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_7 scripts/submit_compute_canada_cluster_waymo.sh --split train_7

sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_8 scripts/submit_compute_canada_cluster_waymo.sh --split train_8
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_9 scripts/submit_compute_canada_cluster_waymo.sh --split train_9
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_10 scripts/submit_compute_canada_cluster_waymo.sh --split train_10
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_11 scripts/submit_compute_canada_cluster_waymo.sh --split train_11

sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_12 scripts/submit_compute_canada_cluster_waymo.sh --split train_12
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_13 scripts/submit_compute_canada_cluster_waymo.sh --split train_13
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_14 scripts/submit_compute_canada_cluster_waymo.sh --split train_14
sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo_15 scripts/submit_compute_canada_cluster_waymo.sh --split train_15

sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo_val_0 scripts/submit_compute_canada_cluster_waymo.sh --split val_0
sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo_val_1 scripts/submit_compute_canada_cluster_waymo.sh --split val_1
sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo_val_2 scripts/submit_compute_canada_cluster_waymo.sh --split val_2
sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo_val_3 scripts/submit_compute_canada_cluster_waymo.sh --split val_3
sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo_val_4 scripts/submit_compute_canada_cluster_waymo.sh --split val_4
sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo_val_5 scripts/submit_compute_canada_cluster_waymo.sh --split val_5
sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo_val_6 scripts/submit_compute_canada_cluster_waymo.sh --split val_6
sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo_val_7 scripts/submit_compute_canada_cluster_waymo.sh --split val_7