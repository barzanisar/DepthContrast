# sbatch --time=01:00:00 --nodes=3 --ntasks=3 --array=1-1%1 --job-name=pretrain_pointrcnn scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19664 --cfg_file configs/waymo_pointrcnn.yaml
# sbatch --time=01:00:00 --nodes=3 --ntasks=3 --array=1-1%1 --job-name=pretrain_pointrcnn_v1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19664 --cfg_file configs/waymo_pointrcnn_v1.yaml

### Pretrain in stages e.g. backbone, then head
### centerpoint in stages
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p24 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19300 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p24.yaml
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p12 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19301 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p12.yaml
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p012 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19302 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p012.yaml

### PointRCNN in stages
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p24 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19303 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p24.yaml
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p12 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19304 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p12.yaml
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p012 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19305 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p012.yaml

### Pretrain full networks
# CenterPoint full
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p24 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19306 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p24.yaml
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p12 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19307 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p12.yaml
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p012 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19308 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p012.yaml

### pointrcnn full
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p24 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19309 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p24.yaml
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p12 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19310 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p12.yaml
sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p012 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19311 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p012.yaml

################################### Pretrain on 1 node
### Pretrain in stages e.g. backbone, then head
### centerpoint in stages
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p24_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19312 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p24_1node.yaml
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p12_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19313 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p12_1node.yaml
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p012_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19314 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p012_1node.yaml

### PointRCNN in stages
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p24_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19315 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p24_1node.yaml
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p12_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p12_1node.yaml
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p012_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19317 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p012_1node.yaml

### Pretrain full networks
# CenterPoint full
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p24_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19318 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p24_1node.yaml
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p12_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19319 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p12_1node.yaml
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p012_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19320 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p012_1node.yaml

### pointrcnn full
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p24_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19667 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p24_1node.yaml
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p12_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19668 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p12_1node.yaml
sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p012_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19669 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p012_1node.yaml



# needs 1 hr 14 min for each split (199 sequences) of waymo_10% -> should take around 12-13 hours for each 200 seq split of full waymo
# check if --processed_data_tag waymo_processed_data_v_1_2_0 in /home/barza/DepthContrast/scripts/submit_compute_canada_cluster_waymo.sh


################# Clustering #############################################################################################
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_0
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_1
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_2
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_3

# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_4
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_5
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_6
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_7

# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_8
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_9
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_10
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_11

# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_12
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_13
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_14
# sbatch --time=05:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split train_15

# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_0
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_1
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_2
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_3
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_4
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_5
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_6
# sbatch --time=03:00:00 --array=1-1%1 --job-name=cluster_waymo scripts/submit_compute_canada_cluster_waymo.sh --split val_7


