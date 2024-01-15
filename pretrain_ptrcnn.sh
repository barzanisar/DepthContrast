sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19210 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19211 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p1.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19212 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p2.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19213 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p3.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19214 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p4.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p5 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19215 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p5.yaml

sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p01 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19216 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p01.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=iou_perc_0p01 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19217 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p01.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p025 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19218 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p025.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=iou_perc_0p025 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19219 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p025.yaml

sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=iou_perc_0p05_iou_wt_w3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19216 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p05_iou_wt.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=iou_perc_0p05_esf_wt_w3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19217 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p05_esf_wt.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p05_iou_wt_w3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19218 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05_iou_wt.yaml
sbatch --time=3:00:00 --nodes=1 --ntasks=1 --array=1-10%1 --job-name=esf_perc_0p05_esf_wt_w3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19219 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05_esf_wt.yaml
