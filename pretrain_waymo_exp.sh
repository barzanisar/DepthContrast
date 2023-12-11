# In progress:
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=segcontrast_lower_bs scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19319 --cfg_file configs/waymo_pointrcnn_backbone_segcontrast_lower_bs_epochs_data.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_thresh_0p6_lower_bs scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --cfg_file configs/waymo_pointrcnn_backbone_iou_thresh_0p6_lower_bs_epochs_data.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=vfh_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19210 --cfg_file configs/waymo_pointrcnn_backbone_vfh_thresh_0p2.yaml

# Done
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=esf_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19212 --cfg_file configs/waymo_pointrcnn_backbone_esf_thresh_0p2.yaml
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=vfh_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19214 --cfg_file configs/waymo_pointrcnn_backbone_vfh_weight.yaml
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=esf_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19215 --cfg_file configs/waymo_pointrcnn_backbone_esf_weight.yaml

sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --cfg_file configs/waymo_pointrcnn_backbone_iou_weight.yaml

sbatch --dependency=afterany:23627050 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_segcontrast scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19319 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_segcontrast.yaml
sbatch --dependency=afterany:23627049 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_thresh_0p6 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_thresh_0p6.yaml
sbatch --dependency=afterany:23627051 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19210 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_thresh_0p2.yaml

# In progress:
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_esf_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19212 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_thresh_0p2.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19214 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_weight.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_esf_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19215 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_weight.yaml

# Not Done
#sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_weight.yaml


sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19323 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p05.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19324 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p1.yaml

sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19423 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19424 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p1.yaml

sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=vfh_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19523 --cfg_file configs/waymo_pointrcnn_backbone_vfh_perc_0p05.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=vfh_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19524 --cfg_file configs/waymo_pointrcnn_backbone_vfh_perc_0p1.yaml

# Not Done
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19323 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19324 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p1.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19423 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19424 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p1.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19523 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19524 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_perc_0p1.yaml
