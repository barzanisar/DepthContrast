sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_segcontrast_w3 scripts/submit_ddp_cedar.sh --tcp_port 19831 --cfg_file configs/waymo_minkunet_segcontrast_waymo3.yaml
sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_iou_perc_0p05_w3 scripts/submit_ddp_cedar.sh --tcp_port 19832 --cfg_file configs/waymo_minkunet_iou_perc_0p05_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_iou_perc_0p05_iou_wt_w3 scripts/submit_ddp_cedar.sh --tcp_port 19833 --cfg_file configs/waymo_minkunet_iou_perc_0p05_iou_wt_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_iou_perc_0p05_esf_wt_w3 scripts/submit_ddp_cedar.sh --tcp_port 19834 --cfg_file configs/waymo_minkunet_iou_perc_0p05_esf_wt_waymo3.yaml
sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_iou_perc_0p4_w3 scripts/submit_ddp_cedar.sh --tcp_port 19835 --cfg_file configs/waymo_minkunet_iou_perc_0p4_waymo3.yaml
sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_iou_weight_w3 scripts/submit_ddp_cedar.sh --tcp_port 19836 --cfg_file configs/waymo_minkunet_iou_weight_waymo3.yaml

sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_perc_0p2_w3 scripts/submit_ddp_cedar.sh --tcp_port 19837 --cfg_file configs/waymo_minkunet_esf_perc_0p2_waymo3.yaml
sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_perc_0p3_w3 scripts/submit_ddp_cedar.sh --tcp_port 19838 --cfg_file configs/waymo_minkunet_esf_perc_0p3_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_perc_0p3_iou_wt_w3 scripts/submit_ddp_cedar.sh --tcp_port 19839 --cfg_file configs/waymo_minkunet_esf_perc_0p3_iou_wt_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_perc_0p3_esf_wt_w3 scripts/submit_ddp_cedar.sh --tcp_port 19840 --cfg_file configs/waymo_minkunet_esf_perc_0p3_esf_wt_waymo3.yaml
sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_perc_0p4_w3 scripts/submit_ddp_cedar.sh --tcp_port 19841 --cfg_file configs/waymo_minkunet_esf_perc_0p4_waymo3.yaml
sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_weight_w3 scripts/submit_ddp_cedar.sh --tcp_port 19842 --cfg_file configs/waymo_minkunet_esf_weight_waymo3.yaml
