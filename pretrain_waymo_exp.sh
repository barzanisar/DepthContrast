# In progress:
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=segcontrast_lower_bs scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19319 --cfg_file configs/waymo_pointrcnn_backbone_segcontrast_lower_bs_epochs_data.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=iou_thresh_0p6_lower_bs scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --cfg_file configs/waymo_pointrcnn_backbone_iou_thresh_0p6_lower_bs_epochs_data.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=vfh_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19210 --cfg_file configs/waymo_pointrcnn_backbone_vfh_thresh_0p2.yaml

# Done
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=esf_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19212 --cfg_file configs/waymo_pointrcnn_backbone_esf_thresh_0p2.yaml
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=vfh_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19214 --cfg_file configs/waymo_pointrcnn_backbone_vfh_weight.yaml
# sbatch --time=6:00:00 --mem=250G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=esf_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19215 --cfg_file configs/waymo_pointrcnn_backbone_esf_weight.yaml

# In progress:
#sbatch --time=6:00:00  --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --cfg_file configs/waymo_pointrcnn_backbone_iou_weight.yaml
#sbatch --time=6:00:00  --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --cfg_file configs/waymo_pointrcnn_backbone_iou_weight.yaml
#sbatch --time=6:00:00  --gres=gpu:v100:8 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --cfg_file configs/waymo_pointrcnn_backbone_iou_weight.yaml

# sbatch --time=3:00:00  --dependency=afterany:13839570 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_segcontrast scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19319 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_segcontrast.yaml
# sbatch --time=6:00:00 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19324 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p1.yaml
# sbatch --time=6:00:00 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19523 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_perc_0p05.yaml
# sbatch --time=6:00:00 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19524 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_perc_0p1.yaml

# sbatch --time=6:00:00 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_vfh_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19210 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_thresh_0p2.yaml
# sbatch --time=6:00:00 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19323 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p05.yaml
# sbatch --time=6:00:00 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_weight.yaml
# sbatch --time=6:00:00 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-4%1 --job-name=lpseg_esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19423 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p05.yaml
# sbatch --time=6:00:00 --gres=gpu:t4:4 --mem=100G --nodes=1 --ntasks=1 --array=1-4%1 --job-name=lpseg_esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19424 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p1.yaml

# try cedar
sbatch --time=9:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19324 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p1.yaml
# sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19523 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_perc_0p05.yaml
# sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19524 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_perc_0p1.yaml

# sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19210 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_thresh_0p2.yaml
# sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19323 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p05.yaml
#sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_weight.yaml
# sbatch --time=6:00:00 --dependency=afterany:21243692 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-4%1 --job-name=lpseg_esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19423 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p05.yaml
# sbatch --time=6:00:00 --dependency=afterany:21243695 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-4%1 --job-name=lpseg_esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19424 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p1.yaml
# sbatch --time=6:00:00 --dependency=afterany:21243703 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-4%1 --job-name=lpseg_iou_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19425 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p3.yaml
# sbatch --time=6:00:00 --dependency=afterany:21243705 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-4%1 --job-name=lpseg_iou_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19426 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p4.yaml
sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_esf_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19427 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p3.yaml
sbatch --time=2:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_esf_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19428 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p4.yaml

sbatch --time=10:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19325 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p2.yaml
sbatch --time=10:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_perc_0p5 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19326 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p5.yaml
sbatch --time=10:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_perc_0p3_esf_wt scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19327 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p3_esf_weight.yaml
sbatch --time=10:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=100G --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_perc_0p3_iou_wt scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19328 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p3_iou_weight.yaml
 

# sbatch --dependency=afterany:23627050 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_segcontrast scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19319 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_segcontrast.yaml
# sbatch --dependency=afterany:23627049 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_thresh_0p6 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_thresh_0p6.yaml
# sbatch --dependency=afterany:23627051 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19210 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_thresh_0p2.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_esf_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19212 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_thresh_0p2.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19214 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_weight.yaml
# sbatch --time=6:00:00 --dependency=afterany:23807305 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_esf_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19215 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_weight.yaml

# In progress:
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_weight.yaml


# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19323 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19324 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p1.yaml
# sbatch --time=2:00:00 --nodes=1 --ntasks=1 --array=1-9%1 --job-name=iou_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19325 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p3.yaml
# sbatch --time=2:00:00 --nodes=1 --ntasks=1 --array=1-9%1 --job-name=iou_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19326 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p4.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19325 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p2.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p5 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19326 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p5.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=iou_perc_0p3_esf_wt scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19327 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p3_esf_weight.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p3_iou_wt scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19328 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p3_iou_weight.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19423 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19424 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p1.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=esf_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19425 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p3.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=esf_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19426 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p4.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=vfh_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19523 --cfg_file configs/waymo_pointrcnn_backbone_vfh_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=vfh_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19524 --cfg_file configs/waymo_pointrcnn_backbone_vfh_perc_0p1.yaml

# In progress:
# sbatch --dependency=afterany:23653292 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19323 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p05.yaml
# sbatch --dependency=afterany:23653293 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19324 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p1.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19423 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19424 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_perc_0p1.yaml

# sbatch --dependency=afterany:23653296 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19523 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_perc_0p05.yaml
# sbatch --dependency=afterany:23653297 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19524 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_perc_0p1.yaml




################################### MinkUnet
# sbatch --time=14:00:00 --dependency=afterany:21479439 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=2 --ntasks=2 --array=1-1%1 --job-name=mink_segcontrast scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19520 --cfg_file configs/waymo_minkunet_segcontrast.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19521 --cfg_file configs/waymo_minkunet_esf_perc_0p05.yaml
#sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=mink_esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19522 --cfg_file configs/waymo_minkunet_esf_perc_0p1.yaml
# sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19523 --cfg_file configs/waymo_minkunet_esf_perc_0p2.yaml
# sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19524 --cfg_file configs/waymo_minkunet_esf_perc_0p3.yaml
#sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19525 --cfg_file configs/waymo_minkunet_esf_perc_0p4.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19529 --cfg_file configs/waymo_minkunet_iou_perc_0p05.yaml
# sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19530 --cfg_file configs/waymo_minkunet_iou_perc_0p1.yaml
#sbatch --time=12:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-5%1 --job-name=mink_iou_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19531 --cfg_file configs/waymo_minkunet_iou_perc_0p2.yaml
# sbatch --time=12:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=2 --ntasks=2 --array=1-6%1 --job-name=mink_iou_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19532 --cfg_file configs/waymo_minkunet_iou_perc_0p3.yaml
# sbatch --time=6:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_iou_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19533 --cfg_file configs/waymo_minkunet_iou_perc_0p4.yaml

sbatch --time=10:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_mink_segcontrast scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19521 --downstream --cfg_file configs/waymo_lpseg_minkunet_segcontrast.yaml --linear_probe_last_n_ckpts 10
sbatch --time=10:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_mink_iou_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19522 --downstream --cfg_file configs/waymo_lpseg_minkunet_iou_perc_0p3.yaml --linear_probe_last_n_ckpts 10


##################### Final ###############

#narval
sbatch --time=10:00:00 --nodes=1 --ntasks=1 --array=1-5%1 --job-name=ptrcnn_iou_perc_0p3_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19325 --cfg_file configs/waymo_pointrcnn_iou_perc_0p3_waymo10.yaml
sbatch --time=10:00:00 --nodes=1 --ntasks=1 --array=1-5%1 --job-name=ptrcnn_segcontrast_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19326 --cfg_file configs/waymo_pointrcnn_segcontrast_waymo10.yaml

#cedar
#adjust batchsize to fit in cedar
# Pointrcnn pretrain stage1
sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=ptrcnn_iou_perc_0p3_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19325 --cfg_file configs/waymo_pointrcnn_iou_perc_0p3_waymo10.yaml
sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=ptrcnn_segcontrast_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19326 --cfg_file configs/waymo_pointrcnn_segcontrast_waymo10.yaml

# Pointrcnn pretrain stage2
sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=ptrcnn_iou_perc_0p3_w10_stage2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19325 --cfg_file configs/waymo_pointrcnn_iou_perc_0p3_waymo10_stage2.yaml
sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=ptrcnn_segcontrast_w10_stage2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19326 --cfg_file configs/waymo_pointrcnn_segcontrast_waymo10_stage2.yaml

# Pointrcnn linear probe stage1
sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_ptrcnn_iou_perc_0p3_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19335 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_iou_perc_0p3_waymo10.yaml --linear_probe_last_n_ckpts 10
sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_ptrcnn_segcontrast_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19336 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_segcontrast_waymo10.yaml --linear_probe_last_n_ckpts 10

# Pointrcnn linear probe stage2
sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_ptrcnn_iou_perc_0p3_w10_stage2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19345 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_iou_perc_0p3_waymo10_stage2.yaml --linear_probe_last_n_ckpts 10
sbatch --time=1:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-1%1 --job-name=lpseg_ptrcnn_segcontrast_w10_stage2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19346 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_segcontrast_waymo10_stage2.yaml --linear_probe_last_n_ckpts 10

#minkunet pretrain
sbatch --time=15:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=2 --ntasks=2 --array=1-5%1 --job-name=mink_segcontrast_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19631 --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml
sbatch --time=15:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=2 --ntasks=2 --array=1-5%1 --job-name=mink_iou_perc_0p3_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19621 --cfg_file configs/waymo_minkunet_iou_perc_0p3_waymo10.yaml

#minkunet linear probe
sbatch --time=15:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-5%1 --job-name=lpseg_mink_segcontrast_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19631 --downstream --cfg_file configs/waymo_lpseg_minkunet_segcontrast_waymo10.yaml --linear_probe_last_n_ckpts 10
sbatch --time=15:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=1 --ntasks=1 --array=1-5%1 --job-name=lpseg_mink_iou_perc_0p3_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19621 --downstream --cfg_file configs/waymo_lpseg_minkunet_iou_perc_0p3_waymo10.yaml --linear_probe_last_n_ckpts 10

#minkunet finetune one lr
sbatch --time=15:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=2 --ntasks=2 --array=1-5%1 --job-name=fine_1lr_mink_segcontrast_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19641 --downstream --cfg_file configs/waymo_fine1lr_minkunet_segcontrast_waymo10.yaml --pretrained_ckpt checkpoint.pth.tar
sbatch --time=15:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=2 --ntasks=2 --array=1-5%1 --job-name=fine_1lr_mink_iou_perc_0p3_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19651 --downstream --cfg_file configs/waymo_fine1lr_minkunet_iou_perc_0p3_waymo10.yaml --pretrained_ckpt checkpoint.pth.tar

#minkunet finetune two lrs
sbatch --time=15:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=2 --ntasks=2 --array=1-5%1 --job-name=fine_2lrs_mink_segcontrast_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19661 --downstream --cfg_file configs/waymo_fine2lrs_minkunet_segcontrast_waymo10.yaml --pretrained_ckpt checkpoint.pth.tar
sbatch --time=15:00:00 --account=def-swasland-ab --gres=gpu:v100l:4 --mem=180G --nodes=2 --ntasks=2 --array=1-5%1 --job-name=fine_2lrs_mink_iou_perc_0p3_w10 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19671 --downstream --cfg_file configs/waymo_fine2lrs_minkunet_iou_perc_0p3_waymo10.yaml --pretrained_ckpt checkpoint.pth.tar
