# sbatch --time=01:00:00 --nodes=3 --ntasks=3 --array=1-1%1 --job-name=pretrain_pointrcnn scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19664 --cfg_file configs/waymo_pointrcnn.yaml
# sbatch --time=01:00:00 --nodes=3 --ntasks=3 --array=1-1%1 --job-name=pretrain_pointrcnn_v1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19664 --cfg_file configs/waymo_pointrcnn_v1.yaml

### Pretrain in stages e.g. backbone, then head
### centerpoint in stages
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p24 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19300 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p24.yaml
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p12 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19301 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p12.yaml
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p012 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19302 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p012.yaml

### PointRCNN in stages
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p24 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19303 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p24.yaml
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p12 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19304 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p12.yaml
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p012 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19305 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p012.yaml

### Pretrain full networks
# # CenterPoint full
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p24 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19306 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p24.yaml
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p12 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19307 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p12.yaml
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p012 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19308 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p012.yaml

# ### pointrcnn full
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p24 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19309 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p24.yaml
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p12 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19310 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p12.yaml
# sbatch --time=01:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p012 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19311 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p012.yaml

################################### Pretrain on 1 node
### Pretrain in stages e.g. backbone, then head
### centerpoint in stages
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p24_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19312 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p24_1node.yaml
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p12_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19313 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p12_1node.yaml
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_stages_lr0p012_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19314 --cfg_file configs/waymo_centerpoint_pseudo8_stages_lr0p012_1node.yaml

### PointRCNN in stages
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p24_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19315 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p24_1node.yaml
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p12_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p12_1node.yaml
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_stages_lr0p012_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19317 --cfg_file configs/waymo_pointrcnn_pseudo8_stages_lr0p012_1node.yaml


#Iou backbone variations (thresh, weight)
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --cfg_file configs/waymo_pointrcnn_backbone_iou_weight.yaml
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=seg_contrast scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19348 --cfg_file configs/waymo_pointrcnn_backbone_segcontrast.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_thresh_0p8 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19318 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_thresh_0p8.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_thresh_0p7 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19317 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_thresh_0p7.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_thresh_0p6 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_thresh_0p6.yaml

# sbatch --dependency=afterany:23472769 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_thresh_0p5 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19315 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_thresh_0p5.yaml
# sbatch --dependency=afterany:23472770 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19338 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_weight.yaml
# sbatch --dependency=afterany:23472771 --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=lpseg_iou_segcontrast scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19348 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_segcontrast.yaml

#sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=iou_thresh_0p8 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19318 --cfg_file configs/waymo_pointrcnn_backbone_iou_thresh_0p8.yaml
#sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=iou_thresh_0p7 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19317 --cfg_file configs/waymo_pointrcnn_backbone_iou_thresh_0p7.yaml
#sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=iou_thresh_0p6 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --cfg_file configs/waymo_pointrcnn_backbone_iou_thresh_0p6.yaml
#sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=iou_thresh_0p5 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19315 --cfg_file configs/waymo_pointrcnn_backbone_iou_thresh_0p5.yaml

sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_thresh_0p6_lower_bs scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19316 --cfg_file configs/waymo_pointrcnn_backbone_iou_thresh_0p6_lower_bs_epochs_data.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=segcontrast_lower_bs scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19319 --cfg_file configs/waymo_pointrcnn_backbone_segcontrast_lower_bs_epochs_data.yaml

#shape backbone variations (thresh, weight)
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=vfh_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19210 --cfg_file configs/waymo_pointrcnn_backbone_vfh_thresh_0p2.yaml

# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=vfh_thresh_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19211 --cfg_file configs/waymo_pointrcnn_backbone_vfh_thresh_0p4.yaml
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=esf_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19212 --cfg_file configs/waymo_pointrcnn_backbone_esf_thresh_0p2.yaml
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=gasd_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19213 --cfg_file configs/waymo_pointrcnn_backbone_gasd_thresh_0p2.yaml

# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=vfh_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19214 --cfg_file configs/waymo_pointrcnn_backbone_vfh_weight.yaml
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=esf_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19215 --cfg_file configs/waymo_pointrcnn_backbone_esf_weight.yaml
# sbatch --time=17:00:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=gasd_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19216 --cfg_file configs/waymo_pointrcnn_backbone_gasd_weight.yaml


# #Done
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_thresh_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19211 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_thresh_0p4.yaml
#sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_esf_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19212 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_thresh_0p2.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_gasd_thresh_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19213 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_gasd_thresh_0p2.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_vfh_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19214 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_vfh_weight.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_esf_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19215 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_esf_weight.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_gasd_weight scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19216 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_gasd_weight.yaml

### Pretrain full networks
# CenterPoint full
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p24_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19318 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p24_1node.yaml
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p12_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19319 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p12_1node.yaml
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_centerpoint_pseudo8_lr0p012_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19320 --cfg_file configs/waymo_centerpoint_pseudo8_lr0p012_1node.yaml

### pointrcnn full
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p24_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19667 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p24_1node.yaml
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p12_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19668 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p12_1node.yaml
# sbatch --time=01:00:00 --nodes=1 --ntasks=1 --array=1-4%1 --job-name=pretrain_pointrcnn_pseudo8_lr0p012_1node scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19669 --cfg_file configs/waymo_pointrcnn_pseudo8_lr0p012_1node.yaml



# needs 1 hr 14 min for each split (199 sequences) of waymo_10% -> should take around 12-13 hours for each 200 seq split of full waymo
# check if --processed_data_tag waymo_processed_data_v_1_2_0 in /home/barza/DepthContrast/scripts/submit_compute_canada_cluster_waymo.sh



### Gives Cuda out of memory errors on sort
#Done
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19323 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19327 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p4.yaml
#In progress
#sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19326 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p3.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=iou_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19325 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p2.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-2%1 --job-name=iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19324 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p1.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p5 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19328 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p5.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=iou_perc_0p6 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19329 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p6.yaml
#In progress
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=shape_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19423 --cfg_file configs/waymo_pointrcnn_backbone_shape_perc_0p05.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=shape_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19427 --cfg_file configs/waymo_pointrcnn_backbone_shape_perc_0p4.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=shape_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19426 --cfg_file configs/waymo_pointrcnn_backbone_shape_perc_0p3.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=shape_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19425 --cfg_file configs/waymo_pointrcnn_backbone_shape_perc_0p2.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=shape_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19424 --cfg_file configs/waymo_pointrcnn_backbone_shape_perc_0p1.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=shape_perc_0p5 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19428 --cfg_file configs/waymo_pointrcnn_backbone_shape_perc_0p5.yaml
sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=shape_perc_0p6 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19429 --cfg_file configs/waymo_pointrcnn_backbone_shape_perc_0p6.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19323 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19327 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p4.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-3%1 --job-name=lpseg_iou_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19328 --downstream --cfg_file configs/waymo_lpseg_pointrcnn_backbone_iou_perc_0p3.yaml

# # Not tried yet
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_segcontrast scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19520 --cfg_file configs/waymo_minkunet_segcontrast.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19521 --cfg_file configs/waymo_minkunet_esf_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19522 --cfg_file configs/waymo_minkunet_esf_perc_0p1.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19523 --cfg_file configs/waymo_minkunet_esf_perc_0p2.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_perc_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19524 --cfg_file configs/waymo_minkunet_esf_perc_0p3.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_thres_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19525 --cfg_file configs/waymo_minkunet_esf_thresh_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_thres_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19526 --cfg_file configs/waymo_minkunet_esf_thresh_0p1.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_thres_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19527 --cfg_file configs/waymo_minkunet_esf_thresh_0p2.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_esf_thres_0p3 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19528 --cfg_file configs/waymo_minkunet_esf_thresh_0p3.yaml


# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_iou_perc_0p05 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19529 --cfg_file configs/waymo_minkunet_iou_perc_0p05.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_iou_perc_0p1 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19530 --cfg_file configs/waymo_minkunet_iou_perc_0p1.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_iou_perc_0p2 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19531 --cfg_file configs/waymo_minkunet_iou_perc_0p2.yaml

# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_iou_thres_0p6 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19532 --cfg_file configs/waymo_minkunet_iou_thresh_0p6.yaml
# sbatch --time=6:00:00 --nodes=1 --ntasks=1 --array=1-6%1 --job-name=mink_iou_thres_0p4 scripts/submit_ddp_compute_canada_waymo_multinode.sh --tcp_port 19533 --cfg_file configs/waymo_minkunet_iou_thresh_0p4.yaml
