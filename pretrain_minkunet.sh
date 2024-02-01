# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-7%1 --job-name=mink_segcontrast_w3 scripts/submit_ddp_cedar.sh --tcp_port 19831 --cfg_file configs/waymo_minkunet_segcontrast_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-7%1 --job-name=mink_iou_perc_0p05_w3 scripts/submit_ddp_cedar.sh --tcp_port 19832 --cfg_file configs/waymo_minkunet_iou_perc_0p05_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-7%1 --job-name=mink_iou_perc_0p05_iou_wt_w3 scripts/submit_ddp_cedar.sh --tcp_port 19833 --cfg_file configs/waymo_minkunet_iou_perc_0p05_iou_wt_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-7%1 --job-name=mink_iou_perc_0p05_esf_wt_w3 scripts/submit_ddp_cedar.sh --tcp_port 19834 --cfg_file configs/waymo_minkunet_iou_perc_0p05_esf_wt_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_iou_perc_0p4_w3 scripts/submit_ddp_cedar.sh --tcp_port 19835 --cfg_file configs/waymo_minkunet_iou_perc_0p4_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_iou_weight_w3 scripts/submit_ddp_cedar.sh --tcp_port 19836 --cfg_file configs/waymo_minkunet_iou_weight_waymo3.yaml

# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=mink_esf_perc_0p2_w3 scripts/submit_ddp_cedar.sh --tcp_port 19837 --cfg_file configs/waymo_minkunet_esf_perc_0p2_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_perc_0p3_w3 scripts/submit_ddp_cedar.sh --tcp_port 19838 --cfg_file configs/waymo_minkunet_esf_perc_0p3_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-7%1 --job-name=mink_esf_perc_0p3_iou_wt_w3 scripts/submit_ddp_cedar.sh --tcp_port 19839 --cfg_file configs/waymo_minkunet_esf_perc_0p3_iou_wt_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-7%1 --job-name=mink_esf_perc_0p3_esf_wt_w3 scripts/submit_ddp_cedar.sh --tcp_port 19840 --cfg_file configs/waymo_minkunet_esf_perc_0p3_esf_wt_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_perc_0p4_w3 scripts/submit_ddp_cedar.sh --tcp_port 19841 --cfg_file configs/waymo_minkunet_esf_perc_0p4_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-10%1 --job-name=mink_esf_weight_w3 scripts/submit_ddp_cedar.sh --tcp_port 19842 --cfg_file configs/waymo_minkunet_esf_weight_waymo3.yaml

# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_iou_perc_0p4_w3 scripts/submit_ddp_cedar.sh --tcp_port 19214 --cfg_file configs/waymo_fine1lr_minkunet.yaml --downstream  --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_iou_perc_0p4_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_esf_weight_w3 scripts/submit_ddp_cedar.sh --tcp_port 19214 --cfg_file configs/waymo_fine1lr_minkunet.yaml --downstream  --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_esf_weight_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_iou_weight_w3 scripts/submit_ddp_cedar.sh --tcp_port 19214 --cfg_file configs/waymo_fine1lr_minkunet.yaml --downstream  --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_iou_weight_waymo3

# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=mink_esf_perc_0p01_w3 scripts/submit_ddp_cedar.sh --tcp_port 19837 --cfg_file configs/waymo_minkunet_esf_perc_0p01_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-2%1 --job-name=mink_esf_perc_0p05_w3 scripts/submit_ddp_cedar.sh --tcp_port 19838 --cfg_file configs/waymo_minkunet_esf_perc_0p05_waymo3.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-7%1 --job-name=mink_iou_perc_0p01_w3 scripts/submit_ddp_cedar.sh --tcp_port 19839 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo3.yaml

# sbatch --time=00:30:00 --array=1-1%1 --job-name=fine1lr_mink_esf_perc_0p01_w3 scripts/submit_ddp_cedar.sh --tcp_port 19214 --cfg_file configs/waymo_fine1lr_minkunet.yaml --downstream  --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_esf_perc_0p01_waymo3
# sbatch --time=2:00:00 --array=1-1%1 --job-name=fine1lr_mink_esf_perc_0p05_w3 scripts/submit_ddp_cedar.sh --tcp_port 19215 --cfg_file configs/waymo_fine1lr_minkunet.yaml --downstream  --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_esf_perc_0p05_waymo3
# sbatch --time=2:00:00 --array=1-1%1 --job-name=fine1lr_mink_iou_perc_0p01_w3 scripts/submit_ddp_cedar.sh --tcp_port 19216 --cfg_file configs/waymo_fine1lr_minkunet.yaml --downstream  --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_iou_perc_0p01_waymo3
# sbatch --time=2:00:00 --array=1-1%1 --job-name=fine1lr_mink_segcontrast_w3 scripts/submit_ddp_cedar.sh --tcp_port 19215 --cfg_file configs/waymo_fine1lr_minkunet.yaml --downstream  --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_segcontrast_waymo3


#5% waymo for 100 epochs
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-14%1 --job-name=mink_iou_perc_0p05_w5 scripts/submit_ddp_cedar.sh --tcp_port 19831 --cfg_file configs/waymo_minkunet_iou_perc_0p05_waymo5.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-14%1 --job-name=mink_iou_perc_0p01_w5 scripts/submit_ddp_cedar.sh --tcp_port 19832 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo5.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-14%1 --job-name=mink_esf_perc_0p05_w5 scripts/submit_ddp_cedar.sh --tcp_port 19833 --cfg_file configs/waymo_minkunet_esf_perc_0p05_waymo5.yaml

# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-14%1 --job-name=mink_iou_perc_0p01_iou_wt_w5 scripts/submit_ddp_cedar.sh --tcp_port 19834 --cfg_file configs/waymo_minkunet_iou_perc_0p01_iou_wt_waymo5.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-14%1 --job-name=mink_esf_perc_0p05_iou_wt_w5 scripts/submit_ddp_cedar.sh --tcp_port 19835 --cfg_file configs/waymo_minkunet_esf_perc_0p05_iou_wt_waymo5.yaml

# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-14%1 --job-name=mink_iou_perc_0p01_esf_wt_w5 scripts/submit_ddp_cedar.sh --tcp_port 19836 --cfg_file configs/waymo_minkunet_iou_perc_0p01_esf_wt_waymo5.yaml
# sbatch --time=5:00:00 --nodes=2 --ntasks=2 --array=1-14%1 --job-name=mink_esf_perc_0p05_esf_wt_w5 scripts/submit_ddp_cedar.sh --tcp_port 19837 --cfg_file configs/waymo_minkunet_esf_perc_0p05_esf_wt_waymo5.yaml

# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_segcontrast_w3 scripts/submit_ddp_cedar.sh --tcp_port 19450 --downstream --cfg_file configs/waymo_fine1lr_minkunet.yaml --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_segcontrast_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_iou_perc_0p05_w3 scripts/submit_ddp_cedar.sh --tcp_port 19451 --downstream --cfg_file configs/waymo_fine1lr_minkunet.yaml --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_iou_perc_0p05_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_iou_perc_0p4_w3 scripts/submit_ddp_cedar.sh --tcp_port 19452 --downstream --cfg_file configs/waymo_fine1lr_minkunet.yaml --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_iou_perc_0p4_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_iou_weight_w3 scripts/submit_ddp_cedar.sh --tcp_port 19453 --downstream --cfg_file configs/waymo_fine1lr_minkunet.yaml --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_iou_weight_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_esf_perc_0p2_w3 scripts/submit_ddp_cedar.sh --tcp_port 19454 --downstream --cfg_file configs/waymo_fine1lr_minkunet.yaml --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_esf_perc_0p2_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_esf_perc_0p3_w3 scripts/submit_ddp_cedar.sh --tcp_port 19455 --downstream --cfg_file configs/waymo_fine1lr_minkunet.yaml --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_esf_perc_0p3_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_esf_perc_0p4_w3 scripts/submit_ddp_cedar.sh --tcp_port 19456 --downstream --cfg_file configs/waymo_fine1lr_minkunet.yaml --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_esf_perc_0p4_waymo3
# sbatch --time=1:30:00 --array=1-1%1 --job-name=fine1lr_mink_esf_weight_w3 scripts/submit_ddp_cedar.sh --tcp_port 19457 --downstream --cfg_file configs/waymo_fine1lr_minkunet.yaml --pretrained_ckpt checkpoint-ep49.pth.tar --model_name minkunet_pretrain_esf_weight_waymo3

# in progress in cedar
#sbatch --time=6:00:00 --array=1-1%1 --job-name=mink_esf_perc_0p01_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19831 --cfg_file configs/waymo_minkunet_esf_perc_0p01_waymo5.yaml --model_name minkunet_pretrain_esf_perc_0p01_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 
#sbatch --time=6:00:00 --array=1-1%1 --job-name=mink_esf_perc_0p05_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19832 --cfg_file configs/waymo_minkunet_esf_perc_0p05_waymo5.yaml --model_name minkunet_pretrain_esf_perc_0p05_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 

# done in cedar
sbatch --time=6:00:00 --array=1-1%1 --job-name=mink_esf_perc_0p05_iou_wt_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19833 --cfg_file configs/waymo_minkunet_esf_perc_0p05_iou_wt_waymo5.yaml --model_name minkunet_pretrain_esf_perc_0p05_iou_wt_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 

# done in cedar
sbatch --time=6:00:00 --array=1-2%1 --job-name=mink_esf_perc_0p05_esf_wt_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19834 --cfg_file configs/waymo_minkunet_esf_perc_0p05_esf_wt_waymo5.yaml --model_name minkunet_pretrain_esf_perc_0p05_esf_wt_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 

# in progress in cedar
sbatch --time=6:00:00 --array=1-2%1 --job-name=mink_esf_perc_0p05_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19835 --cfg_file configs/waymo_minkunet_esf_perc_0p05_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_perc_0p05_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 

# done in cedar
sbatch --time=6:00:00 --array=1-2%1 --job-name=mink_esf_perc_0p05_w5_t0p07 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19836 --cfg_file configs/waymo_minkunet_esf_perc_0p05_waymo5_t0p07.yaml --model_name minkunet_pretrain_esf_perc_0p05_waymo5_t0p07 --pretrained_ckpt checkpoint-ep49.pth.tar 
#sbatch --time=6:00:00 --array=1-1%1 --job-name=mink_segcontrast scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19843 --cfg_file configs/waymo_minkunet_segcontrast_waymo5.yaml --model_name minkunet_pretrain_segcontrast_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 

# redo
sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_iou_perc_0p01_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19837 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo5.yaml --model_name minkunet_pretrain_iou_perc_0p01_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 

# in progress in cedar
sbatch --time=6:00:00 --array=1-2%1 --job-name=mink_iou_perc_0p05_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19838 --cfg_file configs/waymo_minkunet_iou_perc_0p05_waymo5.yaml --model_name minkunet_pretrain_iou_perc_0p05_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 

#redo
sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_iou_perc_0p01_iou_wt_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19839 --cfg_file configs/waymo_minkunet_iou_perc_0p01_iou_wt_waymo5.yaml --model_name minkunet_pretrain_iou_perc_0p01_iou_wt_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 

# in progress in cedar
sbatch --time=6:00:00 --array=1-2%1 --job-name=mink_iou_perc_0p01_esf_wt_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19840 --cfg_file configs/waymo_minkunet_iou_perc_0p01_esf_wt_waymo5.yaml --model_name minkunet_pretrain_iou_perc_0p01_esf_wt_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-2%1 --job-name=mink_iou_perc_0p01_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19841 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_perc_0p01_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-2%1 --job-name=mink_iou_perc_0p01_w5_t0p07 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19842 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo5_t0p07.yaml --model_name minkunet_pretrain_iou_perc_0p01_waymo5_t0p07 --pretrained_ckpt checkpoint-ep49.pth.tar 

# sbatch --time=00:20:00 --gres=gpu:2 --array=1-1%1 --job-name=mink_esf_perc_0p01_w5 scripts/submit_ddp_vector.sh --tcp_port 19831 --cfg_file configs/waymo_minkunet_esf_perc_0p01_waymo5.yaml --model_name minkunet_pretrain_esf_perc_0p01_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=00:20:00 --gres=gpu:2 --array=1-1%1 --job-name=esf_perc_0p05 scripts/submit_ddp_vector.sh --tcp_port 19210 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05.yaml --model_name pointrcnn_pretrain_backbone_esf_perc_0p05 --pretrained_ckpt checkpoint-ep49.pth.tar 

# sbatch --time=03:00:00 --array=1-1%1 --job-name=mink_esf_perc_0p01_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19831 --cfg_file configs/waymo_minkunet_esf_perc_0p01_waymo5.yaml --model_name minkunet_pretrain_esf_perc_0p01_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar 

sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_gt_segcontrast_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19841 --cfg_file configs/waymo_minkunet_gt_segcontrast_waymo5_t0p04.yaml --model_name minkunet_pretrain_gt_segcontrast_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 

sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_iou_filter_0p01_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19844 --cfg_file configs/waymo_minkunet_iou_filter_0p01_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_filter_0p01_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_2_iou_filter_0p01_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19845 --cfg_file configs/waymo_minkunet_iou_filter_0p01_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_filter_0p01_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_iou_filter_0p5_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19842 --cfg_file configs/waymo_minkunet_iou_filter_0p5_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_filter_0p5_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_2_iou_filter_0p5_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19843 --cfg_file configs/waymo_minkunet_iou_filter_0p5_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_filter_0p5_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p15_iou_owt_waymo5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19846 --cfg_file configs/waymo_minkunet_iou_knn_0p15_iou_owt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p15_iou_owt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_2_iou_knn_0p15_iou_owt_waymo5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19847 --cfg_file configs/waymo_minkunet_iou_knn_0p15_iou_owt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p15_iou_owt_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p15_iou_iwt_waymo5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19848 --cfg_file configs/waymo_minkunet_iou_knn_0p15_iou_iwt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p15_iou_iwt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_2_iou_knn_0p15_iou_iwt_waymo5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19849 --cfg_file configs/waymo_minkunet_iou_knn_0p15_iou_iwt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p15_iou_iwt_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p15_esf_owt_waymo5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19850 --cfg_file configs/waymo_minkunet_iou_knn_0p15_esf_owt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p15_esf_owt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_2_iou_knn_0p15_esf_owt_waymo5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19851 --cfg_file configs/waymo_minkunet_iou_knn_0p15_esf_owt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p15_esf_owt_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p15_esf_iwt_waymo5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19852 --cfg_file configs/waymo_minkunet_iou_knn_0p15_esf_iwt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p15_esf_iwt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_2_iou_knn_0p15_esf_iwt_waymo5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19853 --cfg_file configs/waymo_minkunet_iou_knn_0p15_esf_iwt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p15_esf_iwt_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouANDesf_knn_0p23_iou_iwt_waymo5_t0p04_warmup_cuboid scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19853 --cfg_file configs/waymo_minkunet_iouANDesf_knn_0p23_iou_iwt_waymo5_t0p04_warmup_cuboid.yaml --model_name minkunet_pretrain_iouANDesf_knn_0p23_iou_iwt_waymo5_t0p04_warmup_cuboid --pretrained_ckpt checkpoint-ep50.pth.tar 
sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouANDesf_knn_0p23_iou_iwt_waymo5_t0p04_warmup scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19854 --cfg_file configs/waymo_minkunet_iouANDesf_knn_0p23_iou_iwt_waymo5_t0p04_warmup.yaml --model_name minkunet_pretrain_iouANDesf_knn_0p23_iou_iwt_waymo5_t0p04_warmup --pretrained_ckpt checkpoint-ep50.pth.tar 

sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouANDesf_knn_0p15_iou_iwt_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19853 --cfg_file configs/waymo_minkunet_iouANDesf_knn_0p15_iou_iwt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iouANDesf_knn_0p15_iou_iwt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p03_iou_iwt_w5_t0p02 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19856 --cfg_file configs/waymo_minkunet_iou_knn_0p03_iou_iwt_waymo5_t0p02.yaml --model_name minkunet_pretrain_iou_knn_0p03_iou_iwt_waymo5_t0p02 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p01_iou_iwt_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19857 --cfg_file configs/waymo_minkunet_iou_knn_0p01_iou_iwt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p01_iou_iwt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p03_iou_iwt_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19858 --cfg_file configs/waymo_minkunet_iou_knn_0p03_iou_iwt_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p03_iou_iwt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 


# sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_segcontrast_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19844 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_t0p04.yaml --model_name minkunet_pretrain_segcontrast_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
