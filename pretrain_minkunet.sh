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

###########
# pd sbatch --time=10:00:00 --array=1-1%1 --job-name=mink_iouplusesf_knn_0p03_weight_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19855 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p03_weight_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p03_weight_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar
# d sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p03_filter_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19856 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p03_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p03_filter_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
# d sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p01_weight_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19857 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p01_weight_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p01_weight_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
# d sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p01_filter_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19858 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p01_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p01_filter_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
# d sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p01_filter_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19768 --cfg_file configs/waymo_minkunet_iou_knn_0p01_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p01_filter_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar

# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p03_weight_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19655 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p03_weight_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p03_weight_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p03_filter_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19656 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p03_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p03_filter_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p01_weight_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19657 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p01_weight_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p01_weight_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p01_filter_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19658 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p01_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p01_filter_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iou_knn_0p01_filter_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19769 --cfg_file configs/waymo_minkunet_iou_knn_0p01_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_iou_knn_0p01_filter_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar

###########
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_esf_knn_0p03_weight_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19755 --cfg_file configs/waymo_minkunet_esf_knn_0p03_weight_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_knn_0p03_weight_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=10:00:00 --array=1-1%1 --job-name=mink_esf_knn_0p03_filter_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19756 --cfg_file configs/waymo_minkunet_esf_knn_0p03_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_knn_0p03_filter_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_esf_knn_0p01_weight_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19757 --cfg_file configs/waymo_minkunet_esf_knn_0p01_weight_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_knn_0p01_weight_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=4:00:00 --array=1-1%1 --job-name=mink_esf_knn_0p01_filter_w5_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19758 --cfg_file configs/waymo_minkunet_esf_knn_0p01_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_knn_0p01_filter_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar

# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_esf_knn_0p03_weight_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19955 --cfg_file configs/waymo_minkunet_esf_knn_0p03_weight_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_knn_0p03_weight_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_esf_knn_0p03_filter_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19956 --cfg_file configs/waymo_minkunet_esf_knn_0p03_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_knn_0p03_filter_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_esf_knn_0p01_weight_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19957 --cfg_file configs/waymo_minkunet_esf_knn_0p01_weight_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_knn_0p01_weight_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar
# pd sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_esf_knn_0p01_filter_w5_t0p04_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19958 --cfg_file configs/waymo_minkunet_esf_knn_0p01_filter_waymo5_t0p04.yaml --model_name minkunet_pretrain_esf_knn_0p01_filter_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar

# sbatch --time=6:00:00 --array=1-4%1 --job-name=mink_segcontrast_t0p04 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19844 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_t0p04.yaml --model_name minkunet_pretrain_segcontrast_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=lidar_aug scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19844 --cfg_file configs/waymo_minkunet_segcontrast_lidar_aug_waymo5_t0p04.yaml --model_name minkunet_pretrain_segcontrast_lidar_aug_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=lidar_aug_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19845 --cfg_file configs/waymo_minkunet_segcontrast_lidar_aug_waymo5_t0p04.yaml --model_name minkunet_pretrain_segcontrast_lidar_aug_waymo5_t0p04_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 


# sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p01_weight_w5_t0p02 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19857 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p01_weight_waymo5_t0p02.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p01_weight_waymo5_t0p02 --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=mink_iouplusesf_knn_0p01_weight_w5_t0p02_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19856 --cfg_file configs/waymo_minkunet_iou_plus_esf_knn_0p01_weight_waymo5_t0p02.yaml --model_name minkunet_pretrain_iou_plus_esf_knn_0p01_weight_waymo5_t0p02_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

# sbatch --time=10:00:00 --array=1-4%1 --job-name=seg_cube_vs0p1 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19858 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_t0p04_cuboid_vs0p1.yaml --model_name minkunet_pretrain_segcontrast_waymo5_t0p04_cuboid_vs0p1 --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=seg_cube_vs0p1_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19859 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_t0p04_cuboid_vs0p1.yaml --model_name minkunet_pretrain_segcontrast_waymo5_t0p04_cuboid_vs0p1_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

# sbatch --time=10:00:00 --array=1-4%1 --job-name=det_0p1w scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19358 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_dethead_0p1w.yaml --model_name minkunet_pretrain_segcontrast_waymo5_dethead_0p1w --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det_0p1w_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19359 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_dethead_0p1w.yaml --model_name minkunet_pretrain_segcontrast_waymo5_dethead_0p1w_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

# sbatch --time=10:00:00 --array=1-4%1 --job-name=det_0p5w scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19558 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_dethead_0p5w.yaml --model_name minkunet_pretrain_segcontrast_waymo5_dethead_0p5w --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det_0p5w_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19559 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_dethead_0p5w.yaml --model_name minkunet_pretrain_segcontrast_waymo5_dethead_0p5w_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

# sbatch --time=10:00:00 --array=1-4%1 --job-name=det_0p8w scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19358 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_dethead_0p8w.yaml --model_name minkunet_pretrain_segcontrast_waymo5_dethead_0p8w --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det_0p8w_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19359 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_dethead_0p8w.yaml --model_name minkunet_pretrain_segcontrast_waymo5_dethead_0p8w_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

# sbatch --time=10:00:00 --array=1-4%1 --job-name=det_1w scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19558 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_dethead_1w.yaml --model_name minkunet_pretrain_segcontrast_waymo5_dethead_1w --pretrained_ckpt checkpoint-ep49.pth.tar 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det_1w_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19559 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_dethead_1w.yaml --model_name minkunet_pretrain_segcontrast_waymo5_dethead_1w_try2 --pretrained_ckpt checkpoint-ep49.pth.tar 

## Try1
sbatch --time=10:00:00 --array=1-10%1 --job-name=dc_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19558 --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml --model_name minkunet_pretrain_depthcontrast_waymo10 --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-10%1 --job-name=dc_lidaraug_dethead_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19559 --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_dethead_0p8w.yaml --model_name minkunet_pretrain_depthcontrast_waymo10_lidar_aug_dethead_0p8w --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets

sbatch --time=10:00:00 --array=1-10%1 --job-name=sc_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19560 --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml --model_name minkunet_pretrain_segcontrast_waymo10 --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-10%1 --job-name=sc_lidaraug_dethead_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19561 --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_dethead_0p8w.yaml --model_name minkunet_pretrain_segcontrast_waymo10_lidar_aug_dethead_0p8w --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-10%1 --job-name=sc_lidaraug_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19562 --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug.yaml --model_name minkunet_pretrain_segcontrast_waymo10_lidar_aug --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-10%1 --job-name=sc_dethead_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19563 --cfg_file configs/waymo_minkunet_segcontrast_waymo10_dethead_0p8w.yaml --model_name minkunet_pretrain_segcontrast_waymo10_dethead_0p8w --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets

sbatch --time=8:00:00 --array=1-1%1 --job-name=scratch_seg scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19570 --cfg_file configs/waymo_scratch_minkunet.yaml --mode scratch --finetune_epochs 15 --other_datasets --downstream_model_dir scratch_epochs15
sbatch --time=8:00:00 --array=1-2%1 --job-name=scratch_seg_5percent scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19571 --cfg_file configs/waymo_scratch_minkunet.yaml --mode scratch --finetune_epochs 15 --other_datasets --downstream_model_dir scratch_5percent --frame_sampling_div 5

# run after pretraining
sbatch --time=10:00:00 --array=1-10%1 --job-name=fine5_dc_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19558 --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml --model_name minkunet_pretrain_depthcontrast_waymo10 --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-10%1 --job-name=fine5_dc_lidaraug_dethead_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19559 --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_dethead_0p8w.yaml --model_name minkunet_pretrain_depthcontrast_waymo10_lidar_aug_dethead_0p8w --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets

sbatch --time=10:00:00 --array=1-10%1 --job-name=fine5_sc_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19560 --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml --model_name minkunet_pretrain_segcontrast_waymo10 --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets --frame_sampling_div 5
sbatch --time=10:00:00 --array=1-10%1 --job-name=fine5_sc_lidaraug_dethead_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19561 --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_dethead_0p8w.yaml --model_name minkunet_pretrain_segcontrast_waymo10_lidar_aug_dethead_0p8w --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets --frame_sampling_div 5
sbatch --time=10:00:00 --array=1-10%1 --job-name=fine5_sc_lidaraug_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19562 --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug.yaml --model_name minkunet_pretrain_segcontrast_waymo10_lidar_aug --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets --frame_sampling_div 5
sbatch --time=10:00:00 --array=1-10%1 --job-name=fine5_sc_dethead_waymo10_50epochs scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19563 --cfg_file configs/waymo_minkunet_segcontrast_waymo10_dethead_0p8w.yaml --model_name minkunet_pretrain_segcontrast_waymo10_dethead_0p8w --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets --frame_sampling_div 5

### Try 2
sbatch --time=10:00:00 --array=1-6%1 --job-name=dc_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19564 --cfg_file configs/waymo_minkunet_depthcontrast_waymo14.yaml --model_name minkunet_pretrain_depthcontrast_waymo14_try2 --pretrained_ckpt checkpoint-ep99.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-7%1 --job-name=dc_lidaraug_dethead_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19565 --cfg_file configs/waymo_minkunet_depthcontrast_waymo14_lidar_aug_dethead_0p8w.yaml --model_name minkunet_pretrain_depthcontrast_waymo14_lidar_aug_dethead_0p8w_try2 --pretrained_ckpt checkpoint-ep49.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-6%1 --job-name=sc_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19566 --cfg_file configs/waymo_minkunet_segcontrast_waymo14.yaml --model_name minkunet_pretrain_segcontrast_waymo14_try2 --pretrained_ckpt checkpoint-ep99.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-7%1 --job-name=sc_lidaraug_dethead_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19567 --cfg_file configs/waymo_minkunet_segcontrast_waymo14_lidar_aug_dethead_0p8w.yaml --model_name minkunet_pretrain_segcontrast_waymo14_lidar_aug_dethead_0p8w_try2 --pretrained_ckpt checkpoint-ep99.pth.tar --other_datasets
# sbatch --time=10:00:00 --array=1-7%1 --job-name=sc_lidaraug_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19568 --cfg_file configs/waymo_minkunet_segcontrast_waymo14_lidar_aug.yaml --model_name minkunet_pretrain_segcontrast_waymo14_lidar_aug_try2 --pretrained_ckpt checkpoint-ep99.pth.tar --other_datasets
# sbatch --time=10:00:00 --array=1-6%1 --job-name=sc_dethead_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19569 --cfg_file configs/waymo_minkunet_segcontrast_waymo14_dethead_0p8w.yaml --model_name minkunet_pretrain_segcontrast_waymo14_dethead_0p8w_try2 --pretrained_ckpt checkpoint-ep99.pth.tar --other_datasets
sbatch --time=10:00:00 --array=1-6%1 --job-name=scratch_seg_try2 scripts/submit_ddp_$CLUSTER_NAME.sh --mode scratch --tcp_port 19570 --cfg_file configs/waymo_scratch_minkunet.yaml --finetune_epochs 15 --other_datasets --downstream_model_dir scratch_epochs15_try2

