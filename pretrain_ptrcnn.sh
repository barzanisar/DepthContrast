sbatch --time=6:00:00 --array=1-10%1 --job-name=esf_perc_0p05 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19210 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05.yaml --model_name pointrcnn_pretrain_backbone_esf_perc_0p05 --pretrained_ckpt checkpoint-ep49.pth.tar 
sbatch --time=6:00:00 --array=1-10%1 --job-name=esf_perc_0p01 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19211 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p01.yaml --model_name pointrcnn_pretrain_backbone_esf_perc_0p01 --pretrained_ckpt checkpoint-ep49.pth.tar

sbatch --time=6:00:00 --array=1-10%1 --job-name=iou_perc_0p05 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19212 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p05.yaml --model_name pointrcnn_pretrain_backbone_iou_perc_0p05 --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=6:00:00 --array=1-10%1 --job-name=iou_perc_0p01 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19213 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p01.yaml --model_name pointrcnn_pretrain_backbone_iou_perc_0p01 --pretrained_ckpt checkpoint-ep49.pth.tar

sbatch --time=6:00:00 --array=1-10%1 --job-name=iou_perc_0p05_iou_wt_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19214 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p05_iou_wt.yaml --model_name pointrcnn_pretrain_backbone_iou_perc_0p05_iou_wt --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=6:00:00 --array=1-10%1 --job-name=iou_perc_0p05_esf_wt_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19215 --cfg_file configs/waymo_pointrcnn_backbone_iou_perc_0p05_esf_wt.yaml --model_name pointrcnn_pretrain_backbone_iou_perc_0p05_esf_wt --pretrained_ckpt checkpoint-ep49.pth.tar

sbatch --time=6:00:00 --array=1-10%1 --job-name=esf_perc_0p05_iou_wt_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19216 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05_iou_wt.yaml --model_name pointrcnn_pretrain_backbone_esf_perc_0p05_iou_wt --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=6:00:00 --array=1-10%1 --job-name=esf_perc_0p05_esf_wt_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19217 --cfg_file configs/waymo_pointrcnn_backbone_esf_perc_0p05_esf_wt.yaml --model_name pointrcnn_pretrain_backbone_esf_perc_0p05_esf_wt --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=6:00:00 --array=1-10%1 --job-name=segcontrast_w5 scripts/submit_ddp_$CLUSTER_NAME.sh --tcp_port 19217 --cfg_file configs/waymo_pointrcnn_segcontrast.yaml --model_name pointrcnn_pretrain_segcontrast_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar
