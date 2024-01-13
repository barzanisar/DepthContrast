#Pretrain-finetune
sbatch --time=00:10:00 --job-name=mink_segcontrast_w3 scripts/submit_ddp_dgx.sh --tcp_port 19831 --cfg_file configs/waymo_minkunet_segcontrast_waymo3.yaml --pretrain_batchsize_per_gpu 6 --pretrain_epochs 50 --finetune_epochs 15 --finetune_batchsize_per_gpu 10 --model_name minkunet_pretrain_segcontrast_waymo3 --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=00:10:00 --job-name=mink_iou_perc_0p01_w3 scripts/submit_ddp_dgx.sh --tcp_port 19832 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo3.yaml --pretrain_batchsize_per_gpu 6 --pretrain_epochs 50 --finetune_epochs 15 --finetune_batchsize_per_gpu 10 --model_name minkunet_pretrain_iou_perc_0p01_waymo3 --pretrained_ckpt checkpoint-ep49.pth.tar
sbatch --time=01:00:00 --job-name=mink_pretrain_test scripts/submit_ddp_dgx.sh --tcp_port 19832 --cfg_file configs/waymo_minkunet_test.yaml --pretrain_batchsize_per_gpu 12 --pretrain_epochs 5 --finetune_epochs 5 --finetune_batchsize_per_gpu 10 --model_name minkunet_pretrain_test --pretrained_ckpt checkpoint-ep4.pth.tar

#linear-probe
sbatch --time=00:10:00 --job-name=mink_lpseg_iou_perc_0p01_w3 scripts/submit_ddp_dgx.sh --tcp_port 19833 --mode linearprobe --cfg_file configs/waymo_lpseg_minkunet.yaml --linearprobe_last_n_ckpts 10 --linearprobe_epochs 10 --linearprobe_batchsize_per_gpu 20 --model_name minkunet_pretrain_iou_perc_0p01_waymo3 
sbatch --time=01:00:00 --job-name=mink_lpseg_test scripts/submit_ddp_dgx.sh --tcp_port 19833 --mode linearprobe --cfg_file configs/waymo_lpseg_minkunet_test.yaml --linearprobe_last_n_ckpts 3 --linearprobe_epochs 4 --linearprobe_batchsize_per_gpu 10 --model_name minkunet_pretrain_test 

#scratch 
#--model_name is already in cfg file
# in checkpoints/minkunet create minkunet_scratch/pretrain_waymo dirs and copy checkpoint-0 from segcontrast
sbatch --time=00:10:00 --job-name=mink_scratch_w1 scripts/submit_ddp_dgx.sh --tcp_port 19834 --mode scratch --cfg_file configs/waymo_scratch_minkunet.yaml --finetune_epochs 15 --finetune_batchsize_per_gpu 10 --pretrained_ckpt checkpoint-ep0.pth.tar --downstream_model_dir scratch_epochs15
sbatch --time=01:00:00 --job-name=mink_scratch_test scripts/submit_ddp_dgx.sh --tcp_port 19834 --mode scratch --cfg_file configs/waymo_scratch_minkunet_test.yaml --finetune_epochs 5 --finetune_batchsize_per_gpu 10 --pretrained_ckpt checkpoint-ep0.pth.tar --downstream_model_dir scratch_test
