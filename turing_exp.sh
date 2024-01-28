#Pretrain-finetune
setsid nohup scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1 --tcp_port 19832 --cfg_file configs/waymo_minkunet_test.yaml --pretrain_batchsize_per_gpu 12 --pretrain_epochs 50 --finetune_epochs 5 --finetune_batchsize_per_gpu 10 --model_name minkunet_pretrain_test --pretrained_ckpt checkpoint-ep4.pth.tar > ./output/log/mink_pretrain_test_$(date +%Y-%m-%d_%H:%M).out 2>&1 &

#linear-probe
setsid nohup scripts/submit_ddp_turing.sh --tcp_port 19833 --mode linearprobe --cfg_file configs/waymo_lpseg_minkunet_test.yaml --linearprobe_last_n_ckpts 3 --linearprobe_epochs 4 --linearprobe_batchsize_per_gpu 10 --model_name minkunet_pretrain_test > ./output/log/mink_lpseg_test_$(date +%Y-%m-%d_%H:%M).out 2>&1 &

#scratch 
#--model_name is already in cfg file
# in checkpoints/minkunet create minkunet_scratch/pretrain_waymo dirs and copy checkpoint-0 from segcontrast
setsid nohup scripts/submit_ddp_turing.sh --cuda_visible_devices 2,3 --tcp_port 19834 --mode scratch --cfg_file configs/waymo_scratch_minkunet_test.yaml --finetune_epochs 50 --finetune_batchsize_per_gpu 10 --pretrained_ckpt checkpoint-ep0.pth.tar --downstream_model_dir scratch_test1 > ./output/log/mink_scratch_test_$(date +%Y-%m-%d_%H:%M).out 2>&1 &

# To kill a GPU process
# nvidia-smi
# check for PID
# kill PID

tmux 
#done
scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19832 --cfg_file configs/waymo_minkunet_segcontrast_waymo20.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_segcontrast_waymo20 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_segcontrast_waymo20_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19833 --cfg_file configs/waymo_pointrcnn_segcontrast_waymo20.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name pointrcnn_pretrain_segcontrast_waymo20 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/pointrcnn_pretrain_segcontrast_waymo20_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19832 --cfg_file configs/waymo_minkunet_esf_perc_0p05_iou_wt_waymo5.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_esf_perc_0p05_iou_wt_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_esf_perc_0p05_iou_wt_waymo5_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19833 --cfg_file configs/waymo_minkunet_esf_perc_0p05_esf_wt_waymo5.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_esf_perc_0p05_esf_wt_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_esf_perc_0p05_esf_wt_waymo5_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19832 --cfg_file configs/waymo_minkunet_esf_perc_0p05_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_esf_perc_0p05_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_esf_perc_0p05_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19832 --cfg_file configs/waymo_minkunet_esf_perc_0p05_waymo5_t0p07.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_esf_perc_0p05_waymo5_t0p07 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_esf_perc_0p05_waymo5_t0p07_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19833 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo5_t0p07.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_perc_0p01_waymo5_t0p07 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_perc_0p01_waymo5_t0p07_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_lovelace.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19832 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_perc_0p01_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_perc_0p01_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_lovelace.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19833 --cfg_file configs/waymo_minkunet_iou_perc_0p01_esf_wt_waymo5.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_perc_0p01_esf_wt_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_perc_0p01_esf_wt_waymo5_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19832 --cfg_file configs/waymo_minkunet_iou_perc_0p01_iou_wt_waymo5.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_perc_0p01_iou_wt_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_perc_0p01_iou_wt_waymo5_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19831 --cfg_file configs/waymo_minkunet_iou_perc_0p01_waymo5.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_perc_0p01_waymo5 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_perc_0p01_waymo5_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_lovelace.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19833 --cfg_file configs/waymo_pointrcnn_segcontrast_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name pointrcnn_pretrain_segcontrast_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/pointrcnn_pretrain_segcontrast_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_lovelace.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19834 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_segcontrast_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_segcontrast_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19834 --cfg_file configs/waymo_minkunet_segcontrast_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_segcontrast_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_segcontrast_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19835 --cfg_file configs/waymo_minkunet_iou_filter_0p15_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_filter_0p15_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_filter_0p15_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19836 --cfg_file configs/waymo_minkunet_iou_filter_0p03_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_filter_0p03_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_filter_0p03_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_lovelace.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19835 --cfg_file configs/waymo_minkunet_iou_filter_0p15_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_filter_0p15_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_filter_0p15_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_lovelace.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19836 --cfg_file configs/waymo_minkunet_iou_filter_0p03_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_filter_0p03_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_filter_0p03_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19835 --cfg_file configs/waymo_minkunet_gt_segcontrast_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_gt_segcontrast_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_gt_segcontrast_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19836 --cfg_file configs/waymo_minkunet_iou_knn_0p15_iou_owt_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_knn_0p15_iou_owt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_knn_0p15_iou_owt_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_lovelace.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19835 --cfg_file configs/waymo_minkunet_gt_segcontrast_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_gt_segcontrast_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_gt_segcontrast_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_lovelace.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19836 --cfg_file configs/waymo_minkunet_iou_knn_0p15_iou_owt_waymo5_t0p04.yaml --pretrain_batchsize_per_gpu 16 --finetune_batchsize_per_gpu 8 --model_name minkunet_pretrain_iou_knn_0p15_iou_owt_waymo5_t0p04 --pretrained_ckpt checkpoint-ep49.pth.tar > ./output/log/minkunet_pretrain_iou_knn_0p15_iou_owt_waymo5_t0p04_$(date +%Y-%m-%d_%H:%M).out 2>&1


To detach: ctrl+b and then d
tmux rename-session -t 0 mink_pretrain_test
tmux attach -t mink_pretrain_test
To kill a process: ctrl+c
after training is over: exit 

# To see hanging threads
#ps -ef | grep -i '[p]ython'
#pids=$(ps aux | grep 'nisarbar' | grep 'python' | awk '{print $2}')
#echo "$pids" | xargs kill


