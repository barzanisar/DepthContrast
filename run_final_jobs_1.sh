# HEREEEEEEEEEEEEEEEEEEEEE - DONE!-lovelace
# pretrain on 2 gpus with 32 bs each gpu for 200 epochs (old commit works since using two gpus)
scripts/submit_ddp_turing_pretrain_nuscenes.sh --tcp_port 18840 --mode p  \
    --cuda_visible_devices 0,1 \
    --model_name nuscenes_sweep1_eps0p3_minkunet_segcontrast  \
    --pretrain_epochs 200 \
    --pretrain_extra_tag 200ep_try0 \
    --workers_per_gpu 8 \
    > ./output/log/nuscenes_sweep1_eps0p3_minkunet_segcontrast_pretrain_ep200$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - DONE!-lovelace
scripts/submit_ddp_turing_pretrain_nuscenes.sh --tcp_port 18841 --mode p  \
    --cuda_visible_devices 2,3 \
    --model_name nuscenes_sweep1_eps0p3_minkunet_segcontrast_det  \
    --pretrain_epochs 200 \
    --pretrain_extra_tag 200ep_try0 \
    --workers_per_gpu 8 \
    > ./output/log/nuscenes_sweep1_eps0p3_minkunet_segcontrast_det_pretrain_ep200$(date +%Y-%m-%d_%H:%M).out 2>&1


# HEREEEEEEEEEEEEEEEEEEEEE - RUNNING -turing - ALSO optimizer, single gpu, train shuffle on and drop last in val false!
scripts/submit_ddp_turing_pretrain_nuscenes_also.sh --mode f  \
    --cuda_visible_devices 3 \
    --model_name nuscenes_sweep1_eps0p3_minkunet_segcontrast_det  \
    --pretrain_epochs 200 \
    --pretrain_extra_tag 200ep_try0 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag bs8_try00_also \
    --workers_per_gpu 8 \
    > ./output/log/nuscenes_sweep1_eps0p3_minkunet_segcontrast_det_ep200_fine-0p1-1-10-50-100perc_bs8_also_optim_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - RUNNING -turing - ALSO optimizer, single gpu, train shuffle on and drop last in val false!
scripts/submit_ddp_turing_pretrain_nuscenes_also_wo_0p1.sh --mode f  \
    --cuda_visible_devices 2 \
    --model_name nuscenes_sweep1_eps0p3_minkunet_segcontrast  \
    --pretrain_epochs 200 \
    --pretrain_extra_tag 200ep_try0 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag bs8_try0_also \
    --workers_per_gpu 8 \
    > ./output/log/nuscenes_sweep1_eps0p3_minkunet_segcontrast_ep200_fine-1-10-50-100perc_bs8_also_optim_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO: select best num epochs for 0.1 percent and finetune SC+det 5 times on each perc 0.1,1,10,50,100
#TODO: finetune SC 5 times on each perc 0.1,1,10,50,100
###############################

#Hyper param sensitivity on lidar aug prob with eps 0.2:
# HEREEEEEEEEEEEEEEEEEEEEE - DONE-turing! pretrain on 2 gpus, 30 epochs on 10% waymo, total_bs 32 - fintune on 1% wns for 15 epochs 2 gpus, total_bs 16 (with drop last) 
echo "SC lidar+det lidar p32_0p3"
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode pf --datasets wns --extra_tag try0_drop \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p3_eps0p2.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p3_eps0p2  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p3_eps0p2_fine1_15epochs_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - DONE-turing!
echo "SC lidar+det lidar p32_0p5"
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode pf --datasets wns --extra_tag try0_drop \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p5_eps0p2.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p5_eps0p2  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p5_eps0p2_fine1_15epochs_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - DONE-turing!
echo "SC lidar+det lidar p32_0p6"
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode pf --datasets wns --extra_tag try0_drop \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p6_eps0p2.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p6_eps0p2  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p6_eps0p2_fine1_15epochs_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO: repeat these above exp again (low priority)
###############################
#Hyper param sensitivity on cluster_eps: pretrain on 2 gpus, 30 epochs on 10% waymo, total_bs 32 - fintune on 1% wns for 15 epochs 2 gpus, total_bs 16 (with drop last) 
# HEREEEEEEEEEEEEEEEEEEEEE - DONE-lovelace! finetuning on lovelace wit drop last
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode pf --datasets wns --extra_tag try0_drop \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4_eps0p1.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps0p1  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps0p1_fine1_15epochs_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - DONE!-turing
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode pf --datasets wns --extra_tag try0_drop \
    --cuda_visible_devices 1,2  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4_eps0p2.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps0p2  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps0p2_fine1_15epochs_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - DONE-turing!
scripts/submit_ddp_turing.sh --tcp_port 18841 --mode pf --datasets wns --extra_tag try0_drop \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4_eps0p3.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps0p3  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps0p3_fine1_15epochs_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - TODO-turing! -> generating clusters in progress on lovelace
scripts/submit_ddp_turing.sh --tcp_port 18841 --mode pf --datasets wns --extra_tag try0_drop \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4_eps0p4.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps0p4  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps0p4_fine1_15epochs_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO: repeat these above exp again (low priority)
#####################################################
## Rebuttal experiments redo on 2 GPUS!
# HEREEEEEEEEEEEEEEEEEEEEE - DONE-turing
# redo scratch and segcontrast finetuning experiments on 5% wns for 30, 100, 100 epochs using 2 gpus this time (before we used a single gpu with train shuffling false so buggy) and drop last
echo "SC and scratch finetuning on 5% waymo, 30 epochs, 2 gpus (drop last)"
scripts/submit_ddp_turing.sh --tcp_port 18842 --mode fs --datasets w --extra_tag try0_2gpus_drop \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml \
    --model_name segcontrast_10perc_waymo_minkunet  \
    --finetune_epochs 30 \
    --frame_sampling_div 5 \
    --drop_last_val \
    > ./output/log/segcontrast_10perc_waymo_minkunet_fine5_30epochs_waymo_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - DONE-turing
echo "SC and scratch finetuning on 5% ns, 100 epochs, 2 gpus (drop last)"
scripts/submit_ddp_turing.sh --tcp_port 18842 --mode fs --datasets ns --extra_tag try0_2gpus_drop \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml \
    --model_name segcontrast_10perc_waymo_minkunet  \
    --finetune_epochs 100 \
    --frame_sampling_div 5 \
    --data_skip_ratio 20 \
    --val_after_epochs 50 \
    --drop_last_val \
    > ./output/log/segcontrast_10perc_waymo_minkunet_fine5_100epochs_ns_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - RUNNING-turing
echo "SC+lidar_det  finetuning on 5% waymo, 30 epochs, 2 gpus (drop last)"
scripts/submit_ddp_turing.sh --tcp_port 18842 --mode f --datasets w --extra_tag try0_2gpus_drop \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --finetune_epochs 30 \
    --frame_sampling_div 5 \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_fine5_30epochs_waymo_$(date +%Y-%m-%d_%H:%M).out 2>&1

# HEREEEEEEEEEEEEEEEEEEEEE - DONE-turing
echo "SC+lidar_det  finetuning on 5% ns, 100 epochs, 2 gpus (drop last)"
scripts/submit_ddp_turing.sh --tcp_port 18842 --mode f --datasets ns --extra_tag try0_2gpus_drop \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --finetune_epochs 100 \
    --frame_sampling_div 5 \
    --data_skip_ratio 20 \
    --val_after_epochs 50 \
    --drop_last_val \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_fine5_100epochs_ns_drop_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO: repeat these above exp again
################# HERE