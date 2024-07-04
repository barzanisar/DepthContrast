#Debug
scripts/submit_ddp_turing.sh --mode d

    # In progress
    # seg fine 1 % for 100 epochs
    # seglidar fine 1% for 100 epochs
    
    # #Done
    # fine all ablations on 1 % for 15 epochs
    
    #To do after pretraining ends
    # fine prop, segreg, dc, dc_lidar for 5% for 15 epochs
    

# Ablations

# 1. Polar Mix
# in progress on lovelace
# scripts/submit_ddp_turing.sh --mode pf --datasets wns --extra_tag try0 \
#     --cuda_visible_devices 0,1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_mixed.yaml \
#     --model_name segcontrast_lidaraug_mixed_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidaraug_mixed_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#not needed
# scripts/submit_ddp_turing.sh --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
#     --cuda_visible_devices 0,1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_mixed.yaml \
#     --model_name segcontrast_lidaraug_mixed_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidaraug_mixed_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# #not needed
# scripts/submit_ddp_turing.sh --mode f --datasets wns --extra_tag try0 --finetune_epochs 100 \
#     --cuda_visible_devices 0,1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_mixed.yaml \
#     --model_name segcontrast_lidaraug_mixed_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidaraug_mixed_ep200_1percfine100epochs$(date +%Y-%m-%d_%H:%M).out 2>&1

# 2. Single Pattern
# scripts/submit_ddp_turing.sh --tcp_port 18820 --mode pf --datasets wns --extra_tag try0 \
#     --cuda_visible_devices 0,1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single.yaml \
#     --model_name segcontrast_lidaraug_single_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidaraug_single_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#not needed
# scripts/submit_ddp_turing.sh --tcp_port 18820 --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
#     --cuda_visible_devices 0,1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single.yaml \
#     --model_name segcontrast_lidaraug_single_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidaraug_single_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

#not needed
# scripts/submit_ddp_turing.sh --tcp_port 18820 --mode f --datasets wns --extra_tag try0 --finetune_epochs 100 \
#     --cuda_visible_devices 0,1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single.yaml \
#     --model_name segcontrast_lidaraug_single_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidaraug_single_ep200_1percfine100epochs$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 2,3  --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single.yaml  --model_name minkunet_pretrain_segcontrast_waymo10_lidar_aug_single_ep200  > ./output/log/sc_lidaraug_single_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1


# 3. Single branch det
# scripts/submit_ddp_turing.sh --tcp_port 18830 --mode pf --datasets wns --extra_tag try0 \
#     --cuda_visible_devices 2,3  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml \
#     --model_name segcontrast_lidar_det_single_branch_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidar_det_single_branch_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#not needed
# scripts/submit_ddp_turing.sh --tcp_port 18830 --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
#     --cuda_visible_devices 2,3  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml \
#     --model_name segcontrast_lidar_det_single_branch_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidar_det_single_branch_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# #not needed
# scripts/submit_ddp_turing.sh --tcp_port 18830 --mode f --datasets wns --extra_tag try0 --finetune_epochs 100 \
#     --cuda_visible_devices 2,3  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml \
#     --model_name segcontrast_lidar_det_single_branch_10perc_waymo_minkunet  \
#     > ./output/log/sc_lidar_det_single_branch_ep200_1percfine100epochs$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml  --model_name minkunet_pretrain_segcontrast_waymo10_lidar_det_single_branch  > ./output/log/sc_lidar_det_singlehead_ep200$(date +%Y-%m-%d_%H:%M).out 2>&1

#4. Supervised SC #TODO

scripts/submit_ddp_turing.sh --tcp_port 18870 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_gt_segcontrast_waymo10.yaml \
    --model_name gt_segcontrast_10perc_waymo_minkunet  \
    > ./output/log/gt_segcontrast_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_gt_segcontrast_waymo7p5.yaml  --model_name minkunet_pretrain_gt_segcontrast_waymo7p5  --frame_sampling_div 5 > ./output/log/sc_gt_fine5_ep200$(date +%Y-%m-%d_%H:%M).out 2>&1

#5. Proposal contrast 
#IN progress
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml \
    --model_name proposalcontrast_10perc_waymo_minkunet  \
    > ./output/log/proposalcontrast_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --tcp_port 18840 --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml \
    --model_name proposalcontrast_10perc_waymo_minkunet  \
    > ./output/log/proposalcontrast_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml  --model_name minkunet_pretrain_proposalcontrast_waymo10  > ./output/log/pc_ep200$(date +%Y-%m-%d_%H:%M).out 2>&1

#6. Scale/rot reg Head
#IN progress
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml \
    --model_name segcontrast_reghead_10perc_waymo_minkunet  \
    > ./output/log/segcontrast_reghead_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --tcp_port 18850 --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml \
    --model_name segcontrast_reghead_10perc_waymo_minkunet  \
    > ./output/log/segcontrast_reghead_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml  --model_name minkunet_pretrain_segcontrast_waymo10_reghead  --frame_sampling_div 5 > ./output/log/screg_fine5_ep200$(date +%Y-%m-%d_%H:%M).out 2>&1

# 7. DC
#In progress
scripts/submit_ddp_turing.sh --tcp_port 18850 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml \
    --model_name depthcontrast_10perc_waymo_minkunet  \
    > ./output/log/depthcontrast_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml \
    --model_name depthcontrast_10perc_waymo_minkunet  \
    > ./output/log/depthcontrast_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 2,3  --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml --pretrain_epochs 200 --model_name minkunet_pretrain_depthcontrast_waymo10_ep200  > ./output/log/dc_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1


#8. DC+laug+det
#IN progress
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name depthcontrast_lidaraug_det_10perc_waymo_minkunet  \
    > ./output/log/depthcontrast_lidaraug_det_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name depthcontrast_lidaraug_det_10perc_waymo_minkunet  \
    > ./output/log/depthcontrast_lidaraug_det_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml --pretrain_epochs 200 --model_name minkunet_pretrain_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w_ep200  > ./output/log/dc_lidar_s_randh_det0p5_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#9. SC

#In progress
scripts/submit_ddp_turing.sh --tcp_port 18870 --mode f --datasets wns --extra_tag try0 --finetune_epochs 100 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml \
    --model_name segcontrast_10perc_waymo_minkunet  \
    > ./output/log/sc_ep200_1percfine100epochs$(date +%Y-%m-%d_%H:%M).out 2>&1

#In progress
scripts/submit_ddp_turing.sh --tcp_port 18890 --mode f --datasets wns --extra_tag try0 --finetune_epochs 100 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    > ./output/log/sc_lidarplusdet_ep200_1percfine100epochs$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --cuda_visible_devices 2,3  --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml --pretrain_epochs 200 --model_name minkunet_pretrain_segcontrast_waymo10_ep200_t2  > ./output/log/sc_ep200_t2_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml --pretrain_epochs 200 --model_name minkunet_pretrain_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w_ep200_t2  > ./output/log/sc_lidar_s_randh_det0p5_ep200_t2_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_segcontrast_waymo10_dethead_0p5w.yaml --pretrain_epochs 200 --model_name minkunet_pretrain_segcontrast_waymo10_dethead_0p5w_ep200_t2  > ./output/log/sc_det0p5_ep200_t2_$(date +%Y-%m-%d_%H:%M).out 2>&1
scripts/submit_ddp_turing.sh --cuda_visible_devices 2,3  --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh.yaml --pretrain_epochs 200 --model_name minkunet_pretrain_segcontrast_waymo10_lidar_aug_single_randh_ep200_t2  > ./output/log/sc_lidar_s_randh_ep200_t2_$(date +%Y-%m-%d_%H:%M).out 2>&1


#10. pretrain on waymo 50% for 50 epochs
scripts/submit_ddp_turing.sh --tcp_port 18960 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo50_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_50perc_waymo_minkunet  \
    --pretrain_epochs 50 \
    --pretrained_ckpt checkpoint-ep49.pth.tar \
    > ./output/log/sc_lidarplusdet_waymo50perc_ep50_$(date +%Y-%m-%d_%H:%M).out 2>&1
