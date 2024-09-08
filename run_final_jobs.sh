#Debug
scripts/submit_ddp_turing.sh --mode d

    # In progress
    # seg fine 1 % for 100 epochs
    # seglidar fine 1% for 100 epochs
    
    # #Done
    # fine all ablations on 1 % for 15 epochs
    
    #In progress
    #dc_lidar 1% and 5% for 15 epochs on turing and lovelace respectively

    #To do after pretraining ends
    # fine prop, segreg on 5% for 15 epochs
    
#Important:
#seglidar 1% for 15 epochs, and 5% for 15 epochs - try2
#segdet 1% for 15 epochs, and 5% for 15 epochs - try2
#seglidarplusdet 1% for 15 epochs, and 5% for 15 epochs - try2
#dc 1% for 15 epochs, and 5% for 15 epochs - try1
#-->dclidarplusdet 1% for 15 epochs, and 5% for 15 epochs - try1
#segreg 1% for 15 epochs, and 5% for 15 epochs - try0 and try1
#proposalcontrast 1% for 15 epochs, and 5% for 15 epochs - try0 and try1
#seg lidar mixed, seg lidar single, seg lidar single det branch, on 1% for 15 epochs - try1
#Others:
# seg fine 1 % for 100 epochs-try1
# seglidarplusdet fine 1% for 100 epochs-try1



# Ablations

# 1. Polar Mix
##IN progress lovelace
scripts/submit_ddp_turing.sh --tcp_port 18821 --mode f --datasets wns --extra_tag try1 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_mixed.yaml \
    --model_name segcontrast_lidaraug_mixed_10perc_waymo_minkunet  \
    > ./output/log/sc_lidaraug_mixed_ep200_try1_$(date +%Y-%m-%d_%H:%M).out 2>&1

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
##IN progress lovelace
scripts/submit_ddp_turing.sh --tcp_port 18821 --mode f --datasets wns --extra_tag try1 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single.yaml \
    --model_name segcontrast_lidaraug_single_10perc_waymo_minkunet  \
    > ./output/log/sc_lidaraug_single_ep200_try1_$(date +%Y-%m-%d_%H:%M).out 2>&1

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
##IN progress lovelace
scripts/submit_ddp_turing.sh --tcp_port 18821 --mode f --datasets wns --extra_tag try1 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml \
    --model_name segcontrast_lidar_det_single_branch_10perc_waymo_minkunet  \
    > ./output/log/sc_lidar_det_single_branch_ep200_try1_$(date +%Y-%m-%d_%H:%M).out 2>&1

#not needed
echo "sc+ lidar + single det-finetune 5%"
scripts/submit_ddp_turing.sh --tcp_port 18930 --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml \
    --model_name segcontrast_lidar_det_single_branch_10perc_waymo_minkunet  \
    > ./output/log/sc_lidar_det_single_branch_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

#not needed
echo "sc+ lidar + single det-finetune 1% for 100 epochs"
scripts/submit_ddp_turing.sh --tcp_port 18940 --mode f --datasets wns --extra_tag try0 --finetune_epochs 100 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml \
    --model_name segcontrast_lidar_det_single_branch_10perc_waymo_minkunet  \
    > ./output/log/sc_lidar_det_single_branch_ep200_1percfine100epochs$(date +%Y-%m-%d_%H:%M).out 2>&1

#not needed
echo "sc+ lidar + single det-finetune 5%"
scripts/submit_ddp_turing.sh --tcp_port 18950 --mode f --datasets wns --extra_tag try1 --frame_sampling_div 5 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml \
    --model_name segcontrast_lidar_det_single_branch_10perc_waymo_minkunet  \
    > ./output/log/sc_lidar_det_single_branch_ep200_5percfine_try1$(date +%Y-%m-%d_%H:%M).out 2>&1

#not needed
echo "sc+ lidar + single det-finetune 1% for 100 epochs"
scripts/submit_ddp_turing.sh --tcp_port 18960 --mode f --datasets wns --extra_tag try1 --finetune_epochs 100 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_det_single_branch.yaml \
    --model_name segcontrast_lidar_det_single_branch_10perc_waymo_minkunet  \
    > ./output/log/sc_lidar_det_single_branch_ep200_1percfine100epochs_try1$(date +%Y-%m-%d_%H:%M).out 2>&1

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

#TODO
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode f --datasets wn --extra_tag try0 --frame_sampling_div 5 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml \
    --model_name proposalcontrast_10perc_waymo_minkunet  \
    > ./output/log/proposalcontrast_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode f --datasets wns --extra_tag try1 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml \
    --model_name proposalcontrast_10perc_waymo_minkunet  \
    > ./output/log/proposalcontrast_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
scripts/submit_ddp_turing.sh --tcp_port 18850 --mode f --datasets wn --extra_tag try1 --frame_sampling_div 5 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml \
    --model_name proposalcontrast_10perc_waymo_minkunet  \
    > ./output/log/proposalcontrast_ep200_5percfine_try1$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml  --model_name minkunet_pretrain_proposalcontrast_waymo10  > ./output/log/pc_ep200$(date +%Y-%m-%d_%H:%M).out 2>&1

#6. Scale/rot reg Head

#IN progress
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml \
    --model_name segcontrast_reghead_10perc_waymo_minkunet  \
    > ./output/log/segcontrast_reghead_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
scripts/submit_ddp_turing.sh --tcp_port 18850 --mode f --datasets wns --extra_tag try0 --frame_sampling_div 5 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml \
    --model_name segcontrast_reghead_10perc_waymo_minkunet  \
    > ./output/log/segcontrast_reghead_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
scripts/submit_ddp_turing.sh --tcp_port 18840 --mode f --datasets wns --extra_tag try1 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml \
    --model_name segcontrast_reghead_10perc_waymo_minkunet  \
    > ./output/log/segcontrast_reghead_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
scripts/submit_ddp_turing.sh --tcp_port 18850 --mode f --datasets wns --extra_tag try1 --frame_sampling_div 5 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml \
    --model_name segcontrast_reghead_10perc_waymo_minkunet  \
    > ./output/log/segcontrast_reghead_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml  --model_name minkunet_pretrain_segcontrast_waymo10_reghead  --frame_sampling_div 5 > ./output/log/screg_fine5_ep200$(date +%Y-%m-%d_%H:%M).out 2>&1

# 7. DC

#in progress
scripts/submit_ddp_turing.sh --tcp_port 18850 --mode pf --datasets wns --extra_tag try1 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml \
    --model_name depthcontrast_10perc_waymo_minkunet  \
    > ./output/log/depthcontrast_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#in progress
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets wns --extra_tag try1 --frame_sampling_div 5 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml \
    --model_name depthcontrast_10perc_waymo_minkunet  \
    > ./output/log/depthcontrast_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 2,3  --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml --pretrain_epochs 200 --model_name minkunet_pretrain_depthcontrast_waymo10_ep200  > ./output/log/dc_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1


#8. DC+laug+det
#in progress
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode pf --datasets wns --extra_tag try1 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name depthcontrast_lidaraug_det_10perc_waymo_minkunet  \
    > ./output/log/depthcontrast_lidaraug_det_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#in progress
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets wns --extra_tag try1 --frame_sampling_div 5 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name depthcontrast_lidaraug_det_10perc_waymo_minkunet  \
    > ./output/log/depthcontrast_lidaraug_det_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --cuda_visible_devices 0,1  --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml --pretrain_epochs 200 --model_name minkunet_pretrain_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w_ep200  > ./output/log/dc_lidar_s_randh_det0p5_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#9. SC

#TODO
#segdet, 1%, 15 epochs -try2
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets wns --extra_tag try2 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_dethead_0p5w.yaml \
    --model_name segcontrast_det_10perc_waymo_minkunet  \
    > ./output/log/sc_det_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
#segdet, 5%, 15 epochs -try2
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets w --extra_tag try2 --frame_sampling_div 5 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_dethead_0p5w.yaml \
    --model_name segcontrast_det_10perc_waymo_minkunet  \
    > ./output/log/sc_det_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1


#TODO
#seglidar, 1%, 15 epochs -try2
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets s --extra_tag try2 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh.yaml \
    --model_name segcontrast_lidaraug_single_randh_10perc_waymo_minkunet  \
    > ./output/log/sc_lidaraug_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
#seglidar, 5%, 15 epochs -try2
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets wns --extra_tag try2 --frame_sampling_div 5 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh.yaml \
    --model_name segcontrast_lidaraug_single_randh_10perc_waymo_minkunet  \
    > ./output/log/sc_lidaraug_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
#seglidarplusdet, 1%, 15 epochs -try2
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets wns --extra_tag try2 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    > ./output/log/sc_lidarplusdet_ep200_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
#seglidarplusdet, 5%, 15 epochs -try2
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode f --datasets wns --extra_tag try2 --frame_sampling_div 5 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    > ./output/log/sc_lidarplusdet_ep200_5percfine$(date +%Y-%m-%d_%H:%M).out 2>&1

#In progress
scripts/submit_ddp_turing.sh --tcp_port 18870 --mode f --datasets n --extra_tag try0 --finetune_epochs 100 \
    --cuda_visible_devices 2,3  \
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
#DONE after everything
scripts/submit_ddp_turing.sh --tcp_port 18960 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo50_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_50perc_waymo_minkunet  \
    --pretrain_epochs 50 \
    --pretrained_ckpt checkpoint-ep49.pth.tar \
    > ./output/log/sc_lidarplusdet_waymo50perc_ep50_$(date +%Y-%m-%d_%H:%M).out 2>&1


#pretrain on waymo 50% for 50 epochs
#DONE after everything
scripts/submit_ddp_turing.sh --tcp_port 18961 --mode f --datasets wns --extra_tag try0 \
    --cuda_visible_devices 0,1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo50_lidar_det_single_branch.yaml \
    --model_name segcontrast_lidaraug_det_single_branch_50perc_waymo_minkunet  \
    --pretrain_epochs 50 \
    --pretrained_ckpt checkpoint-ep49.pth.tar \
    > ./output/log/sc_lidar_single_det_waymo50perc_ep50_$(date +%Y-%m-%d_%H:%M).out 2>&1


#10. pretrain on waymo 50% for 100 epochs and finetune on 1% wns for 100 epochs  - try 0
#Pending after everything
scripts/submit_ddp_turing.sh --tcp_port 18961 --mode pf --datasets wns --extra_tag try0 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo50_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_50perc_waymo_minkunet  \
    --pretrain_epochs 100 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_epochs 100 \
    > ./output/log/sc_lidarplusdet_waymo50perc_ep100_$(date +%Y-%m-%d_%H:%M).out 2>&1

# pretrain on waymo 50% for 100 epochs and finetune on 1% wns for 100 epochs - try 1
# Pending
scripts/submit_ddp_turing.sh --tcp_port 18961 --mode f --datasets wns --extra_tag try1 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo50_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_50perc_waymo_minkunet  \
    --pretrain_epochs 100 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_epochs 100 \
    > ./output/log/sc_lidarplusdet_waymo50perc_ep100$(date +%Y-%m-%d_%H:%M).out 2>&1


# pretrain on waymo 50% for 100 epochs and finetune on 5% wns for 15 epochs - try 0
# Pending
scripts/submit_ddp_turing.sh --tcp_port 18961 --mode f --datasets wns --extra_tag try0 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo50_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_50perc_waymo_minkunet  \
    --pretrain_epochs 100 \
    --frame_sampling_div 5 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    > ./output/log/sc_lidarplusdet_waymo50perc_ep100_$(date +%Y-%m-%d_%H:%M).out 2>&1

# pretrain on waymo 50% for 100 epochs and finetune on 5% wns for 15 epochs - try 1
# Pending
scripts/submit_ddp_turing.sh --tcp_port 18961 --mode f --datasets wns --extra_tag try1 \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo50_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_50perc_waymo_minkunet  \
    --pretrain_epochs 100 \
    --frame_sampling_div 5 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    > ./output/log/sc_lidarplusdet_waymo50perc_ep100_$(date +%Y-%m-%d_%H:%M).out 2>&1

# To see hanging threads
#ps -ef | grep -i '[p]ython'
pids=$(ps aux | grep 'nisarbar' | grep 'python' | awk '{print $2}')
echo "$pids" | xargs kill





scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode f  \
    --cuda_visible_devices 0  \
    --model_name nuscenes_sweep1_minkunet_segcontrast_lidarplusdet  \
    --finetune_cfg_file nuscenes_fine1lr_minkunet\
    --extra_tag sgd_100ep_try0 \
    > ./output/log/nuscenes_sweep1_minkunet_segcontrast_lidarplusdet_ep200_alsofine_1percent$(date +%Y-%m-%d_%H:%M).out 2>&1

# #TODO:
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode p  \
    --cuda_visible_devices 2,3  \
    --model_name nuscenes_sweep3_minkunet_segcontrast_lidarplusdet  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrain_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --workers_per_gpu 8 \
    > ./output/log/nuscenes_sweep3_minkunet_segcontrast_lidarplusdet_ep100_$(date +%Y-%m-%d_%H:%M).out 2>&1

# dont have available gpus
scripts/submit_ddp_turing.sh --tcp_port 18860 --mode pf --datasets wns \
    --cuda_visible_devices 2,3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop32lidar.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_drop32lidar  \
    > ./output/log/sc_lidarplusdet_ep200_drop32lidar_$(date +%Y-%m-%d_%H:%M).out 2>&1

# 2gpus - works
sbatch --time=12:00:00 --gres=gpu:rtx6000:2 --array=1-10%1 --job-name=lidarplusdet_drop32lidar_rtx2gpus_100ep scripts/submit_vector_pretrain_waymo.sh --mode p \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop32lidar.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_drop32lidar \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 100

sbatch --time=12:00:00 --gres=gpu:a40:2 --array=1-10%1 --job-name=lidarplusdet_drop64lidar_rtx2gpus_100ep scripts/submit_vector_pretrain_waymo.sh --mode p \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop64lidar.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_drop64lidar \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 100

sbatch --time=12:00:00 --gres=gpu:rtx6000:2 --array=1-10%1 --job-name=lidarplusdet_dropalllidar_rtx2gpus_100ep scripts/submit_vector_pretrain_waymo.sh --mode p \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_dethead_0p5w.yaml \
    --model_name segcontrast_det_10perc_waymo_minkunet \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 100


# tune p32
sbatch --time=12:00:00 --gres=gpu:rtx6000:1 --array=1-2%1 --job-name=lidarplusdet_p32_0p3 scripts/submit_vector_pretrain_waymo.sh --mode p \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p3.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p3 \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 30

sbatch --time=12:00:00 --gres=gpu:rtx6000:1 --array=1-3%1 --job-name=lidarplusdet_p32_0p4 scripts/submit_vector_pretrain_waymo.sh --mode p \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4 \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 30

sbatch --time=12:00:00 --gres=gpu:rtx6000:1 --array=1-3%1 --job-name=lidarplusdet_p32_0p5 scripts/submit_vector_pretrain_waymo.sh --mode p \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p5.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p5 \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 30

sbatch --time=12:00:00 --gres=gpu:rtx6000:1 --array=1-3%1 --job-name=lidarplusdet_p32_0p6 scripts/submit_vector_pretrain_waymo.sh --mode p \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p6.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p6 \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 30


# scripts/submit_ddp_turing.sh --mode pf --datasets wns \
#     --cuda_visible_devices 1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_dethead_0p5w.yaml \
#     --model_name segcontrast_det_10perc_waymo_minkunet  \
#     --pretrain_bs_per_gpu 32 \
#     --pretrain_epochs 100 \
#     --workers_per_gpu 8 \
#     --finetune_bs_per_gpu 16 \
#     --pretrained_ckpt checkpoint-ep99.pth.tar \
#     > ./output/log/waymo_minkunet_segcontrast_waymo10_dethead_0p5w_drop_all_lidar_100ep_fine1percent_for_100ep_$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --mode pf --datasets wns \
#     --cuda_visible_devices 1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_dethead_0p5w.yaml \
#     --model_name segcontrast_det_10perc_waymo_minkunet  \
#     --pretrain_bs_per_gpu 32 \
#     --pretrain_epochs 100 \
#     --workers_per_gpu 8 \
#     --finetune_bs_per_gpu 16 \
#     --pretrained_ckpt checkpoint-ep99.pth.tar \
#     > ./output/log/waymo_minkunet_segcontrast_waymo10_dethead_0p5w_drop_all_lidar_100ep_fine1percent_for_100ep_$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --mode pf --datasets wns \
#     --cuda_visible_devices 3  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop32lidar.yaml \
#     --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_drop32lidar  \
#     --pretrain_bs_per_gpu 32 \
#     --pretrain_epochs 50 \
#     --workers_per_gpu 8 \
#     --finetune_bs_per_gpu 16 \
#     --pretrained_ckpt checkpoint-ep49.pth.tar \
#     > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop32lidar_50ep_fine1percent_for_100ep_$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing.sh --mode pf --datasets wns \
#     --cuda_visible_devices 1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop64lidar.yaml \
#     --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_drop64lidar  \
#     --pretrain_bs_per_gpu 32 \
#     --pretrain_epochs 50 \
#     --workers_per_gpu 8 \
#     --finetune_bs_per_gpu 16 \
#     --pretrained_ckpt checkpoint-ep49.pth.tar \
#     > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop64lidar_50ep_fine1percent_for_100ep_$(date +%Y-%m-%d_%H:%M).out 2>&1

scripts/submit_ddp_turing.sh --mode pf --datasets wns \
    --cuda_visible_devices 3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop32lidar.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_drop32lidar  \
    --pretrain_bs_per_gpu 32 \
    --pretrain_epochs 100 \
    --workers_per_gpu 8 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_drop32lidar_100ep_fine1percent_for_100ep_$(date +%Y-%m-%d_%H:%M).out 2>&1

### finetune our waymo model on all percentages of nuscenes
scripts/submit_ddp_turing_finetune_nuscenes.sh --mode f  \
    --cuda_visible_devices 3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 8 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_1 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_nuscenes_all_perc_try_1_$(date +%Y-%m-%d_%H:%M).out 2>&1


#TODO:
# #sweep3 lidar+det - 500 ep
# scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode pf  \
#     --cuda_visible_devices 0  \
#     --model_name nuscenes_sweep3_minkunet_segcontrast_lidarplusdet  \
#     --pretrain_epochs 100 \
#     --pretrain_extra_tag 100ep_try0 \
#     --pretrained_ckpt checkpoint-ep99.pth.tar \
#     --finetune_cfg_file nuscenes_fine1lr_minkunet_also_cfg \
#     --extra_tag also_cfg_try0 \
#     > ./output/log/nuscenes_sweep3_minkunet_segcontrast_lidarplusdet_ep100_fine1perc_500ep$(date +%Y-%m-%d_%H:%M).out 2>&1

# #sweep3 lidar+det - 250 ep finetune
# scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode pf  \
#     --cuda_visible_devices 0  \
#     --model_name nuscenes_sweep3_minkunet_segcontrast_lidarplusdet  \
#     --pretrain_epochs 100 \
#     --pretrain_extra_tag 100ep_try0 \
#     --pretrained_ckpt checkpoint-ep99.pth.tar \
#     --finetune_cfg_file nuscenes_fine1lr_minkunet \
#     --extra_tag try0 \
#     > ./output/log/nuscenes_sweep3_minkunet_segcontrast_lidarplusdet_ep100_fine1perc_100ep$(date +%Y-%m-%d_%H:%M).out 2>&1

# #sweep 1 det - 500 ep
# scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode pf  \
#     --cuda_visible_devices 1  \
#     --model_name nuscenes_sweep1_minkunet_segcontrast_det  \
#     --pretrain_epochs 100 \
#     --pretrain_extra_tag 100ep_try0 \
#     --pretrained_ckpt checkpoint-ep99.pth.tar \
#     --finetune_cfg_file nuscenes_fine1lr_minkunet_also_cfg \
#     --extra_tag also_cfg_try0 \
#     > ./output/log/nuscenes_sweep1_minkunet_segcontrast_det_ep100_fine1perc_500ep$(date +%Y-%m-%d_%H:%M).out 2>&1

#sweep 1, eps0.7 det -250 ep with 8 bs
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode f  \
    --cuda_visible_devices 2  \
    --model_name nuscenes_sweep1_minkunet_segcontrast_det  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_cfg_file nuscenes_fine1lr_minkunet\
    --extra_tag 250ep_bs8_try0 \
    --workers_per_gpu 6 \
    > ./output/log/nuscenes_sweep1_minkunet_segcontrast_det_ep100_fine1perc_250ep$(date +%Y-%m-%d_%H:%M).out 2>&1


#sweep 1, eps1 det -250 ep with 8 bs
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode pf  \
    --cuda_visible_devices 2  \
    --model_name nuscenes_sweep1_eps1_minkunet_segcontrast_det  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_cfg_file nuscenes_fine1lr_minkunet \
    --extra_tag 250ep_bs8_try0 \
    --workers_per_gpu 6 \
    > ./output/log/nuscenes_sweep1_eps1_minkunet_segcontrast_det_ep100_fine1perc_250ep$(date +%Y-%m-%d_%H:%M).out 2>&1


#sweep 1, eps0.4 det -250 ep with 8 bs
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode f  \
    --cuda_visible_devices 2 \
    --model_name nuscenes_sweep1_eps0p4_minkunet_segcontrast_det  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_cfg_file nuscenes_fine1lr_minkunet \
    --extra_tag 250ep_bs8_try0 \
    --workers_per_gpu 6 \
    > ./output/log/nuscenes_sweep1_eps0p4_minkunet_segcontrast_det_ep100_fine1perc_250ep$(date +%Y-%m-%d_%H:%M).out 2>&1




################# HERE
#15 - nus sweep1 seg ------- gpu 2
#ours p32-0.3 ----gpu 1 
#ours p32-0.6 ----gpu 3 
#scratch nusc1perc 100 ep bs 16 ----gpu 0

#gpu 0 -- ours on 10% semkitti for 200 epochs
#gpu 1-- nus sweep1 seg
#gpu 2 -- waymo 5% for 30 epochs scratch, seg, ours
#gpu 3 -- nusc 10% for 300 epochs and 100 epochs

#tmux 12: semkitti 1%, 30,40,50,70 epochs -> gpu2
#tmux 11: nusc 10%, 10,20,30,40 epochs -> gpu 0
#tmux 5: waymo 5% 30 epochs, seg -> gpu1
#tmux 13: semkitti 10% 10,20,30,40 epochs -> gpu3


echo "10 perc nusc 10, 20, 30, 40"
scripts/submit_ddp_turing_finetune_nuscenes_10.sh --mode f  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 8 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_nuscenes_10_perc_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1


FINETUNE_EPOCHS=(30 40 50 70)
for epoch in "${FINETUNE_EPOCHS[@]}"; do
    echo "1 perc semkitti $epoch epochs"
    scripts/submit_ddp_turing.sh --mode f --datasets s  \
        --cuda_visible_devices 2  \
        --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
        --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
        --workers_per_gpu 8 \
        --finetune_bs_per_gpu 16 \
        --pretrained_ckpt checkpoint-ep199.pth.tar \
        --extra_tag try_0 \
        --finetune_epochs $epoch \
        > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_1_perc_"$epoch"ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1
done

FINETUNE_EPOCHS=(10 20 30 40)
for epoch in "${FINETUNE_EPOCHS[@]}"; do
    echo "10 perc semkitti $epoch epochs"
    scripts/submit_ddp_turing.sh --mode f --datasets s  \
        --cuda_visible_devices 3  \
        --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
        --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
        --workers_per_gpu 8 \
        --finetune_bs_per_gpu 16 \
        --pretrained_ckpt checkpoint-ep199.pth.tar \
        --extra_tag try_0 \
        --frame_sampling_div 10 \
        --finetune_epochs $epoch \
        > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_10_perc_"$epoch"ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1
done



echo "waymo 5% 30 epochs, ours and then scratch"
scripts/submit_ddp_turing_finetune_5perc.sh --mode fs --datasets w \
    --cuda_visible_devices 2  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 6 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --finetune_epochs 30 \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_w_5_perc_30ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "waymo 5% 30 epochs, seg"
scripts/submit_ddp_turing_finetune_5perc.sh --mode f --datasets w \
    --cuda_visible_devices 2  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml \
    --model_name segcontrast_10perc_waymo_minkunet  \
    --workers_per_gpu 6 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --finetune_epochs 30 \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_finetune_w_5_perc_30ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1



echo "1 perc semkitti 100 epochs"
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --finetune_epochs 100 \
    --val_after_epochs 50 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_1_perc_100ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "1 perc semkitti 130 epochs"
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --finetune_epochs 130 \
    --val_after_epochs 50 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_1_perc_130ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#TODO
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode pf  \
    --cuda_visible_devices 0  \
    --model_name nuscenes_sweep3_minkunet_segcontrast_lidarplusdet  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_cfg_file nuscenes_fine1lr_minkunet \
    --extra_tag try0 \
    > ./output/log/nuscenes_sweep3_minkunet_segcontrast_lidarplusdet_ep100_fine1perc_100ep$(date +%Y-%m-%d_%H:%M).out 2>&1

#Done
scripts/submit_ddp_turing_pf_lidaraug.sh --mode f --datasets wns \
    --cuda_visible_devices 2  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4  \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 30 \
    --workers_per_gpu 8 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --finetune_epochs 15 \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4_finetune_wns_1_perc_15ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#Done
scripts/submit_ddp_turing_pf_lidaraug.sh --mode f --datasets wns \
    --cuda_visible_devices 3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p5.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p5  \
    --pretrain_bs_per_gpu 16 \
    --pretrain_epochs 30 \
    --workers_per_gpu 8 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    --finetune_epochs 15 \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p5_finetune_wns_1_perc_15ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#DONE
scripts/submit_ddp_turing_finetune_5perc.sh --mode f --datasets ns \
    --cuda_visible_devices 1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --finetune_epochs 100 \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_wns_5_perc_100ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#DONE
scripts/submit_ddp_turing_finetune_5perc.sh --mode f --datasets ns \
    --cuda_visible_devices 1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml \
    --model_name segcontrast_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --finetune_epochs 100 \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_finetune_wns_5_perc_100ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#DONE
scripts/submit_ddp_turing_finetune_5perc.sh --mode s --datasets ns \
    --cuda_visible_devices 3  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --finetune_epochs 100 \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_scratch_ns_5_perc_100ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

### finetune our waymo model on all percentages of nuscenes

# DONE but cannot beat bevcontrast at 700 ep, try 300 ep
scripts/submit_ddp_turing_finetune_nuscenes_0p1.sh --mode f  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_nuscenes_0p1_perc_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#DONE cant beat bevcontrast
scripts/submit_ddp_turing_finetune_nuscenes_10.sh --mode f  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_nuscenes_10_perc_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#Done ours on semkitti 10% for 100 epochs
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --frame_sampling_div 10 \
    --finetune_epochs 100 \
    --val_after_epochs 50 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_10_perc_100ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1



#RUNNING ours on semkitti 1% for 200 epochs -try 150 if 200 does not work
echo "1 perc semkitti 200 epochs"
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --finetune_epochs 200 \
    --val_after_epochs 80 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_1_perc_200ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "1 perc semkitti 100 epochs"
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --finetune_epochs 100 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_1_perc_100ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "1 perc semkitti 130 epochs"
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --finetune_epochs 130 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_1_perc_130ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#RUNNING ours on semkitti 10% for 130 epochs
echo "10 perc semkitti 50 epochs"
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --frame_sampling_div 10 \
    --finetune_epochs 50 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_10_perc_50ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "10 perc semkitti 70 epochs"
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --frame_sampling_div 10 \
    --finetune_epochs 70 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_10_perc_70ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "10 perc semkitti 130 epochs"
scripts/submit_ddp_turing.sh --mode f --datasets s  \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    --frame_sampling_div 10 \
    --finetune_epochs 130 \
    --val_after_epochs 80 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_semkitti_10_perc_130ep_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "10 perc nusc 300 epochs and 100 epochs"
scripts/submit_ddp_turing_finetune_nuscenes_10.sh --mode f  \
    --cuda_visible_devices 3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 4 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_nuscenes_10_perc_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

#PAUSED
scripts/submit_ddp_turing_finetune_nuscenes_50.sh --mode f  \
    --cuda_visible_devices 1  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --workers_per_gpu 6 \
    --finetune_bs_per_gpu 16 \
    --pretrained_ckpt checkpoint-ep199.pth.tar \
    --extra_tag try_0 \
    > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_nuscenes_50_perc_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1

# scripts/submit_ddp_turing_finetune_nuscenes_100.sh --mode f  \
#     --cuda_visible_devices 1  \
#     --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
#     --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
#     --workers_per_gpu 6 \
#     --finetune_bs_per_gpu 16 \
#     --pretrained_ckpt checkpoint-ep199.pth.tar \
#     --extra_tag try_0 \
#     > ./output/log/waymo_minkunet_segcontrast_waymo10_lidarplusdet_finetune_nuscenes_100_perc_try_0_$(date +%Y-%m-%d_%H:%M).out 2>&1



#RUNNING
#sweep 1, eps0.4 segcontrast -100 ep with 16 bs
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode pf  \
    --cuda_visible_devices 1 \
    --model_name nuscenes_sweep1_eps0p4_minkunet_segcontrast  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_cfg_file nuscenes_fine1lr_minkunet \
    --extra_tag 100ep_bs16_try0 \
    --workers_per_gpu 8 \
    > ./output/log/nuscenes_sweep1_eps0p4_minkunet_segcontrast_ep100_fine1perc_100ep$(date +%Y-%m-%d_%H:%M).out 2>&1

#RUNNING-DONE
# #scratch nuscenes 1 percent for 100 ep with 16 bs - already done -check later
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode s  \
    --cuda_visible_devices 1 \
    --extra_tag 100ep_bs16_try0 \
    --workers_per_gpu 8 \
    > ./output/log/nuscenes_scratch_1perc_16bs_100ep$(date +%Y-%m-%d_%H:%M).out 2>&1


#DONE
#sweep 1, eps1 det -100 ep with 16 bs
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode f  \
    --cuda_visible_devices 1  \
    --model_name nuscenes_sweep1_eps1_minkunet_segcontrast_det  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_cfg_file nuscenes_fine1lr_minkunet \
    --extra_tag 100ep_bs16_try0 \
    --workers_per_gpu 4 \
    > ./output/log/nuscenes_sweep1_eps1_minkunet_segcontrast_det_ep100_fine1perc_100ep$(date +%Y-%m-%d_%H:%M).out 2>&1


#sweep 1, eps0.4 det -100 ep with 16 bs
#DONE
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode f  \
    --cuda_visible_devices 1 \
    --model_name nuscenes_sweep1_eps0p4_minkunet_segcontrast_det  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_cfg_file nuscenes_fine1lr_minkunet \
    --extra_tag 100ep_bs16_try0 \
    --workers_per_gpu 4 \
    > ./output/log/nuscenes_sweep1_eps0p4_minkunet_segcontrast_det_ep100_fine1perc_100ep$(date +%Y-%m-%d_%H:%M).out 2>&1

#DONE
#sweep 1, eps0.7 det -100 ep with 16 bs
scripts/submit_ddp_turing_pretrain_nuscenes.sh --mode f  \
    --cuda_visible_devices 3  \
    --model_name nuscenes_sweep1_minkunet_segcontrast_det  \
    --pretrain_epochs 100 \
    --pretrain_extra_tag 100ep_try0 \
    --pretrained_ckpt checkpoint-ep99.pth.tar \
    --finetune_cfg_file nuscenes_fine1lr_minkunet\
    --extra_tag 100ep_bs16_try0 \
    --workers_per_gpu 4 \
    > ./output/log/nuscenes_sweep1_minkunet_segcontrast_det_ep100_fine1perc_100ep$(date +%Y-%m-%d_%H:%M).out 2>&1



################# HERE




sbatch --time=1:30:00 --array=1-1%1 --mem=100GB --cpus-per-task=16 --job-name=p_nusc_s3_ld scripts/submit_compute_canada_pretrain_nuscenes.sh --mode p --model_name nuscenes_sweep3_minkunet_segcontrast_lidarplusdet 

sbatch --time=1:30:00 --array=1-1%1 --job-name=f_nusc_s3_ld_also scripts/submit_compute_canada_pretrain_nuscenes.sh --mode f --model_name nuscenes_sweep3_minkunet_segcontrast_lidarplusdet \
    --finetune_cfg_file nuscenes_fine1lr_minkunet_also_cfg \
    --extra_tag also_cfg_try0 

sbatch --time=1:30:00 --array=1-1%1 --job-name=f_nusc_s3_ld_own scripts/submit_compute_canada_pretrain_nuscenes.sh --mode f --model_name nuscenes_sweep3_minkunet_segcontrast_lidarplusdet \
    --finetune_cfg_file nuscenes_fine1lr_minkunet \
    --extra_tag try0 








sbatch --time=1:30:00 --cpus-per-task=16 --mem=80G --account=rrg-swasland --job-name=sweep1_eps1_784_800 scripts/submit_compute_canada_cluster_nuscenes_with_gpu.sh --start_idx 784 --end_idx 800 --sweeps 1 --eps 1
sbatch --time=1:30:00 --cpus-per-task=16 --mem=80G --account=rrg-swasland --job-name=sweep1_eps0p7_770_800 scripts/submit_compute_canada_cluster_nuscenes_with_gpu.sh --start_idx 770 --end_idx 800 --sweeps 1 --eps 0.7
sbatch --time=1:30:00 --cpus-per-task=16 --mem=80G --account=rrg-swasland --job-name=sweep1_eps0p4_765_800 scripts/submit_compute_canada_cluster_nuscenes_with_gpu.sh --start_idx 765 --end_idx 800 --sweeps 1 --eps 0.4
sbatch --time=2:30:00 --cpus-per-task=16 --mem=80G --account=rrg-swasland --job-name=sweep3_eps0p7_525_600 scripts/submit_compute_canada_cluster_nuscenes_with_gpu.sh --start_idx 525 --end_idx 600 --sweeps 3 --eps 0.7
sbatch --time=2:30:00 --cpus-per-task=16 --mem=80G --account=rrg-swasland --job-name=sweep3_eps0p7_125_200 scripts/submit_compute_canada_cluster_nuscenes_with_gpu.sh --start_idx 125 --end_idx 200 --sweeps 3 --eps 0.7
sbatch --time=2:30:00 --cpus-per-task=16 --mem=80G --account=rrg-swasland --job-name=sweep3_eps0p7_670_800 scripts/submit_compute_canada_cluster_nuscenes_with_gpu.sh --start_idx 670 --end_idx 800 --sweeps 3 --eps 0.7
sbatch --time=2:30:00 --cpus-per-task=16 --mem=80G --account=rrg-swasland --job-name=sweep3_eps0p7_285_400 scripts/submit_compute_canada_cluster_nuscenes_with_gpu.sh --start_idx 285 --end_idx 400 --sweeps 3 --eps 0.7




#!/bin/bash

# Define parameters
start_idx=0
end_idx=100
sweeps=3
eps=0.4
eps_name=0p4

# Loop through the ranges
for ((i = 0; i <= 800; i += 200)); do
    # Define the start and end index for the current iteration
    start_index=$i
    end_index=$((i + 200))
    
    # Execute the command and redirect output to the log file
    sbatch --time=5:00:00 --cpus-per-task=16 --mem=80G --account=rrg-swasland --job-name=sweep${sweeps}_eps${eps_name}_${start_index}_${end_index} scripts/submit_compute_canada_cluster_nuscenes_with_gpu.sh --start_idx ${start_index} --end_idx ${end_index} --sweeps ${sweeps} --eps ${eps}


done
