############################### table 1 #############################
TRY=0
echo "scratch long try"$TRY""
scripts/submit_ddp_turing_1.sh --mode s --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 0  \
    --long_finetuning \
    > ./output/log/scratch_finetune_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "DC long try"$TRY""
scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 1  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml \
    --model_name depthcontrast_10perc_waymo_minkunet  \
    --long_finetuning \
    > ./output/log/depthcontrast_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "DC lidar+det long try"$TRY""
scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 1  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name depthcontrast_lidaraug_det_10perc_waymo_minkunet  \
    --long_finetuning \
    > ./output/log/depthcontrast_lidarplusdet_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "SC long try"$TRY""
scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 2  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml \
    --model_name segcontrast_10perc_waymo_minkunet  \
    --long_finetuning \
    > ./output/log/segcontrast_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "SC lidar+det long try"$TRY""
scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 2  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --long_finetuning \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "SC reghead long try"$TRY""
scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 3  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml \
    --model_name segcontrast_reghead_10perc_waymo_minkunet  \
    --long_finetuning \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

echo "SC + attn long try"$TRY""
scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 3  \
    --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml \
    --model_name proposalcontrast_10perc_waymo_minkunet  \
    --long_finetuning \
    > ./output/log/proposalcontrast_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

##########################################################################################################################

echo "scratch short try"$TRY""
scripts/submit_ddp_turing_1.sh --mode s --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 0  \
    > ./output/log/scratch_finetune_short_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

#Hyper param sensitivity on cluster_eps (p32 is fixed to 0.4): fintune on 1% wns for 15 epochs total_bs 16 
eps=0p1 #0p2, 0p3, 0p4
echo "SC lidar+det eps$eps try"$TRY""
scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4_eps$eps.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps$eps  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps"$eps"_fine1_15epochs_try"$TRY"_final_$(date +%Y-%m-%d_%H:%M).out 2>&1

eps=0p2 #0p3
echo "SC eps$eps try"$TRY""
scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_eps$eps.yaml \
    --model_name segcontrast_10perc_waymo_minkunet_eps$eps  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    > ./output/log/segcontrast_10perc_waymo_minkunet_eps"$eps"_fine1_15epochs_try"$TRY"_final_$(date +%Y-%m-%d_%H:%M).out 2>&1

##########################################################################################################################

#Hyper param sensitivity on lidar aug prob (with eps fixed to 0.2):
# fintune on 1% wns for 15 epochs total_bs 16 
prob=0p3 #0p5 0p6
echo "SC lidar+det lidar p32_"$prob""
scripts/submit_ddp_turing_1.sh  --mode f --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 0  \
    --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_"$prob"_eps0p2.yaml \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_"$prob"_eps0p2  \
    --pretrain_epochs 30 \
    --pretrained_ckpt checkpoint-ep29.pth.tar \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_"$prob"_eps0p2_fine1_15epochs_try"$TRY"_final_$(date +%Y-%m-%d_%H:%M).out 2>&1

##########################################################################################################################





