############################### table 1 #############################
try_values=(0 1 2)  #-> turing, lovelace, lovelace
for TRY in "${try_values[@]}"; do
    echo "scratch long try"$TRY"" #-> turing, turing
    scripts/submit_ddp_turing_w5perc.sh --mode s --datasets w --extra_tag try"$TRY"_final \
        --cuda_visible_devices 2  \
        --long_finetuning \
        > ./output/log/scratch_finetune_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1
done


try_values=(1 2)  #-> turing, lovelace, lovelace
for TRY in "${try_values[@]}"; do
    echo "DC long try"$TRY""
    scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
        --cuda_visible_devices 0  \
        --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml \
        --model_name depthcontrast_10perc_waymo_minkunet  \
        --long_finetuning \
        > ./output/log/depthcontrast_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1
done

#Run on turing
TRY=0
echo "DC long try"$TRY""
scripts/submit_ddp_turing_w5perc.sh --mode f --datasets w --extra_tag try"$TRY"_final \
    --cuda_visible_devices 2  \
    --cfg_file configs/waymo_minkunet_depthcontrast_waymo10.yaml \
    --model_name depthcontrast_10perc_waymo_minkunet  \
    --long_finetuning \
    > ./output/log/depthcontrast_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

try_values=(0 1 2)  #-> lovelace, lovelace
for TRY in "${try_values[@]}"; do
    echo "DC lidar+det long try"$TRY"" #-> lovelace, lovelace, lovelace
    scripts/submit_ddp_turing_w5perc.sh --mode f --datasets w --extra_tag try"$TRY"_final \
        --cuda_visible_devices 2  \
        --cfg_file configs/waymo_minkunet_depthcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
        --model_name depthcontrast_lidaraug_det_10perc_waymo_minkunet  \
        --long_finetuning \
        > ./output/log/depthcontrast_lidarplusdet_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1
done

try_values=(0 1 2)  #-> lovelace, lovelace
for TRY in "${try_values[@]}"; do
    echo "SC long try"$TRY"" #-> lovelace
    scripts/submit_ddp_turing_w5perc.sh --mode f --datasets w --extra_tag try"$TRY"_final \
        --cuda_visible_devices 2  \
        --cfg_file configs/waymo_minkunet_segcontrast_waymo10.yaml \
        --model_name segcontrast_10perc_waymo_minkunet  \
        --long_finetuning \
        > ./output/log/segcontrast_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1
done

##############################
try_values=(0 1 2) 
for TRY in "${try_values[@]}"; do
    echo "SC lidar+det long try"$TRY"" #->turing
    scripts/submit_ddp_turing_w5perc.sh --mode f --datasets w --extra_tag try"$TRY"_final \
        --cuda_visible_devices 0  \
        --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidar_aug_single_randh_dethead_0p5w.yaml \
        --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
        --long_finetuning \
        > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1
done

try_values=(0 1 2) 
for TRY in "${try_values[@]}"; do
    echo "SC reghead long try"$TRY"" #->turing
    scripts/submit_ddp_turing_w5perc.sh --mode f --datasets w --extra_tag try"$TRY"_final \
        --cuda_visible_devices 1  \
        --cfg_file configs/waymo_minkunet_segcontrast_waymo10_reghead.yaml \
        --model_name segcontrast_reghead_10perc_waymo_minkunet  \
        --long_finetuning \
        > ./output/log/segcontrast_reghead_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1
done

try_values=(0 1 2) 
for TRY in "${try_values[@]}"; do
    echo "SC + attn long try"$TRY"" #->turing
    scripts/submit_ddp_turing_w5perc.sh --mode f --datasets w --extra_tag try"$TRY"_final \
        --cuda_visible_devices 2  \
        --cfg_file configs/waymo_minkunet_proposalcontrast_waymo10.yaml \
        --model_name proposalcontrast_10perc_waymo_minkunet  \
        --long_finetuning \
        > ./output/log/proposalcontrast_10perc_waymo_minkunet_long_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

done

##########################################################################################################################

echo "scratch short try"$TRY"" #->lovelace
scripts/submit_ddp_turing_1.sh --mode s --datasets wns --extra_tag try"$TRY"_final \
    --cuda_visible_devices 3  \
    > ./output/log/scratch_finetune_short_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

#Hyper param sensitivity on cluster_eps (p32 is fixed to 0.4): fintune on 1% wns for 15 epochs total_bs 16 
eps_values=("0p1" "0p2" "0p3" "0p4")
try_values=(0 1 2)
for eps in "${eps_values[@]}"; do
    for TRY in "${try_values[@]}"; do
        echo "SC lidar+det eps$eps try"$TRY""
        scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
            --cuda_visible_devices 0  \
            --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_0p4_eps$eps.yaml \
            --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps$eps  \
            --pretrain_epochs 30 \
            --pretrained_ckpt checkpoint-ep29.pth.tar \
            > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_0p4_eps"$eps"_fine1_15epochs_try"$TRY"_final_$(date +%Y-%m-%d_%H:%M).out 2>&1
    done
done

eps_values=("0p2" "0p3")
try_values=(0 1 2)
for eps in "${eps_values[@]}"; do
    for TRY in "${try_values[@]}"; do
        echo "SC eps$eps try"$TRY""
        scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
            --cuda_visible_devices 1  \
            --cfg_file configs/waymo_minkunet_segcontrast_waymo10_eps$eps.yaml \
            --model_name segcontrast_10perc_waymo_minkunet_eps$eps  \
            --pretrain_epochs 30 \
            --pretrained_ckpt checkpoint-ep29.pth.tar \
            > ./output/log/segcontrast_10perc_waymo_minkunet_eps"$eps"_fine1_15epochs_try"$TRY"_final_$(date +%Y-%m-%d_%H:%M).out 2>&1
    done
done


##########################################################################################################################

#Hyper param sensitivity on lidar aug prob (with eps fixed to 0.2):
# fintune on 1% wns for 15 epochs total_bs 16 
prob_values=("0p3" "0p5" "0p6")
try_values=(0 1 2)
for prob in "${prob_values[@]}"; do
    for TRY in "${try_values[@]}"; do
        echo "SC lidar+det lidar p32_"$prob""
        scripts/submit_ddp_turing_1.sh  --mode f --datasets wns --extra_tag try"$TRY"_final \
            --cuda_visible_devices 2  \
            --cfg_file configs/waymo_minkunet_segcontrast_waymo10_lidarplusdet_p32_"$prob"_eps0p2.yaml \
            --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_"$prob"_eps0p2  \
            --pretrain_epochs 30 \
            --pretrained_ckpt checkpoint-ep29.pth.tar \
            > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_p32_"$prob"_eps0p2_fine1_15epochs_try"$TRY"_final_$(date +%Y-%m-%d_%H:%M).out 2>&1
    done
done
##########################################################################################################################

#Ablations

try_values=(0 1 2)
for TRY in "${try_values[@]}"; do
    echo "SC -> 1% finetune for 15 epochs "$TRY""
    scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
        --cuda_visible_devices 3  \
        --model_name segcontrast_10perc_waymo_minkunet  \
        > ./output/log/segcontrast_10perc_waymo_minkunet_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

    echo "SC + polarmix -> 1% finetune for 15 epochs "$TRY""
    scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
        --cuda_visible_devices 3  \
        --model_name segcontrast_lidaraug_mixed_10perc_waymo_minkunet  \
        > ./output/log/segcontrast_lidaraug_mixed_10perc_waymo_minkunet_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

    echo "SC + single pattern + randh -> 1% finetune for 15 epochs "$TRY""
    scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
        --cuda_visible_devices 3  \
        --model_name segcontrast_lidaraug_single_randh_10perc_waymo_minkunet  \
        > ./output/log/segcontrast_lidaraug_single_randh_10perc_waymo_minkunet_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

    echo "SC + det -> 1% finetune for 15 epochs "$TRY""
    scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
        --cuda_visible_devices 3  \
        --model_name segcontrast_det_10perc_waymo_minkunet  \
        > ./output/log/segcontrast_det_10perc_waymo_minkunet_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

    echo "SC + lidarplusdet -> 1% finetune for 15 epochs "$TRY""
    scripts/submit_ddp_turing_1.sh --mode f --datasets wns --extra_tag try"$TRY"_final \
        --cuda_visible_devices 3  \
        --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
        > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1
    
done

##########################################################################################################################

#finetune our model pretrained on waymo on semantickitti with bs8 using ALSO optimizer and bevcontrast finetuning protocol ->lovelace
scripts/submit_ddp_turing_finetune_semkitti_also.sh --mode f  \
    --cuda_visible_devices 2 \
    --model_name segcontrast_lidarplusdet_10perc_waymo_minkunet  \
    --extra_tag bs8_try"$TRY"_also \
    --workers_per_gpu 4 \
    > ./output/log/segcontrast_lidarplusdet_10perc_waymo_minkunet_fineSemkitti-0p1-1-10-50-100perc_bs8_also_optim_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

##########################################################################################################################

#finetune SC+det on nusc with bs8 using ALSO optimizer and bevcontrast finetuning protocol
scripts/submit_ddp_turing_pretrain_nuscenes_also.sh --mode f  \
    --cuda_visible_devices 3 \
    --model_name nuscenes_sweep1_eps0p3_minkunet_segcontrast_det  \
    --extra_tag bs8_try"$TRY"_also \
    --workers_per_gpu 4 \
    > ./output/log/nuscenes_sweep1_eps0p3_minkunet_segcontrast_det_ep200_fine-0p1-1-10-50-100perc_bs8_also_optim_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1

#finetune SC on nusc with bs8 using ALSO optimizer and bevcontrast finetuning protocol
scripts/submit_ddp_turing_pretrain_nuscenes_also.sh --mode f  \
    --cuda_visible_devices 3 \
    --model_name nuscenes_sweep1_eps0p3_minkunet_segcontrast  \
    --extra_tag bs8_try"$TRY"_also \
    --workers_per_gpu 4 \
    > ./output/log/nuscenes_sweep1_eps0p3_minkunet_segcontrast_ep200_fine-0p1-1-10-50-100perc_bs8_also_optim_try"$TRY"_$(date +%Y-%m-%d_%H:%M).out 2>&1
