#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np

def point_vox_moco_collator(batch):
    batch_size = len(batch)
    
    point = [x["data"] for x in batch]
    point_moco = [x["data_moco"] for x in batch]
    vox = [x["vox"] for x in batch]
    vox_moco = [x["vox_moco"] for x in batch]

    points_moco = torch.stack([point_moco[i][0] for i in range(batch_size)]) #(8, 16384, 4)
    points = torch.stack([point[i][0] for i in range(batch_size)]) #(8, 16384, 4)

    #TODO: aug matrices and gt boxes for linear probe
    data_aug_matrix = [x["data_aug_matrix"] for x in batch]
    data_moco_aug_matrix = [x["data_moco_aug_matrix"] for x in batch]  
    points_moco_aug_matrix = torch.stack([data_moco_aug_matrix[i] for i in range(batch_size)]) #(8, 3, 3)
    points_aug_matrix = torch.stack([data_aug_matrix[i] for i in range(batch_size)])#(8, 3, 3)
    vox_aug_matrix = [x["vox_aug_matrix"] for x in batch]
    vox_moco_aug_matrix = [x["vox_moco_aug_matrix"] for x in batch]  
    vox_moco_aug_matrix = torch.stack([vox_moco_aug_matrix[i] for i in range(batch_size)]) #(8, 3, 3)
    vox_aug_matrix = torch.stack([vox_aug_matrix[i] for i in range(batch_size)])#(8, 3, 3)


    vox_data = {"voxels":[], "voxel_coords":[], "voxel_num_points":[]}
    counter = 0
    for data in vox:
        temp = data[0]
        voxels_shape = temp["voxels"].shape
        voxel_num_points_shape = temp["voxel_num_points"].shape
        voxel_coords_shape = temp["voxel_coords"].shape
        for key,val in temp.items():
            if key in ['voxels', 'voxel_num_points']:
                if len(vox_data[key]) > 0:
                    vox_data[key] = np.concatenate([vox_data[key], val], axis=0)
                else:
                    vox_data[key] = val
            elif key == 'voxel_coords':
                coor = np.pad(val, ((0, 0), (1, 0)), mode='constant', constant_values=counter) #Pad batch index to (z,y,x) coord -> (batch idx, z,y,x)
                if len(vox_data[key]) > 0:
                    vox_data[key] = np.concatenate([vox_data[key], coor], axis=0)
                else:
                    vox_data[key] = coor
        counter += 1
        
    vox_moco_data = {"voxels":[], "voxel_coords":[], "voxel_num_points":[]}
    counter = 0
    for data in vox_moco:
        temp = data[0]
        voxels_shape = temp["voxels"].shape
        voxel_num_points_shape = temp["voxel_num_points"].shape
        voxel_coords_shape = temp["voxel_coords"].shape
        for key,val in temp.items():                
            if key in ['voxels', 'voxel_num_points']:
                if len(vox_moco_data[key]) > 0:
                    vox_moco_data[key] = np.concatenate([vox_moco_data[key], val], axis=0)
                else:
                    vox_moco_data[key] = val
            elif key in 'voxel_coords':
                coor = np.pad(val, ((0, 0), (1, 0)), mode='constant', constant_values=counter)

                if len(vox_moco_data[key]) > 0:
                    vox_moco_data[key] = np.concatenate([vox_moco_data[key], coor], axis=0)
                else:
                    vox_moco_data[key] = coor
        counter += 1
        
    vox_data = {k:torch.from_numpy(vox_data[k]) for k in vox_data}
    vox_moco_data = {k:torch.from_numpy(vox_moco_data[k]) for k in vox_moco_data}

    output_batch = {
        "points": points,
        "points_moco": points_moco,
        "vox": vox_data,
        "vox_moco": vox_moco_data,
        "points_aug_matrix": points_aug_matrix,
        "points_moco_aug_matrix": points_moco_aug_matrix,
        "vox_aug_matrix": vox_aug_matrix,
        "vox_moco_aug_matrix": vox_moco_aug_matrix
    }

    return output_batch
