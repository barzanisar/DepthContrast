#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

#from datasets.transforms import transforms

import numpy as np

# collate_fn = transforms.cfl_collate_fn_factory(0)

# def vox_moco_collator(batch):
#     batch_size = len(batch)
    
#     data_point = [x["data"] for x in batch]
#     data_moco = [x["data_moco"] for x in batch]
#     # labels are repeated N+1 times but they are the same
#     labels = [int(x["label"][0]) for x in batch]
#     labels = torch.LongTensor(labels).squeeze()

#     # data valid is repeated N+1 times but they are the same
#     data_valid = torch.BoolTensor([x["data_valid"][0] for x in batch])

#     vox_moco = collate_fn([data_moco[i][0] for i in range(batch_size)])
#     vox = collate_fn([data_point[i][0] for i in range(batch_size)])
    
#     output_batch = {
#         "vox": vox,
#         "vox_moco": vox_moco,
#         "label": labels,
#         "data_valid": data_valid,
#     }
    
#     return output_batch
def vox_moco_collator(batch):
    batch_size = len(batch)
    
    vox = [x["data"] for x in batch]
    vox_moco = [x["data_moco"] for x in batch]

    #TODO: aug matrices and gt boxes for linear probe

    vox_aug_matrix = [x["data_aug_matrix"] for x in batch]
    vox_moco_aug_matrix = [x["data_moco_aug_matrix"] for x in batch]  
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
        "vox": vox_data,
        "vox_moco": vox_moco_data,
        "vox_aug_matrix": vox_aug_matrix,
        "vox_moco_aug_matrix": vox_moco_aug_matrix
    }

    return output_batch
