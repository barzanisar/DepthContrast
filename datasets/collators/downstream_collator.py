#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def downstream_collator(batch):
    batch_size = len(batch)
    
    if 'vox' in batch[0]:
        vox_dict = {'vox':{}}
        
        for k in ['voxels', 'voxel_num_points']:
            val = [x['vox'][k] for x in batch]
            vox_dict['vox'][k] = np.concatenate(val, axis=0)
        
        vox_dict['vox']['voxels'] = vox_dict['vox']['voxels'][:,:,:-1] #remove seg lbl
        #append batch id in voxel_coords
        val = [x['vox']['voxel_coords'] for x in batch]
        coors=[]
        for i, coor in enumerate(val): # append batch_id in front so points: (Nxbs, 6=bxyzie) voxel_coords:(num voxelsx bs, 4=bzyx vox coord)
            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
            coors.append(coor_pad)
        vox_dict['vox']['voxel_coords'] = np.concatenate(coors, axis=0)

    
    #append batchidx, x,y,z,i
    coords = []
    for bidx, x in enumerate(batch):
        coor_pad = np.pad(x['points'][:,:4], ((0, 0), (1, 0)), mode='constant', constant_values=bidx) 
        coords.append(coor_pad)
    points = np.concatenate(coords, axis=0) #(N1 + ... Nbs, 5=bxyzi)
    seg_labels = np.concatenate([x["points"][:,-1] for x in batch], axis=0) # (N1, N2, ..., Nbs)
    
    output_batch = {'input': 
                    {'points': points,
                     'seg_labels': seg_labels,
                     'batch_size': batch_size}
                    
                    }
    
    # if downstream task is detection:
    if 'gt_boxes' in batch[0]:
        pt_wise_gtbox_idxs = np.concatenate([x["pt_wise_gtbox_idxs"] for x in batch], axis=0) # (N1, N2, ..., Nbs)
        
        
        # make gt boxes in shape (batch size, max gt box len, 8) (xyz, lwh, rz, class label)
        max_gt = max([len(x['gt_boxes']) for x in batch])
        batch_gt_boxes3d = np.zeros((batch_size, max_gt, batch[0]['gt_boxes'].shape[-1]), dtype=np.float32) # (batch size = 2, max_gt_boxes in a pc in this batch = 67, 8)
        for k in range(batch_size):
            batch_gt_boxes3d[k, :batch[k]['gt_boxes'].__len__(), :] = batch[k]['gt_boxes']


        output_batch['input'].update(
                        {'gt_boxes': batch_gt_boxes3d,
                        'pt_wise_gtbox_idxs': pt_wise_gtbox_idxs})
    
    if 'vox' in batch[0]:
        output_batch['input'].update({
                     'voxels': vox_dict['vox']['voxels'],
                     'voxel_num_points': vox_dict['vox']['voxel_num_points'],
                     'voxel_coords': vox_dict['vox']['voxel_coords']})
        
    return output_batch
