#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

def point_moco_collator(batch):
    batch_size = len(batch)

    pcs_batch = [x["points"][:,:4] for x in batch] #select xyzi
    
    #append batchidx, x,y,z,i
    for batch_id, pc in enumerate(pcs_batch):
        pcs_batch[batch_id]= np.pad(pc, ((0, 0), (1, 0)), mode='constant', constant_values=batch_id) 
    points = np.concatenate(pcs_batch, axis=0) #(16384 x 8, 4)

    pcs_moco_batch = [x["points_moco"][:,:4] for x in batch]
    #append batchidx, x,y,z,i
    for batch_id, pc in enumerate(pcs_moco_batch):
        pcs_moco_batch[batch_id]= np.pad(pc, ((0, 0), (1, 0)), mode='constant', constant_values=batch_id) 
    points_moco = np.concatenate(pcs_moco_batch, axis=0) #(16384 x 8, 4)
    
    # TODO: old remove
    #pcs_moco_batch = [x["data_moco"][0] for x in batch]
    #points_moco = np.stack([pc[:,:4] for pc in pcs_moco_batch]) #(8, 16384, 4)

    box_ids_of_pts = np.stack([x["box_ids_of_pts"] for x in batch]) #(2, 20000)
    box_ids_of_pts_moco = np.stack([x["box_ids_of_pts_moco"] for x in batch])   

    # make gt boxes in shape (batch size, max gt box len, 8)
    max_gt = max([len(x['gt_boxes']) for x in batch])
    batch_gt_boxes3d = np.zeros((batch_size, max_gt, batch[0]['gt_boxes'].shape[-1]), dtype=np.float32) # (batch size = 2, max_gt_boxes in a pc in this batch = 67, 8)
    for k in range(batch_size):
        batch_gt_boxes3d[k, :batch[k]['gt_boxes'].__len__(), :] = batch[k]['gt_boxes']

    # make gt boxes in shape (batch size, max gt box len, 8)
    max_gt = max([len(x['gt_boxes_moco']) for x in batch])
    batch_gt_boxes3d_moco = np.zeros((batch_size, max_gt, batch[0]['gt_boxes_moco'].shape[-1]), dtype=np.float32) # (batch size = 2, max_gt_boxes in a pc in this batch = 67, 8)
    for k in range(batch_size):
        batch_gt_boxes3d_moco[k, :batch[k]['gt_boxes_moco'].__len__(), :] = batch[k]['gt_boxes_moco']

    batch_gt_boxes_idx = np.array([x["gt_boxes_idx"] for x in batch])
    batch_gt_boxes_moco_idx = np.array([x["gt_boxes_moco_idx"] for x in batch])

    output_batch = {'input': 
                    {'points': points,
                     'box_ids_of_pts': box_ids_of_pts,
                     'gt_boxes': batch_gt_boxes3d, 
                     'gt_boxes_idx': batch_gt_boxes_idx, 
                     'batch_size': batch_size},
                    'input_moco': 
                    {'points': points_moco,
                     'box_ids_of_pts': box_ids_of_pts_moco, 
                     'gt_boxes': batch_gt_boxes3d_moco, 
                     'gt_boxes_idx': batch_gt_boxes_moco_idx, 
                     'batch_size': batch_size}
    }

    return output_batch
