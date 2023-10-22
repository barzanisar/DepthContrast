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

def vox_moco_collator(batch):
    batch_size = len(batch)
    
    vox_dict = {'vox':{}, 'vox_moco':{}}
    for key in ['vox', 'vox_moco']:
        for k in ['voxels', 'voxel_num_points']:
            val = [x[key][k] for x in batch]
            vox_dict[key][k] = np.concatenate(val, axis=0)
        
        vox_dict[key]['voxels'] = vox_dict[key]['voxels'][:,:,:-1] #remove cluster lbl
        #apend batch id in voxel_coords
        val = [x[key]['voxel_coords'] for x in batch]
        coors=[]
        for i, coor in enumerate(val): # append batch_id in front so points: (Nxbs, 6=bxyzie) voxel_coords:(num voxelsx bs, 4=bzyx vox coord)
            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
            coors.append(coor_pad)
        vox_dict[key]['voxel_coords'] = np.concatenate(coors, axis=0)
        

    # TODO: should these pts and cluster ids be the ones taken from voxels?
    points = [x["points"][:,:4] for x in batch] #select xyzi # 8 elems (num pts each pc, 4)
    cluster_ids = [x["vox"]["voxels"][:,:,-1] for x in batch] # 8 elems (num pts each pc,)
    gt_boxes_cluster_ids = [x["gt_boxes_cluster_ids"] for x in batch]
    

    points_moco =  [x["points_moco"][:,:4] for x in batch]
    cluster_ids_moco = [x["vox_moco"]["voxels"][:,:,-1] for x in batch]
    gt_boxes_moco_cluster_ids = [x["gt_boxes_moco_cluster_ids"] for x in batch]
    
    common_cluster_ids = []
    for i in range(batch_size):
        # get unique labels from pcd_i and pcd_j
        unique_i = np.unique(cluster_ids[i])
        unique_j = np.unique(cluster_ids_moco[i])

        # get labels present on both pcd (intersection)
        common_ij = np.intersect1d(unique_i, unique_j)[1:]
        common_cluster_ids.append(common_ij)

    common_cluster_gtbox_idx=[]
    common_cluster_gtbox_moco_idx=[]
    for i in range(batch_size):
        common_box_idx = np.where(np.isin(gt_boxes_cluster_ids[i], common_cluster_ids[i]))[0]
        common_box_idx_moco = np.where(np.isin(gt_boxes_moco_cluster_ids[i], common_cluster_ids[i]))[0]
        assert (gt_boxes_cluster_ids[i][common_box_idx] - gt_boxes_moco_cluster_ids[i][common_box_idx_moco]).sum() == 0
        assert (gt_boxes_cluster_ids[i][common_box_idx] - common_cluster_ids[i]).sum() == 0
        common_cluster_gtbox_idx.append(common_box_idx)
        common_cluster_gtbox_moco_idx.append(common_box_idx_moco)
        
    # make gt boxes in shape (batch size, max gt box len, 8) (xyz, lwh, rz, cluster label)
    max_gt = max([len(x['gt_boxes']) for x in batch])
    batch_gt_boxes3d = np.zeros((batch_size, max_gt, batch[0]['gt_boxes'].shape[-1]), dtype=np.float32) # (batch size = 2, max_gt_boxes in a pc in this batch = 67, 8)
    for k in range(batch_size):
        batch_gt_boxes3d[k, :batch[k]['gt_boxes'].__len__(), :] = batch[k]['gt_boxes']

    # make gt boxes in shape (batch size, max gt box len, 8)
    max_gt = max([len(x['gt_boxes_moco']) for x in batch])
    batch_gt_boxes3d_moco = np.zeros((batch_size, max_gt, batch[0]['gt_boxes_moco'].shape[-1]), dtype=np.float32) # (batch size = 2, max_gt_boxes in a pc in this batch = 67, 8)
    for k in range(batch_size):
        batch_gt_boxes3d_moco[k, :batch[k]['gt_boxes_moco'].__len__(), :] = batch[k]['gt_boxes_moco']


    output_batch = {'input': 
                    {'points': points,
                     'cluster_ids': cluster_ids,
                    #  'box_ids_of_pts': box_ids_of_pts,
                     'gt_boxes': batch_gt_boxes3d,
                     'gt_boxes_cluster_ids': gt_boxes_cluster_ids, 
                    #  'gt_boxes_idx': batch_gt_boxes_idx,
                    'common_cluster_ids': common_cluster_ids, 
                    'common_cluster_gtbox_idx': common_cluster_gtbox_idx,
                     'batch_size': batch_size,
                     'voxels': vox_dict['vox']['voxels'],
                     'voxel_num_points': vox_dict['vox']['voxel_num_points'],
                     'voxel_coords': vox_dict['vox']['voxel_coords']},
                    'input_moco': 
                    {'points': points_moco,
                     'cluster_ids': cluster_ids_moco,
                    #  'box_ids_of_pts': box_ids_of_pts_moco, 
                     'gt_boxes': batch_gt_boxes3d_moco,
                     'gt_boxes_cluster_ids': gt_boxes_moco_cluster_ids, 
                    #  'gt_boxes_idx': batch_gt_boxes_moco_idx, 
                    'common_cluster_ids': common_cluster_ids,
                    'common_cluster_gtbox_idx': common_cluster_gtbox_moco_idx,
                     'batch_size': batch_size,
                     'voxels': vox_dict['vox_moco']['voxels'],
                     'voxel_num_points': vox_dict['vox_moco']['voxel_num_points'],
                     'voxel_coords': vox_dict['vox_moco']['voxel_coords']},
                    
    }

    return output_batch
