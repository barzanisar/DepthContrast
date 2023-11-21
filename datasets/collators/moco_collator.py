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

def moco_collator(batch):
    batch_size = len(batch)
    shape_descs_required = 'shape_desc_cluster_ids' in batch[0]
    
    if 'vox' in batch[0]:
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
        

    #append batchidx, x,y,z,i
    coords = []
    coords_moco = []
    for bidx, x in enumerate(batch):
        coor_pad = np.pad(x['points'][:,:4], ((0, 0), (1, 0)), mode='constant', constant_values=bidx) 
        coor_pad_moco = np.pad(x['points_moco'][:,:4], ((0, 0), (1, 0)), mode='constant', constant_values=bidx) 
        coords.append(coor_pad)
        coords_moco.append(coor_pad_moco)
    points = np.concatenate(coords, axis=0) #(N1 + ... Nbs, 5=bxyzi)
    points_moco = np.concatenate(coords_moco, axis=0) #(N1 + ... Nbs, 5=bxyzi)

    cluster_ids = [x["points"][:,-1] for x in batch] # ([N1], [N2], ..., [Nbs])
    gt_boxes_cluster_ids = [x["gt_boxes_cluster_ids"] for x in batch]
    
    cluster_ids_moco = [x["points_moco"][:,-1] for x in batch]
    gt_boxes_moco_cluster_ids = [x["gt_boxes_moco_cluster_ids"] for x in batch]
    
    common_cluster_ids = []
    if shape_descs_required:
        shape_desc_cluster_ids = [x["shape_desc_cluster_ids"] for x in batch]
    for i in range(batch_size):
        # get unique labels from pcd_i and pcd_j
        unique_i = np.unique(cluster_ids[i])
        unique_j = np.unique(cluster_ids_moco[i])

        # get labels present on both pcd (intersection)
        common_ij = np.intersect1d(unique_i, unique_j)[1:]
        if shape_descs_required:
            common_ij_which_has_shape_feats = np.isin(common_ij, shape_desc_cluster_ids[i])
            common_ij = common_ij[common_ij_which_has_shape_feats]

        common_cluster_ids.append(common_ij)

    common_cluster_gtbox_idx=[]
    common_cluster_gtbox_moco_idx=[]
    unscaled_lwhz = [x["unscaled_lwhz_cluster_id"] for x in batch]
    common_unscaled_lwhz = []
    for i in range(batch_size):
        common_box_idx = np.where(np.isin(gt_boxes_cluster_ids[i], common_cluster_ids[i]))[0]
        common_box_idx_moco = np.where(np.isin(gt_boxes_moco_cluster_ids[i], common_cluster_ids[i]))[0]
        assert (gt_boxes_cluster_ids[i][common_box_idx] - gt_boxes_moco_cluster_ids[i][common_box_idx_moco]).sum() == 0
        assert (gt_boxes_cluster_ids[i][common_box_idx] - common_cluster_ids[i]).sum() == 0
        assert (gt_boxes_cluster_ids[i][common_box_idx] - unscaled_lwhz[i][common_box_idx, -1]).sum() == 0
        common_cluster_gtbox_idx.append(common_box_idx)
        common_cluster_gtbox_moco_idx.append(common_box_idx_moco)
        common_unscaled_lwhz.append(unscaled_lwhz[i][common_box_idx, :-1]) #exclude cluster id
        

    common_unscaled_lwhz = np.concatenate(common_unscaled_lwhz, axis=0)

    if shape_descs_required:
        # map_bid_Idxcommonclusterids_Idxshapedescclusterids = []
        map_bid_Idxgtclusterids_Idxshapedescclusterids = []
        map_bid_Idxgtmococlusterids_Idxshapedescclusterids = []

        # common_cluster_ids_has_shape_feats_mask=[]
        # common_shape_feats_mask = []
        # gt_box_idx_has_shape_feats = []
        # gt_box_moco_idx_has_shape_feats = []
        shape_cluster_ids_is_common_mask_batch = []
        for i in range(batch_size):
            shape_cluster_ids_is_common_mask= np.isin(shape_desc_cluster_ids[i], common_cluster_ids[i])
            # idxOf_common_clusterids_has_shape_feats = np.where(np.isin(common_cluster_ids[i], shape_desc_cluster_ids[i]))[0] #mask of common cluster_ids length with true if shape desc id is in common cluster id
            assert (common_cluster_ids[i]- shape_desc_cluster_ids[i][shape_cluster_ids_is_common_mask]).sum() == 0
            # common_cluster_ids_has_shape_feats_mask.append(common_has_shape_feats_mask) 
            # common_shape_feats_mask.append(shape_cluster_ids_is_common_mask)

            gt_box_idx = np.where(np.isin(gt_boxes_cluster_ids[i], shape_desc_cluster_ids[i]))[0]
            shape_des_idx = np.where(np.isin(shape_desc_cluster_ids[i], gt_boxes_cluster_ids[i]))[0]
            assert (gt_boxes_cluster_ids[i][gt_box_idx] - shape_desc_cluster_ids[i][shape_des_idx]).sum() == 0

            
            gt_box_moco_idx = np.where(np.isin(gt_boxes_moco_cluster_ids[i], shape_desc_cluster_ids[i]))[0]
            shape_des_moco_idx = np.where(np.isin(shape_desc_cluster_ids[i], gt_boxes_moco_cluster_ids[i]))[0]
            assert (gt_boxes_moco_cluster_ids[i][gt_box_moco_idx] - shape_desc_cluster_ids[i][shape_des_moco_idx]).sum() == 0

            # bid_Idxcommonclusterids_Idxshapedescclusterids = np.hstack([
            #     i*np.ones((idxOf_common_clusterids_has_shape_feats.shape[0], 1)),
            #     idxOf_common_clusterids_has_shape_feats.reshape(-1,1), 
            #     idxOf_shape_cluster_ids_is_common.reshape(-1,1)])

            bid_Idxgtclusterids_Idxshapedescclusterids = np.hstack([
                i*np.ones((gt_box_idx.shape[0], 1)),
                gt_box_idx.reshape(-1,1), 
                shape_des_idx.reshape(-1,1)])
            
            bid_Idxgtmococlusterids_Idxshapedescclusterids = np.hstack([
                i*np.ones((gt_box_moco_idx.shape[0], 1)),
                gt_box_moco_idx.reshape(-1,1), 
                shape_des_moco_idx.reshape(-1,1)])
            
            shape_cluster_ids_is_common_mask_batch.append(shape_cluster_ids_is_common_mask)
            # map_bid_Idxcommonclusterids_Idxshapedescclusterids.append(bid_Idxcommonclusterids_Idxshapedescclusterids)
            map_bid_Idxgtclusterids_Idxshapedescclusterids.append(bid_Idxgtclusterids_Idxshapedescclusterids)
            map_bid_Idxgtmococlusterids_Idxshapedescclusterids.append(bid_Idxgtmococlusterids_Idxshapedescclusterids)


            b=1

        shape_cluster_ids_is_common_mask = np.concatenate(shape_cluster_ids_is_common_mask, axis=0)
        # map_bid_Idxcommonclusterids_Idxshapedescclusterids = np.concatenate(map_bid_Idxcommonclusterids_Idxshapedescclusterids, axis=0)
        map_bid_Idxgtclusterids_Idxshapedescclusterids = np.concatenate(map_bid_Idxgtclusterids_Idxshapedescclusterids, axis=0)
        map_bid_Idxgtmococlusterids_Idxshapedescclusterids = np.concatenate(map_bid_Idxgtmococlusterids_Idxshapedescclusterids, axis=0)

        shape_desc_cluster_ids = []
        for bidx, x in enumerate(batch):
            shape_descs_pad = np.pad(x['shape_desc_cluster_ids'], ((0, 0), (1, 0)), mode='constant', constant_values=bidx) 
            shape_desc_cluster_ids.append(shape_descs_pad)
        bid_shape_desc_cluster_ids = np.concatenate(shape_desc_cluster_ids, axis=0)
        shape_descs = np.concatenate([x["shape_descs"] for x in batch], axis=0)


    # make gt boxes in shape (batch size, max gt box len, 8) (xyz, lwh, rz, class label)
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
                     'gt_boxes': batch_gt_boxes3d,
                     'gt_boxes_cluster_ids': gt_boxes_cluster_ids, 
                    'common_cluster_ids': common_cluster_ids, 
                    'common_cluster_gtbox_idx': common_cluster_gtbox_idx,
                    'common_unscaled_lwhz': common_unscaled_lwhz,
                     'batch_size': batch_size},
                    
                    'input_moco': 
                    {'points': points_moco,
                     'cluster_ids': cluster_ids_moco,
                     'gt_boxes': batch_gt_boxes3d_moco,
                     'gt_boxes_cluster_ids': gt_boxes_moco_cluster_ids, 
                    'common_cluster_ids': common_cluster_ids,
                    'common_cluster_gtbox_idx': common_cluster_gtbox_moco_idx,
                     'batch_size': batch_size}
                    
                    }
    
    if 'vox' in batch[0]:
        output_batch['input'].update({
                     'voxels': vox_dict['vox']['voxels'],
                     'voxel_num_points': vox_dict['vox']['voxel_num_points'],
                     'voxel_coords': vox_dict['vox']['voxel_coords']})
        
        output_batch['input_moco'].update({
                     'voxels': vox_dict['vox_moco']['voxels'],
                     'voxel_num_points': vox_dict['vox_moco']['voxel_num_points'],
                     'voxel_coords': vox_dict['vox_moco']['voxel_coords']})
        
    if shape_descs_required:
        output_batch['input'].update({
                     'shape_descs': shape_descs,
                     'bid_shape_desc_cluster_ids': bid_shape_desc_cluster_ids,
                     'shape_cluster_ids_is_common_mask_batch': shape_cluster_ids_is_common_mask_batch,
                     'map_bid_Idxgtclusterids_Idxshapedescclusterids': map_bid_Idxgtclusterids_Idxshapedescclusterids,
                     'map_bid_Idxgtmococlusterids_Idxshapedescclusterids': map_bid_Idxgtmococlusterids_Idxshapedescclusterids})
        


    return output_batch
