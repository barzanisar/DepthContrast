#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from SparseTensor import SparseTensor
    import MinkowskiEngine as ME
except:
    pass

import numpy as np

from utils import main_utils
from third_party.OpenPCDet.pcdet.models.pretext_heads.gather_utils import *

def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(np.prod(p.size())))
    return desc


class BaseSSLMultiInputOutputModel(nn.Module):
    def __init__(self, model_config, cluster, dataset, logger, linear_probe=False):
        """
        Class to implement a self-supervised model.
        The model is split into `trunk' that computes features.
        """
        self.config = model_config
        self.linear_probe = linear_probe
        self.logger = logger
        self.cluster = cluster
        super().__init__()
        self.trunk = self._get_trunk(dataset)
        self.det_head = self._get_head(dataset, key="MODEL_DET_HEAD")
        self.aux_head = self._get_head(dataset, key="MODEL_AUX_HEAD")
        self.m = 0.999 ### Can be tuned momentum parameter for momentum update
        
    def multi_input_with_head_mapping_forward(self, batch_dict):
        output_dict = {}

        outputs, _, _ = self._single_input_forward(batch_dict['input'], 0)
        outputs['batch_dict'].pop('points')
        output_dict['output'] =  outputs['batch_dict']

        outputs_moco, _, _ = self._single_input_forward_MOCO(batch_dict['input_moco'], 1)
        outputs_moco['batch_dict'].pop('points')
        output_dict['output_moco'] =  outputs_moco['batch_dict']
        
        concat_outputs_condition_for_det = 'MODEL_DET_HEAD' in self.config and self.config['MODEL_DET_HEAD'].get('INPUT_MOCO_FEATS', False)
        concat_outputs_condition = concat_outputs_condition_for_det or 'MODEL_AUX_HEAD' in self.config
        if concat_outputs_condition:
            new_batch_dict = self._concat_encoder_outputs(output_dict['output'], output_dict['output_moco'], for_det=concat_outputs_condition_for_det)

        
        if 'MODEL_DET_HEAD' in self.config:
            if self.config['MODEL_DET_HEAD'].get('INPUT_MOCO_FEATS', False):
                outputs_head, _, _ = self.det_head(new_batch_dict) 
            else:
                outputs_head, _, _ = self.det_head(output_dict['output'])
            
            output_dict['loss_det_head']= outputs_head['loss']



        if 'MODEL_AUX_HEAD' in self.config:
            outputs_head, _, _ = self.aux_head(new_batch_dict)
            output_dict['loss_aux_head']= outputs_head['batch_dict']['seg_reg_loss']
            

            #all_outputs.append(loss_dict)
        return output_dict
    
    def _concat_encoder_outputs(self, output_base_dict, output_moco_dict, for_det):
        all_outputs = []
        new_batch_dict = {}

        all_outputs.append(output_base_dict)
        all_outputs.append(output_moco_dict)


        if for_det:
            # All inputs needed only for the detection head
            new_batch_dict['batch_size'] = all_outputs[1]['batch_size'] + all_outputs[0]['batch_size']
            #new_batch_dict['points'] = torch.cat([output['points'] for output in all_outputs], dim=0)

            if self.config['VOX']:
                #TODO check if we need voxel_coords for other heads
                new_batch_dict['spatial_features_2d'] = torch.cat([output['spatial_features_2d'] for output in all_outputs], dim=0)
            else:
                all_outputs[1]['point_coords'][:,0] += all_outputs[0]['batch_size']
                #points (N, 4=xyzi), point_features (N, 128), point_coords (N, 4=bxyz), gt_boxes (B, max gt boxes in a pc, 8)
                new_batch_dict['point_features'] = torch.cat([output['point_features'] for output in all_outputs], dim=0)
                new_batch_dict['point_coords'] = torch.cat([output['point_coords'] for output in all_outputs], dim=0)
            
            
            new_batch_dict['cluster_ids'] = [] #([N1], ..., [Nbs1], [N1],..., [Nbs2])
            new_batch_dict['gt_boxes_cluster_ids'] = [] #(bs+bs, num gt boxes in each pc)
            for output in all_outputs:
                new_batch_dict['gt_boxes_cluster_ids'] += output['gt_boxes_cluster_ids'] 
                new_batch_dict['cluster_ids'] += output['cluster_ids']

        ###################### gt boxes needed for both detection and aux head ###################
        gt_box_dim = all_outputs[0]['gt_boxes'].shape[-1]
        device = all_outputs[0]['gt_boxes'].device
        max_gt_num = max([output['gt_boxes'].shape[1] for output in all_outputs])
        new_gt_boxes = torch.zeros(
            [new_batch_dict['batch_size'], max_gt_num, gt_box_dim],
            dtype=torch.float32,
            device=device)
        

        batch_idx = 0
        for output in all_outputs:
            bs = output['batch_size']
            gt_boxes = output['gt_boxes']
            num_boxes = gt_boxes.shape[1]
            new_gt_boxes[batch_idx: batch_idx+bs, :num_boxes , :] = gt_boxes
            batch_idx += bs

        new_batch_dict['gt_boxes'] = new_gt_boxes #(bs+bs, max num gt, 8)
        
        ###################### Concat seg feats for Aux head ###################
        # # To store the map to pretext_head_feats (seg features) to cluster ids
        # new_batch_dict['common_cluster_ids'] = output_base_dict['common_cluster_ids']

        # Concat cluster features
        #(num common clusters, 128+128)
        new_batch_dict['pretext_head_feats'] = torch.cat([output['pretext_head_feats'] for output in all_outputs], dim=1)
        new_batch_dict['common_cluster_gtbox_idx'] = output_base_dict['common_cluster_gtbox_idx']
        new_batch_dict['common_cluster_gtbox_idx_moco'] = output_moco_dict['common_cluster_gtbox_idx']
        
        return new_batch_dict
    def _single_input_forward(self, batch_dict, target):
        
        main_utils.load_data_to_gpu(batch_dict)
        output = self.trunk[target](batch_dict)
        return output

    @torch.no_grad()
    def _momentum_update_key(self, target=1):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.trunk[target-1].parameters(), self.trunk[target].parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, batch_dict):
        batch_size_this = batch_dict['batch_size']
        
        points = batch_dict['points'] #(N1+...+Nbs, 5=bxyzi)
        num_pts_batch = np.unique(points[:,0].cpu().numpy(), return_counts=True)[1]
        all_size = concat_all_gather(torch.tensor(num_pts_batch).cuda())
        max_size = torch.max(all_size) #max num voxels in any pc
        points_gather = gather_feats(batch_indices=points[:,0], 
                                        feats_or_coords=points, 
                                        batch_size_this=batch_size_this, 
                                        num_vox_or_pts_batch=num_pts_batch, 
                                        max_num_vox_or_pts=max_size)

        batch_size_all = points_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
    
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
        
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()

        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx] #(num_gpus=4, batch_size_this=2)

        batch_dict['batch_size'] = len(idx_this)
        batch_dict['points'] = get_feats_this(idx_this, all_size, points_gather, is_ind=True) # (N1+..Nbs, bxyzi)
        batch_dict['idx_unshuffle'] = idx_unshuffle

        if self.config['VOX']:
            # Each pc has diff num voxels
            voxel_coords = batch_dict['voxel_coords'] #(N1+..+Nbs, bzyx)
            voxels = batch_dict['voxels'] #(N1+..+Nbs, 5, xyzi)
            voxel_num_points = batch_dict['voxel_num_points'] #(N1+..+Nbs,)

            num_voxels_batch = np.unique(voxel_coords[:,0].cpu().numpy(), return_counts=True)[1]
            all_size = concat_all_gather(torch.tensor(num_voxels_batch).cuda())
            max_size = torch.max(all_size) #max num voxels in any pc

            voxel_coords_gather = gather_feats(batch_indices=voxel_coords[:,0], 
                                            feats_or_coords=voxel_coords, 
                                            batch_size_this=batch_size_this, 
                                            num_vox_or_pts_batch=num_voxels_batch, 
                                            max_num_vox_or_pts=max_size)
            voxels_gather = gather_feats(batch_indices=voxel_coords[:,0], 
                                            feats_or_coords=voxels, 
                                            batch_size_this=batch_size_this, 
                                            num_vox_or_pts_batch=num_voxels_batch, 
                                            max_num_vox_or_pts=max_size)
            voxel_num_points_gather = gather_feats(batch_indices=voxel_coords[:,0], 
                                            feats_or_coords=voxel_num_points, 
                                            batch_size_this=batch_size_this, 
                                            num_vox_or_pts_batch=num_voxels_batch, 
                                            max_num_vox_or_pts=max_size)
            batch_dict['voxels'] = get_feats_this(idx_this, all_size, voxels_gather)
            batch_dict['voxel_num_points'] = get_feats_this(idx_this, all_size, voxel_num_points_gather)
            batch_dict['voxel_coords'] = get_feats_this(idx_this, all_size, voxel_coords_gather, is_ind=True)

        return batch_dict

    @torch.no_grad()
    def _batch_shuffle_ddp_vox(self, batch_dict):
        batch_size_this = batch_dict['batch_size']

        # Each pc has diff num voxels
        voxel_coords = batch_dict['voxel_coords'] #(N1+..+Nbs, bzyx)
        voxels = batch_dict['voxels'] #(N1+..+Nbs, 5, xyzi)
        voxel_num_points = batch_dict['voxel_num_points'] #(N1+..+Nbs,)

        num_voxels_batch = np.unique(voxel_coords[:,0].cpu().numpy(), return_counts=True)[1]
        all_size = concat_all_gather(torch.tensor(num_voxels_batch).cuda())
        max_size = torch.max(all_size) #max num voxels in any pc

        voxel_coords_gather = gather_feats(batch_indices=voxel_coords[:,0], 
                                        feats_or_coords=voxel_coords, 
                                        batch_size_this=batch_size_this, 
                                        num_vox_or_pts_batch=num_voxels_batch, 
                                        max_num_vox_or_pts=max_size)
        voxels_gather = gather_feats(batch_indices=voxel_coords[:,0], 
                                        feats_or_coords=voxels, 
                                        batch_size_this=batch_size_this, 
                                        num_vox_or_pts_batch=num_voxels_batch, 
                                        max_num_vox_or_pts=max_size)
        voxel_num_points_gather = gather_feats(batch_indices=voxel_coords[:,0], 
                                        feats_or_coords=voxel_num_points, 
                                        batch_size_this=batch_size_this, 
                                        num_vox_or_pts_batch=num_voxels_batch, 
                                        max_num_vox_or_pts=max_size)

        batch_size_all = voxels_gather.shape[0] # 12 if 2 gpus bcz bs=6 is per gpu

        num_gpus = batch_size_all // batch_size_this # 12/6=2

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda() # shuffled pc id [5,3,7,15,0,2,  1,9,8,...]

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0) # so that all gpus have this same idx_shuffle

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle) # index of shuffled pc id [4,6,5,1,.,0,....]

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx] #(num_gpus, bs per gpu=6) ->[[5,3,7,15,0,2],[1,9,8, ...]] -> choose gpu_idx'th row of pc ids for this gpu e.g. gpu 0 will get [5,3,7,15,0,2,1,9]

        batch_dict['voxels'] = get_feats_this(idx_this, all_size, voxels_gather)
        batch_dict['voxel_num_points'] = get_feats_this(idx_this, all_size, voxel_num_points_gather)
        batch_dict['voxel_coords'] = get_feats_this(idx_this, all_size, voxel_coords_gather, is_ind=True)
        batch_dict['batch_size'] = len(idx_this)
        batch_dict['idx_unshuffle'] = idx_unshuffle

        
        return batch_dict
    

    @torch.no_grad()
    def _batch_shuffle_ddp_pts(self, batch_dict):
        batch_size_this = batch_dict['batch_size']

        points = batch_dict['points'] #(N1+...+Nbs, 5=bxyzi)
        num_pts_batch = np.unique(points[:,0].cpu().numpy(), return_counts=True)[1]
        all_size = concat_all_gather(torch.tensor(num_pts_batch).cuda())
        max_size = torch.max(all_size) #max num voxels in any pc
        points_gather = gather_feats(batch_indices=points[:,0], 
                                        feats_or_coords=points, 
                                        batch_size_this=batch_size_this, 
                                        num_vox_or_pts_batch=num_pts_batch, 
                                        max_num_vox_or_pts=max_size)



        batch_size_all = points_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
    
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
        
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()

        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx] #(num_gpus=4, batch_size_this=2)

        
        batch_dict['batch_size'] = len(idx_this)
        batch_dict['points'] = get_feats_this(idx_this, all_size, points_gather, is_ind=True) # (N1+..Nbs, bxyzi)
        batch_dict['idx_unshuffle'] = idx_unshuffle

        return batch_dict

    
    def _single_input_forward_MOCO(self, batch_dict, target):

        with torch.no_grad():
            self._momentum_update_key(target)  # update the key encoder
            #shuffle for making use of BN
            main_utils.load_data_to_gpu(batch_dict)

            if torch.distributed.is_initialized():
                batch_dict = self._batch_shuffle_ddp(batch_dict)
                
            # Copy to GPU
            #print("Before loading to gpu: device: {}, batch_size: {}, shape: {}".format(batch_dict["points"].device, batch_dict["batch_size"], batch_dict["points"].shape))
            #main_utils.load_data_to_gpu(batch_dict)
            #print("After loading to gpu: device: {}, batch_size: {}, shape: {}".format(batch_dict["points"].device, batch_dict["batch_size"], batch_dict["points"].shape))

            output = self.trunk[target](batch_dict)
            
            batch_dict.pop('idx_unshuffle', None)
            return output

    def forward(self, batch_dict):
        return self.multi_input_with_head_mapping_forward(batch_dict)

    def _get_head(self, dataset, key):
        import models.trunks as models
        if key in self.config:
            return models.build_network(model_cfg=self.config[key], num_class=len(dataset.class_names), dataset=dataset)
        else:
            return None
    

    def _get_trunk(self, dataset):
        import models.trunks as models
        trunks = torch.nn.ModuleList()
        # if 'arch_point' in self.config:
        #     assert self.config['arch_point'] in models.TRUNKS, 'Unknown model architecture'
        #     trunks.append(models.TRUNKS[self.config['arch_point']](**self.config['args_point'], cluster=self.cluster, linear_probe = self.linear_probe))
        #     trunks.append(models.TRUNKS[self.config['arch_point']](**self.config['args_point'], cluster=self.cluster, linear_probe = self.linear_probe))
        # if 'arch_vox' in self.config:
        #     assert self.config['arch_vox'] in models.TRUNKS, 'Unknown model architecture'
        #     trunks.append(models.TRUNKS[self.config['arch_vox']](**self.config['args_vox']))
        #     trunks.append(models.TRUNKS[self.config['arch_vox']](**self.config['args_vox']))

        trunks.append(models.build_network(model_cfg=self.config["MODEL_BASE"], num_class=len(dataset.class_names), dataset=dataset))
        trunks.append(models.build_network(model_cfg=self.config["MODEL_BASE"], num_class=len(dataset.class_names), dataset=dataset))

        # named_params_q = trunks[0].named_parameters()
        # named_params_k = trunks[1].named_parameters()

        # # TODO: test
        # for name, param in named_params_q:
        #     if 'backbone3d.' in name:
        #         named_params_k[name].copy_(param)
        #         named_params_k[name].requires_grad = False
        
        for param_q, param_k in zip(trunks[0].parameters(), trunks[1].parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # for numh in range(len(trunks)//2):
        #     for param_q, param_k in zip(trunks[numh*2].parameters(), trunks[numh*2+1].parameters()):
        #         param_k.data.copy_(param_q.data)  # initialize
        #         param_k.requires_grad = False  # not update by gradient

        # logger = self.logger
        # for model in trunks:
        #     if logger is not None:
        #         if isinstance(model, (list, tuple)):
        #             logger.add_line("=" * 30 + "   Model   " + "=" * 30)
        #             for m in model:
        #                 logger.add_line(str(m))
        #             logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
        #             for m in model:
        #                 logger.add_line(parameter_description(m))
        #         else:
        #             logger.add_line("=" * 30 + "   Model   " + "=" * 30)
        #             logger.add_line(str(model))
        #             logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
        #             logger.add_line(parameter_description(model))

        #             """
        #             SA_modules.0.mlps.0.0.weight                     (layer 1)             | Trainable  | 16 x 4 x 1 x 1 (size of weight)| 64 (product of size)
        #             SA_modules.0.mlps.0.1.weight                    (batch norm)           | Trainable  | 16                             | 16
        #             SA_modules.0.mlps.0.1.bias                                             | Trainable  | 16                             | 16
        #             """
        return trunks

    
    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def model_depth(self):
        raise NotImplementedError

    def validate(self, dataset_output_shape):
        raise NotImplementedError
