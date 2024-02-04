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

from datasets.collators.sparse_collator import numpy_to_sparse_tensor, collate_points_to_sparse_tensor
from utils import main_utils
from third_party.OpenPCDet.pcdet.models.pretext_heads.gather_utils import *
from third_party.OpenPCDet.pcdet.models.detectors import build_detector
from models.SegmentationHead import SegmentationClassifierHead 

def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(np.prod(p.size())))
    return desc


class BaseSSLMultiInputOutputModel(nn.Module):
    def __init__(self, model_config, pretraining, dataset, logger):
        """
        Class to implement a self-supervised model.
        The model is split into `trunk' that computes features.
        """
        self.config = model_config
        self.logger = logger
        self.pretraining = pretraining
        super().__init__()
        self.trunk = self._get_trunk(dataset)
        self.det_head = self._get_head(dataset, key="MODEL_DET_HEAD")
        self.aux_head = self._get_head(dataset, key="MODEL_AUX_HEAD")
        if 'SEGMENTATION_HEAD' in model_config:
            self.segmentation_head = SegmentationClassifierHead(model_config['SEGMENTATION_HEAD'], dataset.point_cloud_range, dataset.voxel_size)
        self.m = 0.999 ### Can be tuned momentum parameter for momentum update
        # self.weight_initialization()

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.SyncBatchNorm):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def downstream_forward(self, batch_dict): 
        output_dict = {}
        if self.config['INPUT'] == 'sparse_tensor':
            batch_dict['input']['sparse_points'] = numpy_to_sparse_tensor(batch_dict['input']['voxel_coords'], batch_dict['input']['points']) # sparse gpu tensors -> (C:(8, 20k, 4=b_id, xyz voxcoord), F:(8, 20k, 4=xyzi pts))
            batch_dict['input'].pop('points')
            batch_dict['input'].pop('voxel_coords')
            
        outputs, _, _, _ = self._single_input_forward(batch_dict['input'])
        if 'points' in outputs['batch_dict']:
            outputs['batch_dict'].pop('points')
        output_dict['output'] =  outputs['batch_dict']
        
        if 'MODEL_DET_HEAD' in self.config:
            outputs_head, tb_dict, _, _ = self.det_head(output_dict['output'])

            output_dict['loss_det_head']= outputs_head['loss']
            if self.config['INPUT'] == 'voxels':
                output_dict['loss_det_cls'] = tb_dict.get('hm_loss_head_0', 0)
                output_dict['loss_det_reg'] = tb_dict.get('loc_loss_head_0', 0)
            else:
                output_dict['loss_det_cls'] = tb_dict.get('point_loss_cls', 0) #TODO
                output_dict['loss_det_reg'] = tb_dict.get('point_loss_box', 0)
                output_dict['loss_det_cls_rcnn'] = tb_dict.get('rcnn_loss_cls', 0) #TODO
                output_dict['loss_det_reg_rcnn'] = tb_dict.get('rcnn_loss_reg', 0)
        
        if 'SEGMENTATION_HEAD' in self.config:
            output_dict['loss_seg'], loss_seg_dict, output_dict['output']['pred_labels'], output_dict['output']['seg_labels'] = self.segmentation_head(output_dict['output'])
            for type in self.segmentation_head.loss_types:
                output_dict[f'loss_seg_{type}'] = loss_seg_dict[type].item()

        return output_dict

    def pretrain_forward(self, batch_dict):
        output_dict = {}

        if self.config['INPUT'] == 'sparse_tensor':
            batch_dict['input']['sparse_points'], batch_dict['input_moco']['sparse_points'] = collate_points_to_sparse_tensor(batch_dict['input']['voxel_coords'], batch_dict['input']['points'], 
                                                                                batch_dict['input_moco']['voxel_coords'], batch_dict['input_moco']['points']) #xi and xj are sparse tensors for normal and moco pts -> (C:(8, 20k, 4=b_id, xyz voxcoord), F:(8, 20k, 4=xyzi pts))
            batch_dict['input']['point_coords'] = torch.cat((batch_dict['input']['sparse_points'].C[:,0].unsqueeze(-1), batch_dict['input']['sparse_points'].F[:,:3]), dim=1) #bid, xyz
            batch_dict['input_moco']['point_coords'] = torch.cat((batch_dict['input_moco']['sparse_points'].C[:,0].unsqueeze(-1), batch_dict['input_moco']['sparse_points'].F[:,:3]), dim=1) #bid, xyz
            
            batch_dict['input'].pop('points')
            batch_dict['input'].pop('voxel_coords')
            batch_dict['input_moco'].pop('points')
            batch_dict['input_moco'].pop('voxel_coords')
            # assert batch_dict['input']['points'].dtype == 'float32'
            #assert len(batch_dict['input']['points'].shape) == 3

        outputs, _, _ , _= self._single_input_forward(batch_dict['input'])

        if 'points' in outputs['batch_dict']:
            outputs['batch_dict'].pop('points')
        output_dict['output'] =  outputs['batch_dict']

        outputs_moco, _, _, _ = self._single_input_forward_MOCO(batch_dict['input_moco'])

        if 'points' in outputs_moco['batch_dict']:
            outputs_moco['batch_dict'].pop('points')
        output_dict['output_moco'] =  outputs_moco['batch_dict']
        
        concat_outputs_condition_for_det = 'MODEL_DET_HEAD' in self.config and self.config['MODEL_DET_HEAD'].get('INPUT_MOCO_FEATS', False)
        concat_outputs_condition = concat_outputs_condition_for_det or 'MODEL_AUX_HEAD' in self.config
        if concat_outputs_condition:
            new_batch_dict = self._concat_encoder_outputs(output_dict['output'], output_dict['output_moco'], for_det=concat_outputs_condition_for_det)

        
        if 'MODEL_DET_HEAD' in self.config:
            if self.config['MODEL_DET_HEAD'].get('INPUT_MOCO_FEATS', False):
                outputs_head, tb_dict, _, _ = self.det_head(new_batch_dict) 
            else:
                outputs_head, tb_dict, _, _ = self.det_head(output_dict['output'])
            
            output_dict['loss_det_head']= outputs_head['loss']
            if self.config['INPUT'] == 'voxels':
                output_dict['loss_det_cls'] = tb_dict.get('hm_loss_head_0', 0)
                output_dict['loss_det_reg'] = tb_dict.get('loc_loss_head_0', 0)
            else:
                output_dict['loss_det_cls'] = tb_dict.get('point_loss_cls', 0) #TODO
                output_dict['loss_det_reg'] = tb_dict.get('point_loss_box', 0)
                output_dict['loss_det_cls_rcnn'] = tb_dict.get('rcnn_loss_cls', 0) #TODO
                output_dict['loss_det_reg_rcnn'] = tb_dict.get('rcnn_loss_reg', 0)


        if 'MODEL_AUX_HEAD' in self.config:
            outputs_head, _, _, _ = self.aux_head(new_batch_dict)
            output_dict['loss_aux_head']= outputs_head['batch_dict']['seg_reg_loss']
            output_dict['loss_aux_head_rot']= outputs_head['batch_dict']['seg_reg_loss_rot']
            output_dict['loss_aux_head_scale']= outputs_head['batch_dict']['seg_reg_loss_scale']

        return output_dict
    
    def _concat_encoder_outputs(self, output_base_dict, output_moco_dict, for_det):
        all_outputs = []
        new_batch_dict = {}

        all_outputs.append(output_base_dict)
        all_outputs.append(output_moco_dict)
        new_batch_dict['batch_size'] = all_outputs[1]['batch_size'] + all_outputs[0]['batch_size']


        if for_det:
            # All inputs needed only for the detection head
            #new_batch_dict['points'] = torch.cat([output['points'] for output in all_outputs], dim=0)

            if self.config['INPUT'] == 'voxels':
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
        
        if 'common_cluster_gtbox_idx' in output_base_dict:
            new_batch_dict['common_cluster_gtbox_idx'] = output_base_dict['common_cluster_gtbox_idx']
            new_batch_dict['common_cluster_gtbox_idx_moco'] = output_moco_dict['common_cluster_gtbox_idx']
            
        return new_batch_dict
    def _single_input_forward(self, batch_dict):
        
        main_utils.load_data_to_gpu(batch_dict)

        output = self.trunk[0](batch_dict)
        return output

    @torch.no_grad()
    def _momentum_update_key(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.trunk[0].parameters(), self.trunk[1].parameters()):
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

        if self.config['INPUT'] == 'voxels':
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
    def _batch_shuffle_ddp_sparse(self, batch_dict):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size = []
        x = batch_dict['sparse_points']
        # sparse tensor should be decomposed
        c, f = x.decomposed_coordinates_and_features #c=list of bs=8 (20k, 3=xyz vox coord), f=list of bs=8 (20k, 4=xyzi pts)

        # each pcd has different size, get the biggest size as default
        newx = list(zip(c, f)) #list of bs=8 [(c,f for pc 1),..,(c,f for pc 8)]
        for bidx in newx:
            batch_size.append(len(bidx[0])) #list of bs=8 [20k, ..., 20k]
        all_size = concat_all_gather(torch.tensor(batch_size).cuda()) # get num pts in each pc from all gpus e.g. if 2 gpus with bs=8 per gpu then all_size = list of 16 =[20k, ..., 20k]
        max_size = torch.max(all_size) #20k = max num pts in all pcs in all gpus

        # create a tensor with shape (batch_size, max_size)
        # copy each sparse tensor data to the begining of the biggest sized tensor
        shuffle_c = [] #list of bs=8 ([max_numpts_in_any_pc=20k, 3], ..., [max_numpts_in_any_pc=20k, 3])
        shuffle_f = [] #list of bs=8 ([max_numpts_in_any_pc=20k, 4], ..., [max_numpts_in_any_pc=20k, 4])
        for bidx in range(len(newx)):
            shuffle_c.append(torch.ones((max_size, newx[bidx][0].shape[-1])).cuda()) #(20k or max num pts in any pc, 3=xyz vox cooord)
            shuffle_c[bidx][:len(newx[bidx][0]),:] = newx[bidx][0]

            shuffle_f.append(torch.ones((max_size, newx[bidx][1].shape[-1])).cuda())
            shuffle_f[bidx][:len(newx[bidx][1]),:] = newx[bidx][1]

        batch_size_this = len(newx) # 8 pcs

        shuffle_c = torch.stack(shuffle_c) # [8, max num pts in any pc = 20k, 3 xyz vox coord]
        shuffle_f = torch.stack(shuffle_f) # [8, max num pts in any pc = 20k, 4 xyzi pts]

        # gather all the ddp batches pcds
        c_gather = concat_all_gather(shuffle_c) # if 2 gpus [8x2=16, max_pts=20k, 3]
        f_gather = concat_all_gather(shuffle_f) # if 2 gpus [8x2=16, max_pts=20k, 4]

        batch_size_all = c_gather.shape[0] # 16 if 2 gpus bcz bs=8 is per gpu

        num_gpus = batch_size_all // batch_size_this # 16/8=2

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda() # shuffled pc id [5,3,7,15,0,2,1,9,  8,...]

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0) # so that all gpus have this same idx_shuffle

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle) # index of shuffled pc id [4,6,5,1,.,0,....]

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx] #(num_gpus, bs per gpu=8) ->[[5,3,7,15,0,2,1,9],[8, ...]] -> choose gpu_idx'th row of pc ids for this gpu e.g. gpu 0 will get [5,3,7,15,0,2,1,9]

        c_this = []
        f_this = []

        # after shuffling we get only the actual information of each tensor
        # :actual_size is the information, actual_size:biggest_size are just ones (ignore)
        for idx in range(len(idx_this)):
            c_this.append(c_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy()) #c_gather is [16, 20k, 3] -> pick the [pc id for this gpu, actual num pts in this pc, 3] 
            f_this.append(f_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())

        # final shuffled coordinates and features, build back the sparse tensor
        # c_this = np.array(c_this) #(8, 20k, 3=xyz vox coord)
        # f_this = np.array(f_this) #(8, 20k, 4=xyzi pts)
        x_this = numpy_to_sparse_tensor(c_this, f_this) # sparse tensor on gpu (C:(8x20k, 4=new_bid on this gpu, xyz vox coord), F:(8x20k, 4=xyzi pts))

        batch_dict['sparse_points'] = x_this
        batch_dict['idx_unshuffle'] = idx_unshuffle
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

    
    def _single_input_forward_MOCO(self, batch_dict):

        with torch.no_grad():
            self._momentum_update_key()  # update the key encoder
            #shuffle for making use of BN
            main_utils.load_data_to_gpu(batch_dict)

            if torch.distributed.is_initialized():
                if self.config['INPUT'] == 'voxels':
                    batch_dict = self._batch_shuffle_ddp_vox(batch_dict)
                elif self.config['INPUT'] == 'sparse_tensor':
                    batch_dict = self._batch_shuffle_ddp_sparse(batch_dict)
                else:
                    batch_dict = self._batch_shuffle_ddp_pts(batch_dict)

                
            # Copy to GPU
            #print("Before loading to gpu: device: {}, batch_size: {}, shape: {}".format(batch_dict["points"].device, batch_dict["batch_size"], batch_dict["points"].shape))
            #main_utils.load_data_to_gpu(batch_dict)
            #print("After loading to gpu: device: {}, batch_size: {}, shape: {}".format(batch_dict["points"].device, batch_dict["batch_size"], batch_dict["points"].shape))

            output = self.trunk[1](batch_dict)
            
            batch_dict.pop('idx_unshuffle', None)
            return output

    def forward(self, batch_dict):
        if self.pretraining:
            return self.pretrain_forward(batch_dict)
        else:
            return self.downstream_forward(batch_dict)

    def _get_head(self, dataset, key):
        if key in self.config:
            return build_network(model_cfg=self.config[key], num_class=len(dataset.class_names), dataset=dataset)
        else:
            return None
    

    def _get_trunk(self, dataset):
        trunks = torch.nn.ModuleList()

        trunks.append(build_network(model_cfg=self.config["MODEL_BASE"], num_class=len(dataset.class_names), dataset=dataset))
        if self.pretraining:
            trunks.append(build_network(model_cfg=self.config["MODEL_BASE"], num_class=len(dataset.class_names), dataset=dataset))

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
