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

def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(np.prod(p.size())))
    return desc

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

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
        # self.eval_mode = None  # this is just informational
        # self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.trunk = self._get_trunk(dataset)
        self.det_head = self._get_head(dataset, key="MODEL_DET_HEAD")
        self.aux_head = self._get_head(dataset, key="MODEL_AUX_HEAD")
        self.m = 0.999 ### Can be tuned momentum parameter for momentum update
        # self.model_input = model_config["model_input"] #['points', 'points_moco']
        
    def multi_input_with_head_mapping_forward(self, batch_dict):
        output_dict = {}

        outputs, _, _ = self._single_input_forward(batch_dict['input'], 0)
        outputs['batch_dict'].pop('points')
        output_dict['output'] =  outputs['batch_dict']

        outputs_moco, _, _ = self._single_input_forward_MOCO(batch_dict['input_moco'], 1)
        outputs_moco['batch_dict'].pop('points')
        output_dict['output_moco'] =  outputs_moco['batch_dict']
        

        if 'MODEL_DET_HEAD' or 'MODEL_AUX_HEAD' in self.config:
            new_batch_dict = self._concat_encoder_outputs(output_dict['output'], output_dict['output_moco'], for_det=self.config.get('MODEL_DET_HEAD', False))

        
        if 'MODEL_DET_HEAD' in self.config:
            if self.config['MODEL_DET_HEAD'].get('INPUT_MOCO_FEATS', False):
                outputs_head, _, _ = self.det_head(new_batch_dict) 
            else:
                outputs_head, _, _ = self.det_head(output_dict['output_base'])
            
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
            new_batch_dict['batch_size'] = all_outputs[0]['batch_size']
            all_outputs[1]['point_coords'][:,0] += all_outputs[0]['batch_size']
            new_batch_dict['batch_size'] += all_outputs[1]['batch_size']

            #points (N, 4=xyzi), point_features (N, 128), point_coords (N, 4=bxyz), gt_boxes (B, max gt boxes in a pc, 8)
            #new_batch_dict['points'] = torch.cat([output['points'] for output in all_outputs], dim=0)
            new_batch_dict['point_features'] = torch.cat([output['point_features'] for output in all_outputs], dim=0)
            new_batch_dict['point_coords'] = torch.cat([output['point_coords'] for output in all_outputs], dim=0)
            new_batch_dict['cluster_ids'] = np.vstack([output['cluster_ids'] for output in all_outputs]) #(bs1+bs1, 20000)
            new_batch_dict['gt_boxes_cluster_ids'] = [] #(bs+bs, num gt boxes in each pc)
            for output in all_outputs:
                new_batch_dict['gt_boxes_cluster_ids'] += output['gt_boxes_cluster_ids'] 

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
        points = batch_dict['points'] #(6, 20k, 4)
        batch_size_this = points.shape[0] #batch_dict['batch_size'] # 
        #gpu_idx = torch.distributed.get_rank()
        #print(f'Inside shuffle, this pc batch size: {batch_size_this}, gpu_idx: {gpu_idx}')
        # Gather all pcs from all gpus
        pcs_gather = torch.from_numpy(points).float().cuda() # (6, 20k, 4) 
        #print(f'Inside shuffle, this pc: {pcs_gather.shape}, {pcs_gather.device}')
        all_pcs = concat_all_gather(pcs_gather) # (batch_size_all=24, 20000, 4)
        #print(f'Inside shuffle, all pcs: {all_pcs.shape}, {all_pcs.device}, {pcs_gather.device}')


        batch_size_all = all_pcs.shape[0]
        num_gpus = batch_size_all // batch_size_this
        #print(f'Inside shuffle, num_gpus: {num_gpus}, {all_pcs.device}')

        
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        #print(f'Inside shuffle, idx_shuffle: {idx_shuffle}, {all_pcs.device}')
    
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
        
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        #print(f'Inside shuffle, idx_unshuffle: {idx_unshuffle}, {all_pcs.device}')

        
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        #print(f'Inside shuffle, gpu_idx: {gpu_idx}, {all_pcs.device}')

        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx] #(num_gpus=4, batch_size_this=2)
        # print(f'Inside shuffle, idx_shuffle reshaped: {idx_shuffle.view(num_gpus, -1).shape}, {all_pcs.device}')
        # print(f'Inside shuffle, idx_this: {idx_this}, {all_pcs.device}')



        # Return new pcs for this gpu
        new_pcs_this = all_pcs[idx_this] #(6, 20000, 4) 
        #print(f'Inside shuffle, new_pcs_this: {new_pcs_this.shape}, {new_pcs_this.device}')
        
        batch_dict['batch_size'] = new_pcs_this.shape[0]
        batch_dict['points'] = new_pcs_this # (6, 20000, 4)
        #print(f'Inside shuffle, points: {points.shape}')

        batch_dict['idx_unshuffle'] = idx_unshuffle

        #torch.distributed.barrier() #TODO: remove

        return batch_dict


    @torch.no_grad()
    def _batch_shuffle_ddp_old(self, x, vox=False, idx_shuffle=None):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        if vox:
            batch_size = []
            for bidx in x:
                batch_size.append(len(bidx))
            all_size = concat_all_gather(torch.tensor(batch_size).cuda())
            max_size = torch.max(all_size)

            ### Change the new size here
            newx = []
            for bidx in range(len(x)):
                newx.append(torch.ones((max_size, x[bidx].shape[1])).cuda())
                newx[bidx][:len(x[bidx]),:] = x[bidx]
            newx = torch.stack(newx)
            batch_size_this = newx.shape[0]
        else:
            batch_size_this = x.shape[0]

        if vox:
            x_gather = concat_all_gather(newx)
        else:
            x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        if idx_shuffle == None:
            # random shuffle index
            idx_shuffle = torch.randperm(batch_size_all).cuda()
        
            # broadcast to all gpus
            torch.distributed.broadcast(idx_shuffle, src=0)
        
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        if vox:
            ret_x = []
            batch_idx = []
            for idx in range(len(idx_this)):
                if x_gather.shape[-1] == 4:
                    ### Change the batch index here
                    tempdata = x_gather[idx_this[idx]][:all_size[idx_this[idx]],:]
                    tempdata[:,0] = idx
                    ret_x.append(tempdata)
                else:
                    ret_x.append(x_gather[idx_this[idx]][:all_size[idx_this[idx]],:])
            ret_x = torch.cat(ret_x)
            return ret_x, idx_unshuffle, idx_shuffle
        else:
            return x_gather[idx_this], idx_unshuffle
    
    def _single_input_forward_MOCO(self, batch_dict, target):

        with torch.no_grad():
            self._momentum_update_key(target)  # update the key encoder
            #shuffle for making use of BN
            if torch.distributed.is_initialized():
                #if "vox" not in input_key: 
                batch_dict = self._batch_shuffle_ddp(batch_dict)
                
            # Copy to GPU
            #print("Before loading to gpu: device: {}, batch_size: {}, shape: {}".format(batch_dict["points"].device, batch_dict["batch_size"], batch_dict["points"].shape))
            main_utils.load_data_to_gpu(batch_dict)
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
