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
    def __init__(self, model_config, cluster, logger, linear_probe=False):
        """
        Class to implement a self-supervised model.
        The model is split into `trunk' that computes features.
        """
        self.config = model_config
        self.linear_probe = linear_probe
        self.logger = logger
        self.cluster = cluster
        super().__init__()
        self.eval_mode = None  # this is just informational
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.trunk = self._get_trunk()
        self.m = 0.999 ### Can be tuned momentum parameter for momentum update
        self.model_input = model_config["model_input"] #['points', 'points_moco']
        
    def multi_input_with_head_mapping_forward(self, batch):
        all_outputs = []
        for input_idx in range(len(self.model_input)): #['points', 'points_moco']
            input_key = self.model_input[input_idx]
            if "moco" in input_key:
                outputs = self._single_input_forward_MOCO(batch[input_key], input_key, input_idx, batch.get(input_key + "_cluster", None))
            else:
                outputs = self._single_input_forward(batch[input_key], input_key, input_idx, batch.get(input_key + "_cluster", None))
            all_outputs.append(outputs)
        return all_outputs
    
    def _single_input_forward(self, batch, input_key, target, cluster_ids=None):
        if "vox" not in input_key:
            assert isinstance(batch, torch.Tensor)

        # if ('vox' in input_key) and ("Lidar" not in self.config):
        #     points = batch
        #     points_coords = points[0]
        #     points_feats = points[1]

        #     ### Invariant to even and odd coords
        #     points_coords[:, 1:] += (torch.rand(3) * 100).type_as(points_coords)
        #     points_feats = points_feats/255.0 - 0.5

        #     batch = SparseTensor(points_feats, points_coords.float())

        if ('vox' in input_key) and ("Lidar" in self.config):
            # Copy to GPU
            for key in batch:
                batch[key] = main_utils.recursive_copy_to_gpu(
                    batch[key], non_blocking=True
                )
        else:
            # Copy to GPU
            batch = main_utils.recursive_copy_to_gpu(
                batch, non_blocking=True
            )
        
        output = self.trunk[target](batch, cluster_ids)
        return output

    @torch.no_grad()
    def _momentum_update_key(self, target=1):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.trunk[target-1].parameters(), self.trunk[target].parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, vox=False, idx_shuffle=None):
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

    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
        
    #     num_gpus = batch_size_all // batch_size_this
        
    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    #     return x_gather[idx_this]
    
    def _single_input_forward_MOCO(self, batch, input_key, target, cluster_ids=None):
        if "vox" not in input_key:
            assert isinstance(batch, torch.Tensor)
        if ('vox' in input_key) and ("Lidar" not in self.config):
            points = batch
            points_coords = points[0]
            points_feats = points[1]

            ### Invariant to even and odd coords
            points_coords[:, 1:] += (torch.rand(3) * 100).type_as(points_coords)
            points_feats = points_feats/255.0 - 0.5
            ### If enable shuffle batch for vox, please comment out this line.
            batch = SparseTensor(points_feats, points_coords.float())

        with torch.no_grad():
            self._momentum_update_key(target)  # update the key encoder
            #shuffle for making use of BN
            if torch.distributed.is_initialized():
                if "vox" not in input_key: 
                    batch, idx_unshuffle = self._batch_shuffle_ddp(batch, vox=False)
            else:
                if ('vox' in input_key) and ("Lidar" not in self.config):
                    batch = SparseTensor(points_feats, points_coords.float())
                
            # Copy to GPU
            if ("Lidar" in self.config) and ("vox" in input_key):
                for key in batch:
                    batch[key] = main_utils.recursive_copy_to_gpu(
                        batch[key], non_blocking=True
                    )
            else:
                batch = main_utils.recursive_copy_to_gpu(
                    batch, non_blocking=True
                )

            output = self.trunk[target](batch, cluster_ids, idx_unshuffle)
            # idx_unshuffle assumes that output features.shape[0] is 8 i.e. equal to number of pcs in the batch
            # Not compatible with Voxelized DC!
            # if torch.distributed.is_initialized():
            #     if "vox" not in input_key:
            #         if 'dc_feats' in feature_names:
            #             output['dc_feats'] = [self._batch_unshuffle_ddp(output['dc_feats'], idx_unshuffle)]
            #         if 'vdc_feats' in feature_names:
            #             output['vdc_feats'] = [self._batch_unshuffle_ddp(output['vdc_feats'], idx_unshuffle)]
            #         if self.linear_probe:
            #             output['linear_probe_feats'] = [self._batch_unshuffle_ddp(output['linear_probe_feats'], idx_unshuffle)]
            #     return output
            # else:
            #     return output
            return output

    def forward(self, batch):
        return self.multi_input_with_head_mapping_forward(batch)

    def _get_trunk(self):
        import models.trunks as models
        trunks = torch.nn.ModuleList()
        if 'arch_point' in self.config:
            assert self.config['arch_point'] in models.TRUNKS, 'Unknown model architecture'
            trunks.append(models.TRUNKS[self.config['arch_point']](**self.config['args_point'], cluster=self.cluster, linear_probe = self.linear_probe))
            trunks.append(models.TRUNKS[self.config['arch_point']](**self.config['args_point'], cluster=self.cluster, linear_probe = self.linear_probe))
        if 'arch_vox' in self.config:
            assert self.config['arch_vox'] in models.TRUNKS, 'Unknown model architecture'
            trunks.append(models.TRUNKS[self.config['arch_vox']](**self.config['args_vox']))
            trunks.append(models.TRUNKS[self.config['arch_vox']](**self.config['args_vox']))

        for numh in range(len(trunks)//2):
            for param_q, param_k in zip(trunks[numh*2].parameters(), trunks[numh*2+1].parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

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

    def _print_state_dict_shapes(self, state_dict):
        logging.info("Model state_dict:")
        for param_tensor in state_dict.keys():
            logging.info(f"{param_tensor}:\t{state_dict[param_tensor].size()}")

    def _print_loaded_dict_info(self, state):
        # get the model state dict original
        model_state_dict = {}
        if "," in self.config.TRUNK.NAME:
            trunk_state_dict, heads_state_dict = (
                self.trunk.state_dict(),
                self.heads.state_dict(),
            )
        else:
            trunk_state_dict, heads_state_dict = (
                self.trunk.state_dict(),
                self.heads.state_dict(),
            )
        model_state_dict.update(trunk_state_dict)
        model_state_dict.update(heads_state_dict)

        # get the checkpoint state dict
        checkpoint_state_dict = {}
        checkpoint_state_dict.update(state["trunk"])
        checkpoint_state_dict.update(state["heads"])

        # now we compare the state dict and print information
        not_found, extra_layers = [], []
        max_len_model = max(len(key) for key in model_state_dict.keys())
        for layername in model_state_dict.keys():
            if layername in checkpoint_state_dict:
                logging.info(
                    f"Loaded: {layername: <{max_len_model}} of "
                    f"shape: {model_state_dict[layername].size()} from checkpoint"
                )
            else:
                not_found.append(layername)
                logging.info(f"Not found:\t\t{layername}, not initialized")
        for layername in checkpoint_state_dict.keys():
            if layername not in model_state_dict:
                extra_layers.append(layername)
        logging.info(f"Extra layers not loaded from checkpoint:\n {extra_layers}")

    def get_optimizer_params(self):
        regularized_params, unregularized_params = [], []
        conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            apex.parallel.SyncBatchNorm,
        )
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, conv_types):
                regularized_params.append(module.weight)
                if module.bias is not None:
                    if self.optimizer_config["regularize_bias"]:
                        regularized_params.append(module.bias)
                    else:
                        unregularized_params.append(module.bias)
            elif isinstance(module, bn_types):
                if module.weight is not None:
                    if self.optimizer_config["regularize_bn"]:
                        regularized_params.append(module.weight)
                    else:
                        unregularized_params.append(module.weight)
                if module.bias is not None:
                    if (
                        self.optimizer_config["regularize_bn"]
                        and self.optimizer_config["regularize_bias"]
                    ):
                        regularized_params.append(module.bias)
                    else:
                        unregularized_params.append(module.bias)
            elif len(list(module.children())) >= 0:
                # for any other layers not bn_types, conv_types or nn.Linear, if
                # the layers are the leaf nodes and have parameters, we regularize
                # them. Similarly, if non-leaf nodes but have parameters, regularize
                # them (set recurse=False)
                for params in module.parameters(recurse=False):
                    regularized_params.append(params)

        non_trainable_params = []
        for name, param in self.named_parameters():
            if name in cfg.MODEL.NON_TRAINABLE_PARAMS:
                param.requires_grad = False
                non_trainable_params.append(param)

        trainable_params = [
            params for params in self.parameters() if params.requires_grad
        ]
        regularized_params = [
            params for params in regularized_params if params.requires_grad
        ]
        unregularized_params = [
            params for params in unregularized_params if params.requires_grad
        ]
        logging.info("Traininable params: {}".format(len(trainable_params)))
        logging.info("Non-Traininable params: {}".format(len(non_trainable_params)))
        logging.info(
            "Regularized Parameters: {}. Unregularized Parameters {}".format(
                len(regularized_params), len(unregularized_params)
            )
        )
        return {
            "regularized_params": regularized_params,
            "unregularized_params": unregularized_params,
        }

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
