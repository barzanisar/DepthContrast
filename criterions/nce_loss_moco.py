#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import math
import pprint

import numpy as np
import torch
from torch import nn

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    torch.dist.all_gather => Gathers key embeddings = normalized output 2 from all gpus in a list
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not (torch.distributed.is_initialized()):
        return tensor
    
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0) # stack key embeddings vertically hcat[(batch size for gpu 1 x 128), (b2 x 128), (b3 x 128), (b4 x 128)] ==> [(row dim = b1+b2+b3+b4 , col dim = 128)]
    print("concat_all_gather each gpu's normalized output 2 shape")
    for each_tensor in tensors_gather:
        print(each_tensor.shape)
    print("concat_all_gather output shape ", output.shape)
    return output

class NCELossMoco(nn.Module):
    """
    Distributed version of the NCE loss. It performs an "all_gather" to gather
    the allocated buffers like memory no a single gpu. For this, Pytorch distributed
    backend is used. If using NCCL, one must ensure that all the buffer are on GPU.
    This class supports training using both NCE and CrossEntropy (InfoNCE).
    """

    def __init__(self, config):
        super(NCELossMoco, self).__init__()

        assert config["NCE_LOSS"]["LOSS_TYPE"] in [
            "cross_entropy",
        ], f"Supported types are cross_entropy."

        self.loss_type = config["NCE_LOSS"]["LOSS_TYPE"]
        self.loss_list = config["LOSS_TYPE"].split(",")
        self.other_queue = config["OTHER_INPUT"]

        self.npid0_w = float(config["within_format_weight0"])
        self.npid1_w = float(config["within_format_weight1"])
        self.cmc0_w = float(config["across_format_weight0"])
        self.cmc1_w = float(config["across_format_weight1"])
        
        self.K = int(config["NCE_LOSS"]["NUM_NEGATIVES"])
        self.dim = int(config["NCE_LOSS"]["EMBEDDING_DIM"])
        self.T = float(config["NCE_LOSS"]["TEMPERATURE"])

        self.register_buffer("queue", torch.randn(self.dim, self.K)) # queue of pointnet based negative key embeddings
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        if self.other_queue:
            self.register_buffer("queue_other", torch.randn(self.dim, self.K))  #queue of UNET based negative key embeddings
            self.queue_other = nn.functional.normalize(self.queue_other, dim=0)
            self.register_buffer("queue_other_ptr", torch.zeros(1, dtype=torch.long))
            
        # cross-entropy loss. Also called InfoNCE
        self.xe_criterion = nn.CrossEntropyLoss()

        # other constants
        self.normalize_embedding = config["NCE_LOSS"]["NORM_EMBEDDING"]

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, okeys=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity
        if (ptr + batch_size) > self.queue.shape[1]:
            batch_size = self.queue.shape[1] - ptr

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = torch.transpose(keys, 0, 1)[:, 0:batch_size] #we could also pick randomly batch size num voxels
        ptr = (ptr + batch_size) % self.K  # move pointer
        
        self.queue_ptr[0] = ptr

        if self.other_queue:
            # gather keys before updating queue
            okeys = concat_all_gather(okeys)
                
            other_ptr = int(self.queue_other_ptr)
        
            # replace the keys at ptr (dequeue and enqueue)
            self.queue_other[:, other_ptr:other_ptr + batch_size] = torch.transpose(okeys, 0, 1)#okeys.T
            other_ptr = (other_ptr + batch_size) % self.K  # move pointer
        
            self.queue_other_ptr[0] = other_ptr
                                                                        
    def forward(self, output, vox_coords):
        assert isinstance(
            output, list
        ), "Model output should be a list of tensors. Got Type {}".format(type(output))
        
        if self.normalize_embedding:
            normalized_output1 = nn.functional.normalize(output[0], dim=1, p=2) #query pointnet embedding
            normalized_output2 = nn.functional.normalize(output[1], dim=1, p=2) #key pointnet embedding
            if self.other_queue:
                normalized_output3 = nn.functional.normalize(output[2], dim=1, p=2) #query unet embedding
                normalized_output4 = nn.functional.normalize(output[3], dim=1, p=2) #key unet embedding

        # Voxelized Depth Contrast
        if vox_coords[0] is not None:
            coords_0 = vox_coords[0] # vox_coords from query encoder = list of len total num voxels=1181, each element has a numpy array with a voxel coord = bzyx
            coords_1 = vox_coords[1] # vox_coords from key encoder

            # Find matching vox_coords in both in order to rearrange normalized output1 and output2 in the order of matching voxel features
            map_coord0_idx0 = {str(coords_0[i, :]): i for i in range(coords_0.shape[0])}
            matched_idx_0 = []
            matched_idx_1 = []
            for idx_1 in range(coords_1.shape[0]):
                if str(coords_1[idx_1, :]) in map_coord0_idx0:
                    idx_0 = map_coord0_idx0[str(coords_1[idx_1, :])]
                    matched_idx_0.append(idx_0)
                    matched_idx_1.append(idx_1)

            normalized_output1 = normalized_output1[matched_idx_0, :] #(matched num voxels, 128)
            normalized_output2 = normalized_output2[matched_idx_1, :] #(matched num voxels, 128)

        # positive logits: Nx1 = batch size of positive examples (or matched num voxels)x 1
        l_pos = torch.einsum('nc,nc->n', [normalized_output1, normalized_output2]).unsqueeze(-1) #(v_i_1).transpose() * v_i_2 => dim (n = matched num voxels)
        
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [normalized_output1, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # apply temperature
        logits /= self.T

        if self.other_queue:
            
            l_pos_p2i = torch.einsum('nc,nc->n', [normalized_output1, normalized_output4]).unsqueeze(-1)
            l_neg_p2i = torch.einsum('nc,ck->nk', [normalized_output1, self.queue_other.clone().detach()])
            logits_p2i = torch.cat([l_pos_p2i, l_neg_p2i], dim=1)
            logits_p2i /= self.T

            
            l_pos_i2p = torch.einsum('nc,nc->n', [normalized_output3, normalized_output2]).unsqueeze(-1)
            l_neg_i2p = torch.einsum('nc,ck->nk', [normalized_output3, self.queue.clone().detach()])
            logits_i2p = torch.cat([l_pos_i2p, l_neg_i2p], dim=1)
            logits_i2p /= self.T

            
            l_pos_other = torch.einsum('nc,nc->n', [normalized_output3, normalized_output4]).unsqueeze(-1)
            l_neg_other = torch.einsum('nc,ck->nk', [normalized_output3, self.queue_other.clone().detach()])
            logits_other = torch.cat([l_pos_other, l_neg_other], dim=1)
            logits_other /= (self.T)
            
        if self.other_queue:
            self._dequeue_and_enqueue(normalized_output2, okeys=normalized_output4)
        else:
            self._dequeue_and_enqueue(normalized_output2)

        
        labels = torch.zeros(
            logits.shape[0], device=logits.device, dtype=torch.int64
        ) # because zero'th class is the true class
        
        loss_npid = self.xe_criterion(torch.squeeze(logits), labels) #loss between pointnet query and key embedding

        loss_npid_other = torch.tensor(0)
        loss_cmc_p2i = torch.tensor(0)
        loss_cmc_i2p = torch.tensor(0)
        
        if self.other_queue:
            loss_cmc_p2i = self.xe_criterion(torch.squeeze(logits_p2i), labels) #loss between pointnet query and unet key embedding
            loss_cmc_i2p = self.xe_criterion(torch.squeeze(logits_i2p), labels) #loss between unet query and pointnet key embedding
            loss_npid_other = self.xe_criterion(torch.squeeze(logits_other), labels) #loss between unet query and key embedding
            
            curr_loss = 0
            for ltype in self.loss_list:
                if ltype == "CMC":
                    curr_loss += loss_cmc_p2i * self.cmc0_w + loss_cmc_i2p * self.cmc1_w
                elif ltype == "NPID":
                    curr_loss += loss_npid * self.npid0_w
                    curr_loss += loss_npid_other * self.npid1_w
        else:
            curr_loss = 0
            curr_loss += loss_npid * self.npid0_w
                        
        loss = curr_loss

        return loss, [loss_npid, loss_npid_other, loss_cmc_p2i, loss_cmc_i2p]

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "loss_type": self.loss_type,
        }
        return pprint.pformat(repr_dict, indent=2)
