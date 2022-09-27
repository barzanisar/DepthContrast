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
    # print("concat_all_gather each gpu's normalized output 2 shape")
    # for each_tensor in tensors_gather:
    #     print(each_tensor.shape)
    # print("concat_all_gather output shape ", output.shape)
    return output

class NCELossMoco(nn.Module):
    """
    Distributed version of the NCE loss. It performs an "all_gather" to gather
    the allocated buffers like memory no a single gpu. For this, Pytorch distributed
    backend is used. If using NCCL, one must ensure that all the buffer are on GPU.
    This class supports training using both NCE and CrossEntropy (InfoNCE).
    """

    def __init__(self, config, cluster):
        super(NCELossMoco, self).__init__()

        self.cluster = cluster

        self.K = int(config["NUM_NEGATIVES"])
        self.dim = int(config["EMBEDDING_DIM"])
        self.T = float(config["TEMPERATURE"])

        self.register_buffer("queue", torch.randn(self.dim, self.K)) # queue of dc (either pointnet or vox) based negative key embeddings
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # cross-entropy loss. Also called InfoNCE
        self.xe_criterion = nn.CrossEntropyLoss()


    @classmethod
    def from_config(cls, config):
        return cls(config)

    @torch.no_grad()
    def _dequeue_and_enqueue_pcd(self, keys):
    # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            tail_size = self.K - ptr
            head_size = batch_size - tail_size
            self.queue[:, ptr:self.K] = keys.T[:, :tail_size]
            self.queue[:, :head_size] = keys.T[:, tail_size:]

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _dequeue_and_enqueue_cluster(self, keys):
        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            # similar to shuffling, since for each gpu the number of segments may not be the same
            # we create a aux variable keys_gather of size (1, MAX_SEG_BATCH, 128)
            # add the current seg batch to [0,:CURR_SEG_BATCH, 128] gather them all in
            # [NUM_GPUS,MAX_SEG_BATCH,128] and concatenate only the filled seg batches
            seg_size = torch.from_numpy(np.array([keys.shape[0]])).cuda()
            all_seg_size = concat_all_gather(seg_size)

            keys_gather = torch.ones((1, all_seg_size.max(), keys.shape[-1])).cuda()
            keys_gather[0, :keys.shape[0],:] = keys[:,:]

            all_keys = concat_all_gather(keys_gather)
            gather_keys = None

            for k in range(len(all_seg_size)):
                if gather_keys is None:
                    gather_keys = all_keys[k][:all_seg_size[k],:]
                else:
                    gather_keys = torch.cat((gather_keys, all_keys[k][:all_seg_size[k],:]))


            keys = gather_keys

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            tail_size = self.K - ptr
            head_size = batch_size - tail_size
            self.queue[:, ptr:self.K] = keys.T[:, :tail_size]
            self.queue[:, :head_size] = keys.T[:, tail_size:]

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
                                                                        
    def forward(self, output_dict):
        
        if self.cluster:
            output_0 = output_dict[0]['seg_feats'] #query features (8, 128)
            output_1 = output_dict[1]['seg_feats'] #key_features = moco features (8, 128)
        else:
            output_0 = output_dict[0]['dc_feats'] #query features (8, 128)
            output_1 = output_dict[1]['dc_feats'] #key_features = moco features (8, 128)
        
        normalized_output1 = nn.functional.normalize(output_0, dim=1, p=2) #query dc embedding or vdc
        normalized_output2 = nn.functional.normalize(output_1, dim=1, p=2) #key dc embedding or vdc

        # positive logits: Nx1 = batch size of positive examples (or matched num voxels)x 1
        l_pos_s12 = torch.einsum('nc,nc->n', [normalized_output1, normalized_output2]).unsqueeze(-1) #(v_i_1).transpose() * v_i_2 => dim (n = matched num voxels)
        
        # negative logits: NxK
        l_neg_s1q2 = torch.einsum('nc,ck->nk', [normalized_output1, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits_s12_s1q2 = torch.cat([l_pos_s12, l_neg_s1q2], dim=1)
        
        # apply temperature
        logits_s12_s1q2 /= self.T

        labels_s12_s1q2 = torch.zeros(
            logits_s12_s1q2.shape[0], device=logits_s12_s1q2.device, dtype=torch.int64
        ) # because zero'th class is the true class

        loss_s12_s1q2 = self.xe_criterion(torch.squeeze(logits_s12_s1q2), labels_s12_s1q2) #loss between pointnet query and key embedding

        if self.cluster:
            self._dequeue_and_enqueue_cluster(normalized_output2)
        else:
            self._dequeue_and_enqueue_pcd(normalized_output2)

            

        return loss_s12_s1q2

    def __repr__(self):
        repr_dict = {
            "name": self._get_name()
        }
        return pprint.pformat(repr_dict, indent=2)
