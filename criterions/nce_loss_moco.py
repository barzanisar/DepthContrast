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
        # keys: (N=num clusters in this gpu's batch, 128)
        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            # similar to shuffling, since for each gpu the number of segments may not be the same
            # we create a aux variable keys_gather of size (1, MAX_SEG_BATCH, 128)
            # add the current seg batch to [0,:CURR_SEG_BATCH, 128] gather them all in
            # [NUM_GPUS,MAX_SEG_BATCH,128] and concatenate only the filled seg batches

            num_clusters_this_gpu = keys.shape[0]
            feature_dim = keys.shape[-1] # 128

            seg_size = torch.from_numpy(np.array([num_clusters_this_gpu])).cuda()
            all_seg_size = concat_all_gather(seg_size)

            keys_gather = torch.ones((1, all_seg_size.max(), feature_dim)).cuda() # (1, max num clusters, 128)
            keys_gather[0, :num_clusters_this_gpu,:] = keys[:,:]

            all_keys = concat_all_gather(keys_gather) # (num gpus, max num clusters, 128)
            gather_keys = None

            for k in range(len(all_seg_size)): #k is the gpu idx
                if gather_keys is None:
                    gather_keys = all_keys[k][:all_seg_size[k],:]
                else:
                    gather_keys = torch.cat((gather_keys, all_keys[k][:all_seg_size[k],:]))


            keys = gather_keys #(num clusters in all gpus, 128)

        batch_size = keys.shape[0] # num clusters in all gpus

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
                                                                        
    def forward(self, output_dict, output_dict_moco):
        
        batch_size = output_dict['batch_size'] #2
        output_0 = output_dict['pretext_head_feats'] #query features (N0= num clusters , 128)
        output_1 = output_dict_moco['pretext_head_feats'] #key_features = moco features (N1, 128)

        if self.cluster:
            box_ids_of_pts_0 = output_dict['box_ids_of_pts'] # (2, 20000)
            box_ids_of_pts_1 = output_dict_moco['box_ids_of_pts']
            
            # Select corresponding object features across views
            mask_0 = []
            mask_1 = []
            for pc_idx in range(batch_size):
                # common_obj_ids = list(set(output_dict['gt_boxes_idx'][pc_idx]) & set(output_dict_moco['gt_boxes_idx'][pc_idx]))
                # mask_0.append(np.in1d(output_dict['gt_boxes_idx'][pc_idx], output_dict_moco['gt_boxes_idx'][pc_idx], assume_unique=True))
                # mask_1.append(np.in1d(output_dict_moco['gt_boxes_idx'][pc_idx], output_dict['gt_boxes_idx'][pc_idx], assume_unique=True))
                cluster_labels_this_pc_0 = np.unique(box_ids_of_pts_0[pc_idx])[1:] # [-1, 3, 4, 8, ...] -> [3, 4, 8, ...]
                cluster_labels_this_pc_1 = np.unique(box_ids_of_pts_1[pc_idx])[1:] # [-1, 2, 5, 8, 6] -> [2, 5, 8, 6]

                mask_0.append(np.in1d(cluster_labels_this_pc_0, cluster_labels_this_pc_1, assume_unique=True))
                mask_1.append(np.in1d(cluster_labels_this_pc_1, cluster_labels_this_pc_0, assume_unique=True))

            output_0 = output_0[np.concatenate(np.array(mask_0))] # (N=num common clusters, C=128)
            output_1 = output_1[np.concatenate(np.array(mask_1))] # (N=num common clusters, C=128)
            
            # if no common obj, return 0 loss
            if output_0.numel() == 0:
                return 0
            
        assert output_0.shape == output_1.shape
        normalized_output1 = nn.functional.normalize(output_0, dim=1, p=2) #query embeddings 
        normalized_output2 = nn.functional.normalize(output_1, dim=1, p=2) #key embeddings

        # positive logits: Nx1 = batch size of positive examples
        l_pos = torch.einsum('nc,nc->n', [normalized_output1, normalized_output2]).unsqueeze(-1)
        
        # negative logits: NxK
        l_neg_s1q2 = torch.einsum('nc,ck->nk', [normalized_output1, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits_s12_s1q2 = torch.cat([l_pos, l_neg_s1q2], dim=1)
        
        # apply temperature
        logits_s12_s1q2 /= self.T

        N = logits_s12_s1q2.shape[0] # num positive pairs
        labels_s12_s1q2 = torch.zeros(
            N, device=logits_s12_s1q2.device, dtype=torch.int64
        ) # because for each pair out of N pairs, zero'th class (out of K=60000 classes) is the true class

        if len(logits_s12_s1q2.shape) > 2:
            loss_s12_s1q2 = self.xe_criterion(torch.squeeze(logits_s12_s1q2), labels_s12_s1q2) #loss between pointnet query and key embedding
        else:
            loss_s12_s1q2 = self.xe_criterion(logits_s12_s1q2, labels_s12_s1q2) # Nx(1+K) logits, N labels
        

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
