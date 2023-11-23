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
from third_party.OpenPCDet.pcdet.ops.iou3d_nms import iou3d_nms_utils

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
        self.embedding_dim = int(config["EMBEDDING_DIM"])
        self.dim = int(config["EMBEDDING_DIM"])
        self.iou_threshold = None
        self.rmse_threshold = None
        self.iou_weight = None
        self.shape_descs_dim = None
        self.iou_guidance = False
        self.neg_queue_filled = False
        if self.cluster:
            self.iou_threshold = config.get("IOU_THRESHOLD", None)
            if self.iou_threshold is not None:
                self.iou_threshold = 1 - self.iou_threshold
            self.shape_descs_dim = config.get("SHAPE_DESCRIPTORS_DIM", None)
            self.iou_weight =  config.get("IOU_WEIGHT", None)
            self.rmse_threshold = config.get("RMSE_THRESHOLD", None)
            self.iou_guidance = self.iou_weight is not None or self.iou_threshold is not None

        if self.shape_descs_dim is not None:
            self.dim += self.shape_descs_dim

        if self.iou_guidance:
            self.dim += 4 #lwhz

        self.T = float(config["TEMPERATURE"])

        self.register_buffer("queue", torch.randn(self.dim, self.K)) # queue of dc (either pointnet or vox) based negative key embeddings
        self.queue[:self.embedding_dim, :] = nn.functional.normalize(self.queue[:self.embedding_dim, :], dim=0)
        
        if self.shape_descs_dim is not None:
            self.queue[self.embedding_dim:self.embedding_dim+self.shape_descs_dim, :] = -1 * torch.ones((self.shape_descs_dim,  self.K))
        if self.iou_guidance:
            if self.shape_descs_dim is not None:
                self.queue[self.embedding_dim+self.shape_descs_dim:, :] = -1 * torch.ones((4,  self.K))
            else:
                self.queue[self.embedding_dim:, :] = -1 * torch.ones((4,  self.K))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # cross-entropy loss. Also called InfoNCE
        self.xe_criterion = nn.CrossEntropyLoss()
        self.loss_weight = config['LOSS_WEIGHT']


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
        # keys: (N=num clusters in this gpu's batch, 128+3)
        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            # similar to shuffling, since for each gpu the number of segments may not be the same
            # we create a aux variable keys_gather of size (1, MAX_SEG_BATCH, 128)
            # add the current seg batch to [0,:CURR_SEG_BATCH, 128] gather them all in
            # [NUM_GPUS,MAX_SEG_BATCH,128] and concatenate only the filled seg batches

            num_clusters_this_gpu = keys.shape[0] #i.e. N
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
        
        if ptr + batch_size >= self.K:
            self.neg_queue_filled = True

        if ptr + batch_size >= self.K:
            self.neg_queue_filled = True
        
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
                                                                        
    def forward(self, output_dict, output_dict_moco):
        
        # batch_size = output_dict['batch_size'] #2
        common_unscaled_lwhz = output_dict['common_unscaled_lwhz']
        output_0 = output_dict['pretext_head_feats'] #query features (N=num common clusters , 128)
        output_1 = output_dict_moco['pretext_head_feats'] #key_features = moco features (N=num common clusters, 128)
        feature_dim = output_0.shape[1]
        N_pos = output_0.shape[0]
            
        # assert output_0.shape == output_1.shape
        normalized_output1 = nn.functional.normalize(output_0, dim=1, p=2) #query embeddings 
        normalized_output2 = nn.functional.normalize(output_1, dim=1, p=2) #key embeddings

        #TODO: class balanced contrastive loss
        if self.iou_guidance:
            # batch_size = output_dict['gt_boxes'].shape[0]
            #dxdydz = torch.cat([output_dict['gt_boxes'][k, output_dict['common_cluster_gtbox_idx'][k], 3:6]  for k in range(batch_size)]) #(N,3)
            #dxdydz = torch.cat([output_dict_moco['gt_boxes'][k, output_dict_moco['common_cluster_gtbox_idx'][k],3:6]  for k in range(batch_size)])
            # Define l= max(dx,dy), w = min(dx,dy)
            l = torch.max(common_unscaled_lwhz[:,:2], dim=-1)[0].view(-1,1)
            w = torch.min(common_unscaled_lwhz[:,:2], dim=-1)[0].view(-1,1)
            h = common_unscaled_lwhz[:, 2].view(-1,1)
            z =  common_unscaled_lwhz[:, 3].view(-1,1)

            if self.neg_queue_filled: # last element of queue has been filled with the dims
                l_neg = self.queue[-4,:].view(1, -1)
                w_neg = self.queue[-3,:].view(1, -1)
                h_neg = self.queue[-2,:].view(1, -1)
                z_neg = self.queue[-1,:].view(1, -1)

                # height overlap
                boxes_pos_height_max = (z + h / 2).view(-1, 1)
                boxes_pos_height_min = (z - h / 2).view(-1, 1)
                boxes_neg_height_max = (z_neg + h_neg / 2).view(1, -1)
                boxes_neg_height_min = (z_neg - h_neg / 2).view(1, -1)
                max_of_min = torch.max(boxes_pos_height_min, boxes_neg_height_min)
                min_of_max = torch.min(boxes_pos_height_max, boxes_neg_height_max)
                overlaps_h = torch.clamp(min_of_max - max_of_min, min=0) # torch.min(h, h_neg)

                vol = (l*w*h).view(-1, 1) # Nx1
                vol_neg = (l_neg * w_neg * h_neg).view(1, -1) # 1xK
                overlap_vol = torch.min(l, l_neg) *  torch.min(w, w_neg) * overlaps_h # NxK
                iou3d = overlap_vol / torch.clamp(vol + vol_neg - overlap_vol, min=1e-6) # NxK
                iou_based_neg_w =  (1-iou3d) # NxK #if high iou, similar sizes -> low weight or threshold

        if self.shape_descs_dim is not None:
            common_shape_descs_mask = output_dict['shape_cluster_ids_is_common_mask_batch']
            shape_descs = output_dict['shape_descs'][common_shape_descs_mask] #N x 604
            rmse = torch.zeros((N_pos, self.K), device=shape_descs.device, dtype=shape_descs.dtype)
            if self.neg_queue_filled:
                shape_descs_neg = self.queue[self.embedding_dim:self.embedding_dim+self.shape_descs_dim, :].T
                for i in range(N_pos):
                    rmse_pos_i_neg_all = torch.norm(shape_descs[i] - shape_descs_neg, dim=1) #norm((Nneg, 604)=(1, 604) - (Nneg, 604)) -> Nneg
                    rmse[i] = rmse_pos_i_neg_all
                # map rmse between 0 to 1
                rmse_based_neg_w = rmse / torch.max(rmse, dim = 1, keepdim=True)[0] #(Npos, Nneg) / (Npos, 1) = (Npos, Nneg)


        # positive logits: Nx1 = batch size of positive examples
        l_pos = torch.einsum('nc,nc->n', [normalized_output1, normalized_output2]).unsqueeze(-1)
        
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [normalized_output1, self.queue.clone().detach()[:feature_dim,:]])

        # # Another option remove neg examples from denominator if iou3d of positive and neg samples > 0.6
        # iou_based_neg_w[iou_based_neg_w<0.4] = 0
        neg_w = 0
        if self.neg_queue_filled:
            if self.shape_descs_dim is not None:
                if self.iou_weight is not None:
                    neg_w = rmse_based_neg_w * (1-self.iou_weight)
                else:
                    neg_w = rmse_based_neg_w
            if self.iou_weight is not None:
                neg_w += (self.iou_weight * iou_based_neg_w)
            if self.shape_descs_dim is not None or self.iou_weight is not None:
                l_neg = l_neg * neg_w

            if self.shape_descs_dim is not None and self.rmse_threshold is not None:
                l_neg=l_neg.masked_fill(rmse_based_neg_w<self.rmse_threshold, -1e9)
            if self.iou_threshold is not None:
                # l_neg[iou_based_neg_w<self.iou_threshold] = float('-inf')
                l_neg=l_neg.masked_fill(iou_based_neg_w<self.iou_threshold, -1e9)
        # logits: Nx(1+K) i.e. N examples or clusters and K+1 class scores
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # apply temperature
        logits /= self.T

        #N = logits.shape[0] # num common clusters in this batch of pcs for segContrast and batch size for DepthContrast
        labels = torch.zeros(
            N_pos, device=logits.device, dtype=torch.int64
        ) # because for each pair out of N pairs, zero'th class (out of K=60000 classes) is the true class

        if len(logits.shape) > 2:
            loss = self.loss_weight * self.xe_criterion(torch.squeeze(logits), labels) #loss between pointnet query and key embedding
        else:
            loss = self.loss_weight * self.xe_criterion(logits, labels) # Nx(1+K) logits, N labels
        

        if self.cluster:
            keys = normalized_output2
            if self.shape_descs_dim is not None:
                keys = torch.cat([keys, shape_descs], dim=1)
            if self.iou_guidance:
                keys = torch.cat([keys, l,w,h,z], dim=1)
            self._dequeue_and_enqueue_cluster(keys)  
        else:
            self._dequeue_and_enqueue_pcd(normalized_output2)

        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name()
        }
        return pprint.pformat(repr_dict, indent=2)
