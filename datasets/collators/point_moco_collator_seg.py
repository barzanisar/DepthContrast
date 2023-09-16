#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from utils.pcd_preprocess_old import *

def point_moco_collator_seg(batch):
    batch_size = len(batch)

    data_point = [x["data"][0] for x in batch]
    data_moco = [x["data_moco"][0] for x in batch]
    pi_cluster = []
    pj_cluster = []

    for i in range(batch_size):
        points = np.asarray(data_point[i])
        points_moco = np.asarray(data_moco[i])
        cluster_pi = points[:,-1] # 16384
        cluster_pj = points_moco[:,-1] # 16384

        # Set cluster id not present in both point clouds to -1
        cluster_pi, cluster_pj = overlap_clusters(cluster_pi, cluster_pj, min_cluster_point=5)
        pi_cluster.append(cluster_pi) #(8, 16384)
        pj_cluster.append(cluster_pj) #(8, 16384)


    points = torch.stack([data_point[i][:,:4] for i in range(batch_size)]) #(8, 16384, 4)
    points_moco = torch.stack([data_moco[i][:,:4] for i in range(batch_size)]) #(8, 16384, 4)      

    output_batch = {
        "points": points,
        "points_moco": points_moco,
        "points_cluster": np.asarray(pi_cluster),
        "points_moco_cluster": np.asarray(pj_cluster)
    }

    return output_batch
