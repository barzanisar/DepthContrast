#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def point_moco_collator(batch):
    batch_size = len(batch)
    
    data_point = [x["data"] for x in batch]
    data_moco = [x["data_moco"] for x in batch]

    data_point_aug_matrix = [x["data_aug_matrix"] for x in batch]
    data_moco_aug_matrix = [x["data_moco_aug_matrix"] for x in batch]        

    gt_boxes_lidar = [x.get("gt_boxes_lidar", None) for x in batch]

    points_moco = torch.stack([data_moco[i][0] for i in range(batch_size)]) #(8, 16384, 4)
    points = torch.stack([data_point[i][0] for i in range(batch_size)]) #(8, 16384, 4)

    points_moco_aug_matrix = torch.stack([data_moco_aug_matrix[i] for i in range(batch_size)]) #(8, 3, 3)
    points_aug_matrix = torch.stack([data_point_aug_matrix[i] for i in range(batch_size)])#(8, 3, 3)

    output_batch = {
        "points": points,
        "points_moco": points_moco,
        "points_aug_matrix": points_aug_matrix,
        "points_moco_aug_matrix": points_moco_aug_matrix
    }
    if gt_boxes_lidar[0] is not None:
        output_batch["gt_boxes_lidar"] = gt_boxes_lidar
       
    return output_batch
