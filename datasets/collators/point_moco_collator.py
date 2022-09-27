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
    LINEAR_PROBE = batch[0].get('linear_probe', False)

    data_point = [x["data"] for x in batch]
    data_point_aug_matrix = [x["data_aug_matrix"] for x in batch]
    points = torch.stack([data_point[i][0] for i in range(batch_size)]) #(8, 16384, 4)
    points_aug_matrix = torch.stack([data_point_aug_matrix[i] for i in range(batch_size)])#(8, 3, 3)


    if not LINEAR_PROBE:
        data_moco = [x["data_moco"] for x in batch]
        points_moco = torch.stack([data_moco[i][0] for i in range(batch_size)]) #(8, 16384, 4)      

        output_batch = {
            "points": points,
            "points_moco": points_moco
        }
    else:
        gt_boxes_lidar = [x["gt_boxes_lidar"] for x in batch]
        output_batch = {
            "points": points,
            "points_aug_matrix": points_aug_matrix,
            "gt_boxes_lidar": gt_boxes_lidar
        }
       
    return output_batch
