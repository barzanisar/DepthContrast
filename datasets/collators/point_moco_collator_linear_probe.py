#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def point_moco_collator_linear_probe(batch):
    batch_size = len(batch)

    data_point = [x["data"][0] for x in batch]
    points = torch.stack([data_point[i][:,:4] for i in range(batch_size)]) #(8, 16384, 4)
    labels = []
    labels = torch.stack([data_point[i][:,-1].long() for i in range(batch_size)]) #(8, 16384)

    output_batch = {
            "points": points,
            "labels": labels
        }
       
    return output_batch
