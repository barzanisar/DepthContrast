#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# try:
#     from models.trunks.pointnet import PointNet
# except:
#     PointNet = None

# try:
#     from models.trunks.spconv.models.res16unet import Res16UNet34
# except:
#     Res16UNet34 = None
    
# try:
#     from models.trunks.pointnet2_backbone_new import PointNet2MSG
#     from models.trunks.spconv_unet import UNetV2_concat as UNetV2
#     from models.trunks.spconv_backbone import VoxelBackBone8x
# except:
#     PointNet2MSG = None
#     UNetV2 = None

from third_party.OpenPCDet.pcdet.models.detectors import build_detector

def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

# TRUNKS = {
#     "pointnet": PointNet,
#     "unet": Res16UNet34,
#     "PointNet2MSG": PointNet2MSG,
#     "UNetV2": UNetV2,
#     "VoxelBackBone8x": VoxelBackBone8x
# }


__all__ = ["TRUNKS"]
