# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from functools import partial

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trunks.mlp import MLP

from .spconv_backbone import post_act_block

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

import numpy as np

class UNetV2_concat(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, use_mlp=False, mlp_dim=None):
        super().__init__()

        input_channels = 4
        voxel_size = [0.05, 0.05, 0.1] #[0.1, 0.1, 0.2]
        point_cloud_range = np.array([0., -40., -3., 70.4, 40., 1.], dtype=np.float32) #DENSE dataset

        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size) # x,y,z = [704, 800, 20]
        grid_size = np.round(grid_size).astype(np.int64)
        model_cfg = {'NAME': 'UNetV2', 'RETURN_ENCODED_TENSOR': False}
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0] #z,y,x grid size = [21, 800, 704]

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        
        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16
        
        self.all_feat_names = [
            "conv4",
        ]
    
        if use_mlp:
            self.use_mlp = True
            self.head = MLP(mlp_dim)

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, x, out_feat_keys=None, aug_matrix=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        ### Pre processing
        voxel_features, voxel_num_points = x['voxels'], x['voxel_num_points'] #voxel_features = (num voxels, 5=max points per voxel, 4=xyzi), voxel_num_points = (num voxels, 1)
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False) #(num voxels, 5=max points per voxel, 4=xyzi) -> sum points per voxel (num voxels, 4=xyzi sum)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features) # (num voxels, 1) gives num points per voxel with min clamped to 1, so 0 points will be counted as 1
        points_mean = points_mean / normalizer
        voxel_features = points_mean.contiguous() # voxel_features =(num voxels, 4=xyzi mean of points in that voxel)

        temp = x['voxel_coords'].detach().cpu().numpy() #(num voxels, 4=batchidx,z,y,x grid coord)
        
        batch_size = len(np.unique(temp[:,0]))
        voxel_coords = x['voxel_coords'] #(num voxels, 4=batchidx,z,y,x grid coord)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features.float(),
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)       #spatial_shape: (21, 800, 704) <- (21, 800, 704)
        x_conv2 = self.conv2(x_conv1) #spatial_shape: (11, 400, 352) <- (21, 800, 704)
        x_conv3 = self.conv3(x_conv2) #spatial_shape: (6, 200, 176) <- (11, 400, 352)
        x_conv4 = self.conv4(x_conv3) #spatial_shape: (2, 100, 88) <- (6, 200, 176)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
            #batch_dict['encoded_spconv_tensor'] = out
            #batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [6, 200, 176] <- [2, 100, 88]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [11, 400, 352] <- [6, 200, 176]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [21, 800, 704] <- [11, 400, 352]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [21, 800, 704] <- [21, 800, 704]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)
        
        end_points = {}

        end_points['conv4_features'] = [x_up4.features, x_up3.features, x_up2.features, x_up1.features]
        end_points['indice'] = [x_up4.indices, x_up3.indices, x_up2.indices, x_up1.indices]
        
        out_feats = [None] * len(out_feat_keys)

        for key in out_feat_keys:
            feat = end_points[key+"_features"]

            featlist = []
            for i in range(batch_size):
                tempfeat = []
                for idx in range(len(end_points['indice'])):
                    temp_idx = end_points['indice'][idx][:,0] == i
                    temp_f = end_points['conv4_features'][idx][temp_idx].unsqueeze(0).permute(0, 2, 1).contiguous()
                    tempfeat.append(F.max_pool1d(temp_f, temp_f.shape[-1]).squeeze(-1)) #(1, 64, numvox in one pc for that conv block idx= 10754)-->(1, 64) max pool to get one vector for one pc
                featlist.append(torch.cat(tempfeat, -1)) # featlist.append((1, 128) feature for pc 0)
            feat = torch.cat(featlist, 0) #(8 pcs, 128) -->change to (num big voxels, 128)
            if self.use_mlp:
                feat = self.head(feat)
            out_feats[out_feat_keys.index(key)] = feat ### Just use smlp
        return out_feats
