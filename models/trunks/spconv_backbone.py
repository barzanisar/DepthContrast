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
#from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V

try:
    try:
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            SPCONV_VER = 1
        except:
            from spconv.utils import VoxelGenerator
            SPCONV_VER = 1
    except:
        #from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
        #from spconv.utils import Point2VoxelGPU3d as VoxelGenerator
        from spconv.pytorch.utils import PointToVoxel as VoxelGenerator
        SPCONV_VER = 2
except:
    pass

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out
def point_to_voxel_func(device = torch.device("cpu:0"), voxel_size =  [5.0, 5.0, 6.0]):
    ### Waymo lidar range
    #POINT_RANGE = np.array([  0. , -75. ,  -3. ,  75.0,  75. ,   3. ], dtype=np.float32)
    POINT_RANGE = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32) ### KITTI and DENSE
    MAX_POINTS_PER_VOXEL = 500
    MAX_NUMBER_OF_VOXELS = 400
    NUM_POINT_FEATURES= 3+128
    if SPCONV_VER == 1:
        voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=POINT_RANGE,
            max_num_points=MAX_POINTS_PER_VOXEL,
            max_voxels=MAX_NUMBER_OF_VOXELS,
            device = device
        )
    else:
        voxel_generator = VoxelGenerator(
            vsize_xyz=voxel_size,
            coors_range_xyz=POINT_RANGE,
            num_point_features=NUM_POINT_FEATURES,
            max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
            max_num_voxels=MAX_NUMBER_OF_VOXELS,
            device = device
        )

    return voxel_generator

import numpy as np
class VoxelBackBone8x(nn.Module):
    def __init__(self, use_mlp=False, mlp_dim=None, linear_probe=False):
        super().__init__()
        self.linear_probe = linear_probe
        input_channels = 4
        voxel_size = [0.05, 0.05, 0.1] #[0.1, 0.1, 0.2]
        point_cloud_range = np.array([0., -40., -3., 70.4, 40., 1.], dtype=np.float32) #DENSE dataset

        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size) # x,y,z = [1600, 1408, 40]
        grid_size = np.round(grid_size).astype(np.int64)
        self.model_cfg = {'NAME': 'VoxelBackBone8x'}
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.sparse_shape = grid_size[::-1] + [1, 0, 0] #[41, 1408, 1600] = z,y,x

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
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        if use_mlp:
            self.use_mlp = True
            self.head = MLP(mlp_dim)

    def forward(self,  x, out_feat_keys=None, aug_matrix=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
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
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        # voxel_centers_x_and_xconv1 = get_voxel_centers(voxel_coords[:,1:], downsample_times=1, voxel_size=self.voxel_size, point_cloud_range = self.point_cloud_range)
        # voxel_centers_xconv2 = get_voxel_centers(x_conv2.indices[:,1:], downsample_times=2, voxel_size=self.voxel_size, point_cloud_range = self.point_cloud_range)
        # voxel_centers_xconv3 = get_voxel_centers(x_conv3.indices[:,1:], downsample_times=4, voxel_size=self.voxel_size, point_cloud_range = self.point_cloud_range)
        # voxel_centers_xconv4 = get_voxel_centers(x_conv4.indices[:,1:], downsample_times=8, voxel_size=self.voxel_size, point_cloud_range = self.point_cloud_range)

        # Dims:
        # dc_feats: (8, 128)
        # vdc_feats: (8 * num_big_voxels in a pc, 128)
        # vdc_voxel_bzyx : (8 * num_big_voxels in a pc, 4) = batch, z, y, x ??
        # linear_probe_feats: list of 8, each element's dim (num out voxels in a pc, 128) same as out.features but in diff view 
        # linear_probe_xyz: list of 8, each element's dim (num out voxels in a pc, 3)
        out_dict = {'dc_feats': None, \
            'vdc_feats': None, 'vdc_voxel_bzyx': None, \
            'linear_probe_feats': None, 'linear_probe_xyz': None, \
            'linear_probe_voxel_size': None}       
        
        dc_feat_dict = {}

        dc_feat_dict['conv4_features'] = [out.features] #[x_conv4.features, x_conv3.features]
        dc_feat_dict['indice'] = [out.indices] #[x_conv4.indices, x_conv3.indices]
        
        if self.linear_probe:
            out_xyz = get_voxel_centers(out.indices[:,1:], downsample_times=torch.tensor([8,8,16], device=out.indices.device), voxel_size=self.voxel_size, point_cloud_range = self.point_cloud_range)
            batch_voxel_features_list = []
            out_xyz_list = []
            if self.use_mlp:
                out_features = self.head(out.features)

            for i in range(batch_size):
                voxel_idx_in_pc_i = out.indices[:,0] == i # (true false mask of len num vox total in all pcs)
                features_in_pc_i = out_features[voxel_idx_in_pc_i] #(num vox in pc i, 128)
                xyz_pc_i = out_xyz[voxel_idx_in_pc_i] #(num vox in pc i, 3)
                xyz_pc_i = xyz_pc_i @ aug_matrix[i] # undo transformation
                out_xyz_list.append(xyz_pc_i)
                batch_voxel_features_list.append(features_in_pc_i)

            #batch_voxel_feats = torch.stack(batch_voxel_features_list) #(total num voxels=1181, 128) = ( 8, numvoxels in each pc, 128)
            #out_xyz_list = torch.stack(out_xyz_list)
            
            out_dict['linear_probe_feats'] = batch_voxel_features_list
            out_dict['linear_probe_xyz'] = out_xyz_list
            out_dict['linear_probe_voxel_size'] = np.array(self.voxel_size) * np.array([8,8,16])
        else:
            if 'vdc_feats' in out_feat_keys:
                out_xyz = get_voxel_centers(out.indices[:,1:], downsample_times=torch.tensor([8,8,16], device=out.indices.device), voxel_size=self.voxel_size, point_cloud_range = self.point_cloud_range)
                voxel_generator = point_to_voxel_func(device=out.indices.device)
                batch_voxel_features_list = []
                vox_coords = []

                for i in range(batch_size):
                    voxel_idx_in_pc_i = out.indices[:,0] == i # (true false mask of len num vox total in all pcs)
                    features_in_pc_i = out.features[voxel_idx_in_pc_i] #(num vox in pc i, 128)
                    xyz_pc_i = out_xyz[voxel_idx_in_pc_i] #(num vox in pc i, 3)
                    xyz_pc_i = xyz_pc_i @ aug_matrix[i] # undo transformation
                    xyz_features = torch.cat([xyz_pc_i, features_in_pc_i], 1) # (num voxels in pc i, 131=3+128)

                    # voxelize
                    voxel_features, coordinates, voxel_num_points = voxel_generator(xyz_features)
                    
                    # try:
                    #     #V.draw_scenes(points=xyz_pc_i)
                    #     voxel_features, coordinates, voxel_num_points = voxel_generator(xyz_features)
                    # except:
                    #     b=1
                    #     V.draw_scenes(points=xyz_pc_i)
                    #     V.draw_scenes(points=original_vox_features[:,:3])
                    #     dist = np.linalg.norm(xyz_pc_i.cpu().numpy()[:, 0:3], axis=1)
                    #     min_value = np.min(dist)
                    #     max_value = np.max(dist)
                    #     try:
                    #         voxel_generator = point_to_voxel_func(device=out.indices.device, voxel_size =  [2.0, 2.0, 6.0])
                    #         voxel_features, coordinates, voxel_num_points = voxel_generator(xyz_features)
                    #     except:
                    #         try:
                    #             voxel_generator = point_to_voxel_func(device=out.indices.device, voxel_size =  [1.0, 1.0, 6.0])
                    #             voxel_features, coordinates, voxel_num_points = voxel_generator(xyz_features)
                    #         except:
                    #             try:
                    #                 voxel_generator = point_to_voxel_func(device=out.indices.device, voxel_size =  [50.0, 50.0, 6.0])
                    #                 voxel_features, coordinates, voxel_num_points = voxel_generator(xyz_features)
                    #             except:
                    #                 try:
                    #                     voxel_generator = point_to_voxel_func(device=out.indices.device, voxel_size =  [0.5, 0.5, 6.0])
                    #                     voxel_features, coordinates, voxel_num_points = voxel_generator(xyz_features)
                    #                 except:
                    #                     voxel_generator = point_to_voxel_func(device=out.indices.device, voxel_size =  [10.0, 10.0, 6.0])
                    #                     voxel_features, coordinates, voxel_num_points = voxel_generator(xyz_features)
                    #     b=1

                    vox_coords.append(np.pad(coordinates.cpu(), ((0, 0), (1, 0)), mode='constant', constant_values=i))
                    # Take max pool of features inside each big voxel
                    voxel_features = F.max_pool1d(voxel_features.permute(0, 2, 1).contiguous(), voxel_features.shape[1]).squeeze(-1) #(num big voxels, 128)
                    batch_voxel_features_list.append(voxel_features[:,3:])

                batch_voxel_feats = torch.cat(batch_voxel_features_list, 0) #(total num voxels=1181, 128) = ( 8 * numvoxels in each pc, 128)
                vox_coords = np.concatenate(vox_coords, axis=0) #(total num vox = 1181, 4)  = [batch_idx, z_idx,y_idx,x_idx]
                
                # Projection head (tot num voxels, 128) -> (tot num voxels, 128)
                if self.use_mlp:
                    batch_voxel_feats = self.head(batch_voxel_feats)
                out_dict['vdc_feats'] = batch_voxel_feats
                out_dict['vdc_voxel_bzyx'] = vox_coords

            if 'dc_feats' in out_feat_keys:
                featlist = [] # list of (1, 128) features for all 8 pcs = [(1, 128) for pc 0, (1, 128) for pc 1, ...]
                for i in range(batch_size):
                    tempfeat = [] # list of max pooled features from diff levels for pc i = [conv4 = (1, 64), conv3 = (1, 64)] OR [out = (1, 128)]
                    for idx in range(len(dc_feat_dict['indice'])): # for each feature level
                        temp_idx = dc_feat_dict['indice'][idx][:,0] == i # dim = (num voxels in pc i), gives idx of all voxels in pc i
                        temp_f = dc_feat_dict['conv4_features'][idx][temp_idx].unsqueeze(0).permute(0, 2, 1).contiguous() # end_points['conv4_features'][idx][temp_idx] has dim (num_voxels_pc_i, 128) -> (1, 128, num voxels pc i)
                        try:
                            tempfeat.append(F.max_pool1d(temp_f, temp_f.shape[-1]).squeeze(-1)) #(1, 64, numvox in pc i)-->(1, 64) max pool to get one vector for one pc
                        except:
                            print(temp_f)
                            print("Shape: ", temp_f.shape)
                            print("Min: ", temp_f.min())
                            print("Max: ", temp_f.max())
                    featlist.append(torch.cat(tempfeat, -1)) # featlist.append((1, 128) feature for pc 0)
                feat = torch.cat(featlist, 0) #(8 pcs, 128)
                if self.use_mlp:
                    feat = self.head(feat)
                out_dict['dc_feats'] = feat
        
                #out_feats = [feat]
        
        
        # vox_coords = None #For Voxelized depth contrast 
        # point_coords = None #For linear probe

        return out_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict
