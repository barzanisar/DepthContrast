# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import numpy as np

from models.trunks.mlp import MLP

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) #trunks
ROOT_DIR = os.path.dirname(ROOT_DIR) #models
ROOT_DIR = os.path.dirname(ROOT_DIR) #DepthContrast
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'OpenPCDet', "pcdet"))

from ops.pointnet2.pointnet2_batch import pointnet2_modules
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

def point_to_voxel_func(device = torch.device("cpu:0")):
    VOXEL_SIZE = [5.0, 5.0, 6.0]
    ### Waymo lidar range
    #POINT_RANGE = np.array([  0. , -75. ,  -3. ,  75.0,  75. ,   3. ], dtype=np.float32)
    POINT_RANGE = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32) ### KITTI and DENSE
    MAX_POINTS_PER_VOXEL = 2000
    MAX_NUMBER_OF_VOXELS = 400
    NUM_POINT_FEATURES= 3+128
    if SPCONV_VER == 1:
        voxel_generator = VoxelGenerator(
            voxel_size=VOXEL_SIZE,
            point_cloud_range=POINT_RANGE,
            max_num_points=MAX_POINTS_PER_VOXEL,
            max_voxels=MAX_NUMBER_OF_VOXELS,
            device = device
        )
    else:
        voxel_generator = VoxelGenerator(
            vsize_xyz=VOXEL_SIZE,
            coors_range_xyz=POINT_RANGE,
            num_point_features=NUM_POINT_FEATURES,
            max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
            max_num_voxels=MAX_NUMBER_OF_VOXELS,
            device = device
        )

    return voxel_generator

class PointNet2MSG(nn.Module):
    def __init__(self, use_mlp=False, mlp_dim=None, linear_probe = False):
        super().__init__()
        self.linear_probe = linear_probe
        input_channels = 4
        
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3 #  C

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        SA_CONFIG = {'NPOINTS': [4096, 1024, 256, 64],
                     'RADIUS': [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]],
                     'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32]],
                     'MLPS': [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]], [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]}
        # (MSG) Multiscale SA(K=4096, r=[0.1, 0.5], PointNets (for r = 0.1 and r = 0.5 respectively) = [[16, 16, 32], [32, 32, 64]])

        FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]] #[last FP= [1+256, 128, 128], second last fp = [96+512, 256, 256], fp = [256+512, 512, 512], fp=[512 + 1024, 512, 512]]
        # The correct order of FP1 = [Cin = Cout of SA second last level + Cout of SA last = 512 + 1024, 512, 512]
        # -> FP2 = [Cin = Cout of SA 3rd last level + Cout of FP1 = 256+512, 512, 512]
        # -> FP3 = [Cin = Cout of SA 4th last level + Cout of FP2 = 96+512, 256, 256]
        # -> FP4 = [Cin = Cout of SA 1st level + Cout of FP3 = 1+256, 128, 128]
        for k in range(SA_CONFIG["NPOINTS"].__len__()):
            mlps = SA_CONFIG["MLPS"][k].copy() #SA1 = sample 4096 centroids, group points that fall within radius of 0.1 for each centroid, group for r= 0.5, mlps = [for r=0.1->[16, 16, 32], for r=0.5 -> [32, 32, 64]]
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=SA_CONFIG["NPOINTS"][k],
                    radii=SA_CONFIG["RADIUS"][k],
                    nsamples=SA_CONFIG["NSAMPLE"][k],
                    mlps=mlps,
                    use_xyz=True,
                )
            )
            # mlps for SA1 becomes [Cin = xyz+(Cout of previous layer or 1 (intensity)) = 3+1 = 4, 16, 16, 32], [3+1, 32, 32, 64]
            # Cout of SA1 = 32 (feature vector for r = 0.1) + 64 (feature vector for r = 0.5) = 96

            #mlps for SA2 becomes [Cin=xyz+96 = 99, 64,64,128], [64, 96, 128], Cout = 128+128 = 256
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k]
                )
            )

        self.num_point_features = FP_MLPS[0][-1] # 128
        
        self.all_feat_names = [
            "fp2",
        ]
        self.use_mlp = False
        if use_mlp:
            self.use_mlp = True
            self.head = MLP(mlp_dim) # projection head

        self.dout=nn.Dropout(p=0.4)
            
    def break_up_pc(self, pc):
        #batch_idx = pc[:, 0]
        xyz = pc[:, :, 0:3].contiguous()
        features = (pc[:, :, 3:].contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, out_feat_keys=None, aug_matrix=None, cluster_id=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = pointcloud.shape[0]
        points = pointcloud
        xyz, features = self.break_up_pc(points) # xyz = (8, 16384, 3), features (intensity) = (8, 1, 16384)

        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            assert l_features[i].is_contiguous()
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            assert li_features.is_contiguous()
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            assert l_features[i - 1].is_contiguous()
            assert l_features[i].is_contiguous()
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)
            assert l_features[i - 1].is_contiguous()

        point_features = l_features[0] #(B=8, 128, num points = 16384)
        # out_dict = {'dc_feats': None, \
        #     'vdc_feats': None, 'vdc_voxel_bzyx': None, \
        #     'linear_probe_feats': None, 'linear_probe_xyz': None}      
        out_dict = {}


        if self.linear_probe:
            xyz = xyz @ aug_matrix.inverse()
            feat = point_features.view(batch_size, -1, point_features.shape[-1]).permute(0, 2, 1).contiguous() #(8, 128, npoints=16384) -> (8, 16384, 128) 
            #out_dict['point_features'] = feat
            if self.use_mlp:
                feat = self.head(feat)
            out_dict['linear_probe_feats'] = feat
            out_dict['linear_probe_xyz'] = xyz
        else:
            if 'seg_feats' in out_feat_keys:
                batch_seg_feats = []
                for pc_idx in range(batch_size):
                    pc_feats = point_features[pc_idx] #(128 x 16384)
                    cluster_labels_this_pc = cluster_id[pc_idx] # (16384,)
                    for segment_lbl in np.unique(cluster_labels_this_pc):
                        if segment_lbl == -1:
                            continue

                        seg_feats = pc_feats[:,cluster_labels_this_pc == segment_lbl] #(128, npoints in this seg)
                        seg_feats = self.dout(seg_feats)
                        seg_feats = seg_feats.unsqueeze(0) #1,128, npoints in this seg
                        npoints = seg_feats.shape[-1]
                        seg_max_feat = F.max_pool1d(seg_feats, npoints).squeeze(-1)
                        batch_seg_feats.append(seg_max_feat)
                
                all_seg_feats = torch.vstack(batch_seg_feats) # num clusters x 128
                out_dict['seg_feats'] = all_seg_feats

            if 'dc_feats' in out_feat_keys:
                nump = point_features.shape[-1] # 16384
                # get one feature vector of dim 128 for the entire point cloud
                feat = torch.squeeze(F.max_pool1d(point_features, nump)) # (8,128)
                if self.use_mlp:
                    feat = self.head(feat) #linear probe out (8=batch, 16384=npoints, 5=nclasses), #Dc out (8, 128)
                out_dict['dc_feats'] = feat


        # out_feats[0] = (1181, 128), vox_coords = (1181, 4=bzyx), point_coords = (8, 16384, 3=xyz)
        return out_dict        