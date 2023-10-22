# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import Dataset
import logging
import os

import numpy as np
from pathlib import Path

from datasets.transforms import database_sampler, data_augmentor, data_processor

from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
    
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

from torch.utils.data import Dataset
import torch

#np.random.seed(1024)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DepthContrastDataset(Dataset):
    """Base Self Supervised Learning Dataset Class."""

    def __init__(self, cfg, linear_probe = False, mode='train', logger=None):
        self.mode = mode
        self.linear_probe = linear_probe
        self.pretraining = self.mode == 'train' and not self.linear_probe
        self.logger = logger
        self.cfg = cfg
        #self.batchsize_per_replica = cfg["BATCHSIZE_PER_REPLICA"]
        #self.dataset_names = cfg["DATASET_NAMES"] #TODO: is this needed?
        self.root_path = (Path(__file__) / '../..').resolve()  # DepthContrast
        self.point_cloud_range = np.array(cfg["POINT_CLOUD_RANGE"], dtype=np.float32)
        self.class_names = cfg["CLASS_NAMES"]
        self.used_num_point_features  = 4

        # if "GT_SAMPLING" in self.cfg:
        #     self.db_sampler = database_sampler.DataBaseSampler(
        #     root_path=self.root_path / cfg["DATA_PATH"],
        #     sampler_cfg=cfg["GT_SAMPLING"],
        #     class_names=self.class_names,
        #     logger=self.logger
        # )
        self.data_augmentor = data_augmentor.DataAugmentor(self.cfg["POINT_TRANSFORMS"]) if self.pretraining else None


        #### Add the voxelizer here
        self.grid_size = None
        self.voxel_size = None
        self.depth_downsample_factor = None 
        if ("Lidar" in cfg) and cfg["VOX"]:
            self.voxel_size = [0.1, 0.1, 0.15] #[0.05, 0.05, 0.1]

            self.MAX_POINTS_PER_VOXEL = 5
            self.MAX_NUMBER_OF_VOXELS = 150000 #80000
            if SPCONV_VER == 1:
                self.voxel_generator = VoxelGenerator(
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range,
                    max_num_points=self.MAX_POINTS_PER_VOXEL,
                    max_voxels=self.MAX_NUMBER_OF_VOXELS
                )
            else:
                self.voxel_generator = VoxelGenerator(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features = 5,
                    max_num_points_per_voxel=self.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=self.MAX_NUMBER_OF_VOXELS
                )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
            self.grid_size = np.round(grid_size).astype(np.int64)

    def toVox(self, points):
        if "Lidar" in self.cfg:
            if SPCONV_VER==1:
                voxel_output = self.voxel_generator.generate(points)
            else:
                voxel_output = self.voxel_generator(torch.from_numpy(points).contiguous())
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                                                  voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

            data_dict = {}
            data_dict['voxels'] = voxels #dim =(num_voxels, max_points_per_voxel=5, coords+feats = 3+1=4), gives xyzi value for each point in each voxel, if num points in voxel < 5, fill in zeros
            data_dict['voxel_coords'] = coordinates #dim= (num voxels, 3), gives z,y,x grid coord of each voxel 
            data_dict['voxel_num_points'] = num_points # dim = num_voxels, gives num points in each voxel
            return data_dict
    
    def prepare_data(self, data_dict):
        
        PLOT= False
        cfg = self.cfg

        # GT sampling
        # if "GT_SAMPLING" in self.cfg:
        #     data_dict = self.db_sampler(data_dict)

        # remove gt_boxes not in class_names  #TODO: approx boxes
        # gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
        # data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
        # data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]

        # remove points and boxes outside range
        data_dict['points'] = data_processor.mask_points_outside_range(data_dict['points'], self.point_cloud_range)
        
        #If we remove boxes outside range, then there could be some points in those boxes
        # which are wihtin range and therefore have not been removed by mask_points_outside_range step.
        # Hence, Don't mask boxes outside range, in the end we will filter boxes with few points.
        # mask = data_processor.mask_boxes_outside_range(data_dict['gt_boxes'], self.point_cloud_range)
        # data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]

        # for cluster_id in np.unique(data_dict['points'][:,-1]):
        #     if cluster_id == -1:
        #         continue
        #     assert cluster_id in data_dict['gt_boxes'][:,-1]


        # data_dict['gt_box_cluster_ids'] = data_dict['gt_box_cluster_ids'][mask]
        # data_dict['gt_names'] = data_dict['gt_names'][mask]

        if PLOT:
            # After gt sampling and cropping
            V.draw_scenes(points=data_dict["points"][:,:4], gt_boxes=data_dict["gt_boxes"][:,:7], color_feature='intensity')

        # # TODO: not needed, remove
        # if 'gt_boxes2d' in data_dict:
        #     data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask][mask]

        if self.pretraining:
            assert len(data_dict['points']) > 0

            #Create different views / augmentation
            data_dict['points_moco'] = np.copy(data_dict["points"])
            data_dict['gt_boxes_moco'] = np.copy(data_dict["gt_boxes"])
            gt_classes_idx = data_dict["gt_boxes"][:,-2].reshape(-1,1)
            gt_cluster_ids = data_dict["gt_boxes"][:,-1].reshape(-1,1)
            
            # transform data_dict points and gt_boxes #TODO: check if augmentor assumes and returns 7 dim gt boxes
            data_dict["points"], data_dict["gt_boxes"] = self.data_augmentor.forward(data_dict["points"], data_dict["gt_boxes"][:,:7], gt_box_cluster_ids=gt_cluster_ids)

            # transform data_dict points_moco and gt_boxes_moco
            data_dict["points_moco"], data_dict["gt_boxes_moco"] = self.data_augmentor.forward(data_dict["points_moco"], data_dict["gt_boxes_moco"][:,:7], gt_box_cluster_ids=gt_cluster_ids)
            
            #reappend the gt class indexes and cluster ids
            data_dict["gt_boxes"] = np.hstack([data_dict["gt_boxes"], gt_classes_idx, gt_cluster_ids])
            data_dict["gt_boxes_moco"] = np.hstack([data_dict["gt_boxes_moco"], gt_classes_idx, gt_cluster_ids])

            for cluster_id in np.unique(data_dict['points'][:,-1]):
                if cluster_id == -1:
                    continue
                assert cluster_id in data_dict['gt_boxes'][:,-1]

            for cluster_id in np.unique(data_dict['points_moco'][:,-1]):
                if cluster_id == -1:
                    continue
                assert cluster_id in data_dict['gt_boxes_moco'][:,-1]
        
        if PLOT:
            # After augmenting both views
            V.draw_scenes(points=data_dict["points"][:,:4], gt_boxes=data_dict["gt_boxes"][:,:7], color_feature='intensity')
            V.draw_scenes(points=data_dict["points_moco"][:,:4], gt_boxes=data_dict["gt_boxes_moco"][:,:7], color_feature='intensity')

        # data processor
        # sample points if pointnet backbone
        if not cfg["VOX"]:
            data_dict['points'] = data_processor.sample_points(data_dict['points'], self.cfg["SAMPLE_NUM_POINTS"] )
            if self.pretraining:
                data_dict['points_moco'] = data_processor.sample_points(data_dict['points_moco'], self.cfg["SAMPLE_NUM_POINTS"])

        # # shuffle points
        if self.mode == 'train':
            data_dict['points'] = data_processor.shuffle_points(data_dict['points'])
            if self.pretraining:
                data_dict['points_moco'] = data_processor.shuffle_points(data_dict['points_moco'])

        # Add class_index to gt boxes and keep note of gtbox id TODO: is this needed?
        # num_gt_boxes = data_dict['gt_boxes'].shape[0]
        # gt_classes = np.ones(num_gt_boxes) #np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
        # #gt_box_ids = np.array([i for i in range(num_gt_boxes)], dtype=np.int32)

        # assert data_dict['gt_boxes'].shape == data_dict['gt_boxes_moco'].shape
        # #append class id as 8th entry in gt boxes
        # data_dict['gt_boxes'] = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
        # #data_dict['gt_boxes_idx'] = gt_box_ids #.reshape(-1, 1).astype(np.float32) #TODO: remove if not needed

        # #append class id as 8th entry in gt boxes
        # data_dict['gt_boxes_moco'] = np.concatenate((data_dict['gt_boxes_moco'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
        #data_dict['gt_boxes_moco_idx'] = gt_box_ids #.reshape(-1, 1).astype(np.float32)

        # for per point fg,bg prediction
        if self.mode == 'train':
            # If augmentor removes a patch with gt box, remove its gt box and label its points as -1
            # mask, box_ids_of_pts = data_processor.mask_boxes_with_few_points(data_dict['points'], data_dict['gt_boxes'])
            # data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            # data_dict['gt_boxes_idx'] = data_dict['gt_boxes_idx'][mask]
            # data_dict['box_ids_of_pts'] = box_ids_of_pts 
            data_dict['points'], data_dict['gt_boxes'] = data_processor.mask_boxes_with_few_points(data_dict['points'], data_dict['gt_boxes'])
            if self.pretraining:
                data_dict['points_moco'], data_dict['gt_boxes_moco'] = data_processor.mask_boxes_with_few_points(data_dict['points_moco'], data_dict['gt_boxes_moco'])
                # mask, box_ids_of_pts_moco = data_processor.mask_boxes_with_few_points(data_dict['points_moco'], data_dict['gt_boxes_moco'])
                # data_dict['gt_boxes_moco'] = data_dict['gt_boxes_moco'][mask]
                # data_dict['gt_boxes_moco_idx'] = data_dict['gt_boxes_moco_idx'][mask]
                # data_dict['box_ids_of_pts_moco'] = box_ids_of_pts_moco
                # get unique labels from pcd_i and pcd_j
            
        # common_obj_idx = list(set(data_dict['gt_boxes_idx']) & set(data_dict['gt_boxes_idx']))
        # assert len(common_obj_idx) > 0
        
        if PLOT:
            # After sampling points and removing empty boxes
            V.draw_scenes(points=data_dict["points"][:,:4], gt_boxes=data_dict["gt_boxes"][:,:7], color_feature='intensity')
            V.draw_scenes(points=data_dict["points_moco"][:,:4], gt_boxes=data_dict["gt_boxes_moco"][:,:7], color_feature='intensity')
        # if vox then transform points to voxels else save points as tensor
        if cfg["VOX"]:
            vox_dict = self.toVox(data_dict["points"]) # xyzil=clusterlabel 
            data_dict["vox"] = vox_dict

            if self.pretraining:
                vox_dict = self.toVox(data_dict["points_moco"])
                data_dict["vox_moco"] = vox_dict
        # else:
        #     #transform to tensor
        #     data_dict['data'] = data_dict.pop('points')
        #     if self.pretraining:
        #         data_dict['data_moco'] = data_dict.pop('points_moco')

        # #TODO: delete unnecessary 
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        # if 'road_plane' in data_dict:
        #     data_dict.pop('road_plane')

        data_dict['gt_boxes_cluster_ids'] = data_dict['gt_boxes'][:,-1]
        data_dict['gt_boxes'] = data_dict['gt_boxes'][:, :8] #xyz,lwh, rz, gt class_index i.e. 1: Vehicle, ...
        if self.pretraining:
            data_dict['gt_boxes_moco_cluster_ids'] = data_dict['gt_boxes_moco'][:,-1]
            data_dict['gt_boxes_moco'] = data_dict['gt_boxes_moco'][:, :8]

        # for cluster_id in np.unique(data_dict['points'][:,-1]):
        #     if cluster_id == -1:
        #         continue
        #     assert cluster_id in data_dict['gt_boxes_cluster_ids']

        # for cluster_id in np.unique(data_dict['points_moco'][:,-1]):
        #     if cluster_id == -1:
        #         continue
        #     assert cluster_id in data_dict['gt_boxes_moco_cluster_ids']
        return data_dict

    def __getitem__(self, idx):
        """
        To support a custom dataset, implement this function to load the raw data (and labels)
        and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    # def get_batchsize_per_replica(self):
    #     # this searches for batchsize_per_replica in self and then in self.dataset
    #     return getattr(self, "batchsize_per_replica", 1)

    # def get_global_batchsize(self):
    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         world_size = torch.distributed.get_world_size()
    #     else:
    #         world_size = 1
    #     return self.get_batchsize_per_replica() * world_size