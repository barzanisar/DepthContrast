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
        self.root_path = (Path(__file__) / '../..').resolve()  # DepthContrast
        self.point_cloud_range = np.array(cfg["POINT_CLOUD_RANGE"], dtype=np.float32)
        self.class_names = cfg["CLASS_NAMES"]
        self.used_num_point_features  = 4

        self.data_augmentor = data_augmentor.DataAugmentor(self.cfg["POINT_TRANSFORMS"]) if self.pretraining else None


        #### Add the voxelizer here
        self.grid_size = None
        self.voxel_size = None
        self.depth_downsample_factor = None 
        if cfg["VOX"]:
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
        if SPCONV_VER==1:
            voxel_output = self.voxel_generator.generate(points)
        else:
            voxel_output = self.voxel_generator(torch.from_numpy(points).contiguous())
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            try:
                voxels, coordinates, num_points = voxel_output
            except: 
                voxels, coordinates, num_points, pc_voxel_id = voxel_output 

        data_dict = {}
        data_dict['voxels'] = voxels #dim =(num_voxels, max_points_per_voxel=5, coords+feats = 3+1=4), gives xyzi value for each point in each voxel, if num points in voxel < 5, fill in zeros
        data_dict['voxel_coords'] = coordinates #dim= (num voxels, 3), gives z,y,x grid coord of each voxel 
        data_dict['voxel_num_points'] = num_points # dim = num_voxels, gives num points in each voxel
        return data_dict
    
    def prepare_data(self, data_dict):
        
        PLOT= False
        cfg = self.cfg

        # remove points and boxes outside range
        data_dict['points'] = data_processor.mask_points_outside_range(data_dict['points'], self.point_cloud_range)
    

        if PLOT:
            # After gt sampling and cropping
            V.draw_scenes(points=data_dict["points"][:,:4], gt_boxes=data_dict["gt_boxes"][:,:7], color_feature='intensity')

        if self.pretraining:
            assert len(data_dict['points']) > 0

            #Create different views / augmentation
            data_dict['points_moco'] = np.copy(data_dict["points"])
            data_dict['gt_boxes_moco'] = np.copy(data_dict["gt_boxes"])
            gt_classes_idx = data_dict["gt_boxes"][:,-2].reshape(-1,1)
            gt_cluster_ids = data_dict["gt_boxes"][:,-1].reshape(-1,1)
            data_dict['unscaled_lwhz_cluster_id'] = np.hstack([data_dict["gt_boxes"][:,3:6], data_dict["gt_boxes"][:,2].reshape(-1,1), gt_cluster_ids])

            
            # transform data_dict points and gt_boxes #TODO: check if augmentor assumes and returns 7 dim gt boxes
            data_dict["points"], data_dict["gt_boxes"] = self.data_augmentor.forward(data_dict["points"], data_dict["gt_boxes"][:,:7], gt_box_cluster_ids=gt_cluster_ids)

            # transform data_dict points_moco and gt_boxes_moco
            data_dict["points_moco"], data_dict["gt_boxes_moco"] = self.data_augmentor.forward(data_dict["points_moco"], data_dict["gt_boxes_moco"][:,:7], gt_box_cluster_ids=gt_cluster_ids)
            
            #reappend the gt class indexes and cluster ids
            data_dict["gt_boxes"] = np.hstack([data_dict["gt_boxes"], gt_classes_idx, gt_cluster_ids])
            data_dict["gt_boxes_moco"] = np.hstack([data_dict["gt_boxes_moco"], gt_classes_idx, gt_cluster_ids])
            
            # cluster_ids, cnts = np.unique(data_dict['points'][:,-1], return_counts=True)
            # for cluster_id, cnt in zip(cluster_ids, cnts):
            #     if cluster_id == -1:
            #         continue
            #     frame_id = data_dict['frame_id']
            #     assert cluster_id in data_dict['gt_boxes'][:,-1], f'{frame_id}, cluster_label: {cluster_id}, cnts:{cnt}'

            # cluster_ids, cnts = np.unique(data_dict['points_moco'][:,-1], return_counts=True)
            # for cluster_id, cnt in zip(cluster_ids, cnts):
            #     if cluster_id == -1:
            #         continue
            #     frame_id = data_dict['frame_id']
            #     assert cluster_id in data_dict['gt_boxes_moco'][:,-1], f'{frame_id}, cluster_label: {cluster_id}, cnts:{cnt}'
        
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

        # for per point fg,bg prediction
        if self.mode == 'train':
            # If augmentor removes a patch with gt box, remove its gt box and label its points as -1
            data_dict['points'], data_dict['gt_boxes'] = data_processor.mask_boxes_with_few_points(data_dict['points'], data_dict['gt_boxes'])
            if self.pretraining:
                data_dict['points_moco'], data_dict['gt_boxes_moco'] = data_processor.mask_boxes_with_few_points(data_dict['points_moco'], data_dict['gt_boxes_moco'])

        
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

        data_dict['gt_boxes_cluster_ids'] = data_dict['gt_boxes'][:,-1]
        data_dict['gt_boxes'] = data_dict['gt_boxes'][:, :8] #xyz,lwh, rz, gt class_index i.e. 1: Vehicle, ...
        if self.pretraining:
            data_dict['gt_boxes_moco_cluster_ids'] = data_dict['gt_boxes_moco'][:,-1]
            data_dict['gt_boxes_moco'] = data_dict['gt_boxes_moco'][:, :8]
        
        # Assert that points contain fg points
        assert (data_dict["points"][:,-1] > -1).sum() > 0
        if self.pretraining:
            assert (data_dict["points_moco"][:,-1] > -1).sum() > 0

        #get box indices of gt_boxes remaining after augmentation and then keep only those unscaled lwhz
        box_idx=np.where(np.isin(data_dict['unscaled_lwhz_cluster_id'][:,-1], data_dict['gt_boxes_cluster_ids']))[0]
        data_dict['unscaled_lwhz_cluster_id'] = data_dict['unscaled_lwhz_cluster_id'][box_idx]
        
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