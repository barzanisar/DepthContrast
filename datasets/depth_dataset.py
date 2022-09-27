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
import pickle
import copy
import sys
import time
from datasets.transforms.augment3d import get_transform3d
from datasets.transforms.weather_transforms import *

from pcdet.utils import calibration_kitti
from lib.LiDAR_snow_sim.tools.snowfall.sampling import snowfall_rate_to_rainfall_rate
from utils.pcd_preprocess import *


from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V

try:
    ### Default uses minkowski engine
    from datasets.transforms.voxelizer import Voxelizer
    from datasets.transforms import transforms
except:
    pass
    
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

### Waymo lidar range
WAYMO_POINT_RANGE = np.array([  0. , -75. ,  -3. ,  75.0,  75. ,   3. ], dtype=np.float32)
# KITTI and DENSE range
DENSE_POINT_RANGE = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)
np.random.seed(1024)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DepthContrastDataset(Dataset):
    """Base Self Supervised Learning Dataset Class."""

    def __init__(self, cfg, linear_probe = False, mode='train', logger=None):
        self.mode = mode
        self.linear_probe = linear_probe
        self.logger = logger
        self.cfg = cfg
        self.batchsize_per_replica = cfg["BATCHSIZE_PER_REPLICA"]
        self.dataset_names = cfg["DATASET_NAMES"]
        self.root_path = (Path(__file__) / '../..').resolve()  # DepthContrast
        if "WaymoDataset" in self.dataset_names:
            self.point_cloud_range = WAYMO_POINT_RANGE
            self.class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
        elif "DenseDataset" or "DenseKittiDataset" in self.dataset_names:
            self.point_cloud_range = DENSE_POINT_RANGE
            self.class_names = ['Car', 'Pedestrian', 'Cyclist'] #, 'LargeVehicle' is now grouped under PassengerCar

        #### Add the voxelizer here
        if ("Lidar" in cfg) and cfg["VOX"]:
            self.VOXEL_SIZE = [0.05, 0.05, 0.1] #[0.1, 0.1, 0.2]

            self.MAX_POINTS_PER_VOXEL = 5
            self.MAX_NUMBER_OF_VOXELS = 16000
            if SPCONV_VER == 1:
                self.voxel_generator = VoxelGenerator(
                    voxel_size=self.VOXEL_SIZE,
                    point_cloud_range=self.point_cloud_range,
                    max_num_points=self.MAX_POINTS_PER_VOXEL,
                    max_voxels=self.MAX_NUMBER_OF_VOXELS
                )
            else:
                self.voxel_generator = VoxelGenerator(
                    vsize_xyz=self.VOXEL_SIZE,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features = 4,
                    max_num_points_per_voxel=self.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=self.MAX_NUMBER_OF_VOXELS
                )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = self.VOXEL_SIZE

    def toVox(self, coords, feats, labels):
        if "Lidar" in self.cfg:
            if SPCONV_VER==1:
                voxel_output = self.voxel_generator.generate(np.concatenate((coords, feats), 1))
            else:
                voxel_output = self.voxel_generator(torch.from_numpy(np.concatenate((coords, feats), 1)).contiguous())
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
    
    def prepare_data(self, data_dict, index):
        
        if self.linear_probe and data_dict.get('gt_boxes_lidar', None) is not None:
            if self.cfg['LABEL_TYPE']  == 'class_names':
                # Select gt names and boxes that are in self.class_names
                selected = [i for i, x in enumerate(data_dict['gt_names']) if x in self.class_names]
                selected = np.array(selected, dtype=np.int64)
                data_dict['gt_boxes_lidar'] = data_dict['gt_boxes_lidar'][selected]
                data_dict['gt_names'] = data_dict['gt_names'][selected]

                # Append gt_name index as the last column: background: 0, car:1, ped:2, cyc:3
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
                gt_boxes = np.concatenate((data_dict['gt_boxes_lidar'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes_lidar'] = gt_boxes
                data_dict.pop('gt_names', None)
            elif self.cfg['LABEL_TYPE']  == 'objects':
                num_gt_boxes = data_dict['gt_boxes_lidar'].shape[0]

                # Append 1 to indicate object class
                gt_boxes = np.concatenate((data_dict['gt_boxes_lidar'], np.ones(num_gt_boxes).reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes_lidar'] = gt_boxes
                data_dict.pop('gt_names', None)

        points = data_dict['data']
        if not self.linear_probe:
            points_moco = data_dict['data_moco']


        cfg = self.cfg
        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        if False: #cfg["DATA_TYPE"] == "point_vox":
            # Across format
            item = {"data": [], "data_aug_matrix": [], 
            "vox": [], "vox_aug_matrix": []}

            item["data"].append(points)
            item["vox"].append(np.copy(points))
            
            if not self.linear_probe:
                item["data_moco"] = []
                item["data_moco_aug_matrix"] = []
                item["vox_moco"] = []
                item["vox_moco_aug_matrix"] = []
                item["data_moco"].append(points_moco)
                item["vox_moco"].append(np.copy(points_moco))

            #item["data_valid"].append(1)
        else:
            # Within format: data is either points or later is voxelized
            item = {"data": [], "data_aug_matrix": []}
            item["data"].append(points)

            if not self.linear_probe:
                item["data_moco"] = []
                item["data_moco_aug_matrix"] = []
                item["data_moco"].append(points_moco)

        # Apply the transformation here
        if False: #(cfg["DATA_TYPE"] == "point_vox"):
            # Points
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data"] = tempdata["data"]
            item["data_aug_matrix"] = tempdata['aug_trans_matrix']
            
            if not self.linear_probe:
                # Points MoCo
                tempitem = {"data": item["data_moco"]}
                tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
                item["data_moco"] = tempdata["data"]
                item["data_moco_aug_matrix"] = tempdata['aug_trans_matrix']

            # Vox
            tempitem = {"data": item["vox"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:, :3]
            feats = tempdata["data"][0][:, 3:6] #* 255.0  # np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox"] = [self.toVox(coords, feats, labels)]
            item["vox_aug_matrix"] = tempdata['aug_trans_matrix']

            if not self.linear_probe:
                # Vox MoCo
                tempitem = {"data": item["vox_moco"]}
                tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
                coords = tempdata["data"][0][:, :3]
                feats = tempdata["data"][0][:, 3:6] #* 255.0  # np.ones(coords.shape)*255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)
                item["vox_moco"] = [self.toVox(coords, feats, labels)]
                item["vox_moco_aug_matrix"] = tempdata['aug_trans_matrix']
        else:
            # Points -> transform -> voxelize if Vox
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
            if False: #cfg["VOX"]:
                coords = tempdata["data"][0][:, :3]
                feats = tempdata["data"][0][:, 3:6] # *255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)
                item["data"] = [self.toVox(coords, feats, labels)]
            else:
                item["data"] = tempdata["data"]
            item["data_aug_matrix"] = tempdata['aug_trans_matrix']
            
            if not self.linear_probe:
                # Points MoCo-> transform -> voxelize if Vox
                tempitem = {"data": item["data_moco"]}
                tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
                if False: #cfg["VOX"]:
                    coords = tempdata["data"][0][:, :3]
                    feats = tempdata["data"][0][:, 3:6] #* 255.0  # np.ones(coords.shape)*255.0
                    labels = np.zeros(coords.shape[0]).astype(np.int32)
                    item["data_moco"] = [self.toVox(coords, feats, labels)]
                else:
                    item["data_moco"] = tempdata["data"]
                item["data_moco_aug_matrix"] = tempdata['aug_trans_matrix']

        if self.linear_probe:
            item['linear_probe'] = True    
        data_dict.update(item)

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

    def get_batchsize_per_replica(self):
        # this searches for batchsize_per_replica in self and then in self.dataset
        return getattr(self, "batchsize_per_replica", 1)

    def get_global_batchsize(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        return self.get_batchsize_per_replica() * world_size

def get_fov_flag(pts_rect, img_shape, calib):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

class DenseKittiDataset(DepthContrastDataset):
    def __init__(self, cfg, linear_probe=False, mode='train', logger=None):
        super().__init__(cfg, linear_probe=linear_probe, mode=mode, logger=logger)

        # Dense
        self.root_dense_path = self.root_path / 'data' / 'dense'

        self.sensor_type = self.cfg["SENSOR_TYPE"]
        self.signal_type = self.cfg["SIGNAL_TYPE"]
        self.dense_lidar_folder = f'lidar_{self.sensor_type}_{self.signal_type}_FOV_clustered_train_all_60'
        self.dense_calib = self.get_dense_calib(self.sensor_type)

        # Kitti
        self.root_kitti_path = self.root_path / 'data' / 'kitti'

        self.infos = []
        self.include_dense_kitti_data()

        #SnowFall augmentation
        self.snowfall_rates = [0.5, 0.5, 1.0, 2.0, 2.5, 1.5]      # mm/h
        self.terminal_velocities = [2.0, 1.2, 1.6, 2.0, 1.6, 0.6] # m/s

        self.rainfall_rates = []

        for i in range(len(self.snowfall_rates)):

            self.rainfall_rates.append(snowfall_rate_to_rainfall_rate(self.snowfall_rates[i],
                                                                      self.terminal_velocities[i]))
    def include_dense_kitti_data(self):
        if self.logger is not None:
            self.logger.add_line('Loading DENSE and Kitti dataset')
        dense_kitti_infos = []

        num_skipped_infos = 0
        for info_path in self.cfg["INFO_PATHS"][self.mode]:
            if self.logger is not None:
                self.logger.add_line(f'Loading info path {info_path}')
            if 'dense' in info_path:
                info_path = self.root_dense_path / info_path
            elif 'kitti' in info_path:
                info_path = self.root_kitti_path / info_path
            else:
                raise ValueError('Only Kitti and Dense infos are supported!')

            if not info_path.exists():
                num_skipped_infos += 1
                if self.logger is not None:
                    self.logger.add_line(f'Path does not exist!: {info_path}')
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                dense_kitti_infos.extend(infos)

        # shuffle infos
        perm_idx = np.random.permutation(len(dense_kitti_infos))
        self.infos.extend(np.array(dense_kitti_infos)[perm_idx].tolist())
        #self.infos.extend(dense_kitti_infos[:])

        # To work on a subset of dense_infos for debugging, 
        # comment the line above and uncomment below
        # perm = np.random.permutation(len(dense_infos))
        # idx = perm[:int(len(dense_infos)/10)]
        # self.dense_infos.extend(np.array(dense_infos)[idx].tolist())

        if self.logger is not None:
            self.logger.add_line('Total skipped info %s' % num_skipped_infos)
            self.logger.add_line('Total samples for DENSE and Kitti dataset: %d' %
                            (len(dense_kitti_infos)))
    
    def get_dense_calib(self, sensor: str = 'hdl64'):
        calib_file = self.root_dense_path / f'calib_{sensor}.txt'
        assert calib_file.exists(), f'{calib_file} not found'
        return calibration_kitti.Calibration(calib_file)

    def get_dense_lidar(self, idx):
        lidar_file = self.root_dense_path / self.dense_lidar_folder / ('%s.bin' % idx)
        assert lidar_file.exists(), f'{lidar_file} not found'
        # try:
        #     pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)
        # except:
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 6)
        return pc
    
    def get_kitti_calib(self, velo_parent_dir, idx):
        calib_file = self.root_kitti_path / velo_parent_dir / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_kitti_lidar(self, velo_parent_dir, idx):
        lidar_file = self.root_kitti_path / velo_parent_dir /'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        pc[:,3] = np.round(pc[:,3] * 255)
        return pc

    def __len__(self):
        return len(self.infos)

    def crop_pc(self, pc, calib, img_shape):
        upper_idx = np.sum((pc[:, 0:3] <= self.point_cloud_range[3:6]).astype(np.int32), 1) == 3
        lower_idx = np.sum((pc[:, 0:3] >= self.point_cloud_range[0:3]).astype(np.int32), 1) == 3

        new_pointidx = (upper_idx) & (lower_idx)
        pc = pc[new_pointidx, :]

        # Extract FOV points
        if self.cfg['FOV_POINTS_ONLY']:
            pts_rect = calib.lidar_to_rect(pc[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, img_shape, calib)
            pc = pc[fov_flag]

        return pc
   
    def __getitem__(self, index):
        
        info = copy.deepcopy(self.infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        dataset = info.get('dataset', 'dense')

        clustered = False
        if dataset == 'kitti':
            calib = self.get_kitti_calib(info['velodyne_parent_dir'], sample_idx)
            points = self.get_kitti_lidar(info['velodyne_parent_dir'], sample_idx) #xyzi or xyzi,cluster_id
            clustered = points.shape[1] > 4
        else:
            calib = self.dense_calib
            points = self.get_dense_lidar(sample_idx)
            clustered = points.shape[1] > 5

        if not self.linear_probe:
            assert clustered, 'PC is unclustered! This is segContrast'
        
        # Crop and extract FOV points
        points = self.crop_pc(points, calib, img_shape)

        # if not clustered:
        #     #V.draw_scenes(points=points, color_feature='intensity')
        #     points = clusterize_pcd(points, 1000, dist_thresh=0.15, eps=1.0)
        #     visualize_pcd_clusters(points)
        

        points_moco = np.copy(points) #points_weather
        
        weather = 'clear'
        if 'weather' in info:
            weather = info['weather'] #kitti
        else:
            weather = info['annos']['weather'] #dense
        
        data_dict = {}

        if not self.linear_probe and self.cfg['APPLY_WEATHER_AUG']:
            apply_dror =  weather != 'clear' and 'DROR' in self.cfg
            dror_applied = False
            if apply_dror:
                points_moco, dror_applied = dror(self.cfg, points_moco, sample_idx, self.logger, self.root_dense_path)

            apply_upsampling = dror_applied and 'UPSAMPLE' in self.cfg
            upsample_applied = False
            if apply_upsampling:
                points_moco, upsample_applied = nn_upsample(self.cfg, points_moco)

            # Snowfall Augmentation
            snowfall_augmentation_applied = False
            apply_snow = weather == 'clear' and 'SNOW' in self.cfg and dataset == 'dense'
            if apply_snow:
                points_moco, snowfall_augmentation_applied = snow_sim(self.cfg, self.logger, self.rainfall_rates, sample_idx, self.root_dense_path, self.dense_lidar_folder, points_moco)
            
            # Wet Surface Augmentation
            wet_surface_applied = False
            apply_wet_surface = weather == 'clear' and 'WET_SURFACE' in self.cfg and dataset == 'dense'
            if apply_wet_surface:
                points_moco, wet_surface_applied = wet_surface_sim(self.cfg, snowfall_augmentation_applied, points_moco, self.logger)
            
            #Fog augmentation
            fog_applied = False
            apply_fog = weather == 'clear' and not snowfall_augmentation_applied and 'FOG_AUGMENTATION' in self.cfg
            if apply_fog:
                points_moco, fog_applied = fog_sim(self.cfg, points_moco) 

        #print(f'dataset: {dataset}, weather: {weather}, dror: {dror_applied}, up: {upsample_applied}, snow: {snowfall_augmentation_applied}, wet: {wet_surface_applied}, fog: {fog_applied}')
        
        # Add gt_boxes for linear probing
        if self.linear_probe:
            data_dict['gt_names'] = info['annos']['name']
            data_dict['gt_boxes_lidar'] = info['annos']['gt_boxes_lidar']
            assert data_dict['gt_names'].shape[0] == data_dict['gt_boxes_lidar'].shape[0]

        # # TODO: mask boxes outside point cloud range

        
        data_dict['data'] = np.hstack((points[:,:4], points[:,-1].reshape((-1,1)))) #x,y,z,i #drop channel or label, add cluster id

        if not self.linear_probe:
            points_moco = self.crop_pc(points_moco, calib, img_shape)
            data_dict['data_moco'] = np.hstack((points_moco[:,:4], points_moco[:, -1].reshape((-1,1)))) #x,y,z,i #drop channel or label , add cluster id

        #V.draw_scenes(points=points, color_feature='intensity')
        #V.draw_scenes(points=points_moco, color_feature='intensity')
        # if fog_applied or snowfall_augmentation_applied:
        #     visualize_pcd_clusters(points)
        #     visualize_pcd_clusters(points_moco)
        # Prepare points and Transform 
        data_dict = self.prepare_data(data_dict, index)

        return data_dict