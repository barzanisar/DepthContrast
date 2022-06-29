# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import Dataset
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
import logging
import os

import numpy as np
from pathlib import Path
import pickle
import copy
import sys

from datasets.transforms.augment3d import get_transform3d

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'OpenPCDet'))


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


# Waymo lidar range
WAYMO_POINT_RANGE = np.array([0., -75.,  -3.,  75.0,  75.,   3.], dtype=np.float32)
# KITTI and DENSE range
DENSE_POINT_RANGE = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)

class DepthContrastDataset(Dataset):
    """Base Self Supervised Learning Dataset Class."""
    # TODO: use own logger for rank = 0

    def __init__(self, cfg, phase = 'linear_probe', mode='train', logger=None):
        self.mode = mode
        self.phase = phase
        self.logger = logger
        self.cfg = cfg
        self.batchsize_per_replica = cfg["BATCHSIZE_PER_REPLICA"]
        self.dataset_names = cfg["DATASET_NAMES"]
        self.root_path = (Path(__file__) / '../..').resolve()  # DepthContrast
        if "WaymoDataset" in self.dataset_names:
            self.point_cloud_range = WAYMO_POINT_RANGE
            self.class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
        elif "DenseDataset" in self.dataset_names:
            self.point_cloud_range = DENSE_POINT_RANGE
            self.class_names = ['PassengerCar', 'Pedestrian', 'RidableVehicle', 'LargeVehicle']

        # Define voxelizer here
        if ("Lidar" in cfg) and cfg["VOX"]:
            self.VOXEL_SIZE = [0.1, 0.1, 0.2]

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
                    num_point_features=4,
                    max_num_points_per_voxel=self.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=self.MAX_NUMBER_OF_VOXELS
                )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = self.VOXEL_SIZE

    def toVox(self, coords, feats, labels):
        if "Lidar" in self.cfg:
            if SPCONV_VER == 1:
                voxel_output = self.voxel_generator.generate(
                    np.concatenate((coords, feats), 1))
            else:
                voxel_output = self.voxel_generator(torch.from_numpy(
                    np.concatenate((coords, feats), 1)).contiguous())
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

            data_dict = {}
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
            return data_dict

    def prepare_data(self, data_dict, index):
        
        points = data_dict['data']
        if self.phase == 'linear_probe' and data_dict.get('gt_boxes_lidar', None) is not None:
            if self.cfg['LABEL_TYPE']  == 'class_names':
                selected = [i for i, x in enumerate(data_dict['gt_names']) if x in self.class_names]
                selected = np.array(selected, dtype=np.int64)
                data_dict['gt_boxes_lidar'] = data_dict['gt_boxes_lidar'][selected]
                data_dict['gt_names'] = data_dict['gt_names'][selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
                gt_boxes = np.concatenate((data_dict['gt_boxes_lidar'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes_lidar'] = gt_boxes
                data_dict.pop('gt_names', None)
            elif self.cfg['LABEL_TYPE']  == 'objects':
                num_gt_boxes = data_dict['gt_boxes_lidar'].shape[0]
                gt_boxes = np.concatenate((data_dict['gt_boxes_lidar'], np.ones(num_gt_boxes).reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes_lidar'] = gt_boxes
                data_dict.pop('gt_names', None)

        #Crop given point cloud range
        upper_idx = np.sum((points[:, 0:3] <= self.point_cloud_range[3:6]).astype(np.int32), 1) == 3
        lower_idx = np.sum((points[:, 0:3] >= self.point_cloud_range[0:3]).astype(np.int32), 1) == 3

        new_pointidx = (upper_idx) & (lower_idx)
        points = points[new_pointidx, :]

        cfg = self.cfg
        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        if cfg["DATA_TYPE"] == "point_vox":
            # Across format
            item = {"data": [],
                    "data_moco": [], "vox": [], "vox_moco": []}

            item["data"].append(points)
            item["data_moco"].append(np.copy(points))
            item["vox"].append(np.copy(points))
            item["vox_moco"].append(np.copy(points))
            #item["data_valid"].append(1)
        else:
            # Within format: data is either points or later is voxelized
            item = {"data": [], "data_moco": []}

            item["data"].append(points)
            item["data_moco"].append(np.copy(points))
            #item["data_valid"].append(1)

        # # Make copies for moco setting #TODO: remove if not needed
        # item["label"] = []
        # item["label"].append(index)

        # Apply the transformation here
        if (cfg["DATA_TYPE"] == "point_vox"):
            # Points
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data"] = tempdata["data"]

            # Points MoCo
            tempitem = {"data": item["data_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data_moco"] = tempdata["data"]

            # Vox
            tempitem = {"data": item["vox"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:, :3]
            feats = tempdata["data"][0][:, 3:6] * 255.0  # np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox"] = [self.toVox(coords, feats, labels)]

            # Vox MoCo
            tempitem = {"data": item["vox_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:, :3]
            feats = tempdata["data"][0][:, 3:6] * 255.0  # np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox_moco"] = [self.toVox(coords, feats, labels)]
        else:
            # Points -> transform -> voxelize if Vox
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
            if cfg["VOX"]:
                coords = tempdata["data"][0][:, :3]
                feats = tempdata["data"][0][:, 3:6]*255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)
                item["data"] = [self.toVox(coords, feats, labels)]
            else:
                item["data"] = tempdata["data"]
            
            # Points MoCo-> transform -> voxelize if Vox
            tempitem = {"data": item["data_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
            if cfg["VOX"]:
                coords = tempdata["data"][0][:, :3]
                feats = tempdata["data"][0][:, 3:6] * 255.0  # np.ones(coords.shape)*255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)
                item["data_moco"] = [self.toVox(coords, feats, labels)]
            else:
                item["data_moco"] = tempdata["data"]
            
            data_dict.update(item)
            #data_dict.pop('points', None)

        return data_dict

    def get_batchsize_per_replica(self):
        # this searches for batchsize_per_replica in self and then in self.dataset
        return getattr(self, "batchsize_per_replica", 1)

    def get_global_batchsize(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        return self.get_batchsize_per_replica() * world_size
    
    def __getitem__(self, idx):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError


class DenseDataset(DepthContrastDataset):

    def __init__(self, cfg, phase='linear_probe', mode='train', logger=None):
        super().__init__(cfg, phase=phase, mode=mode, logger=logger)
        self.root_data_path = self.root_path / 'data' / 'dense'  # DepthContrast/data/waymo
        
        self.sensor_type = self.cfg["SENSOR_TYPE"]
        self.signal_type = self.cfg["SIGNAL_TYPE"]
        self.lidar_folder = f'lidar_{self.sensor_type}_{self.signal_type}'

        self.dense_infos = []
        self.include_dense_data()

    def include_dense_data(self):
        if self.logger is not None:
            self.logger.add_line('Loading DENSE dataset')
        dense_infos = []

        num_skipped_infos = 0
        for info_path in self.cfg["INFO_PATHS"][self.mode]:
            info_path = self.root_data_path / info_path
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                dense_infos.extend(infos)

        self.dense_infos.extend(dense_infos[:])

        if self.logger is not None:
            self.logger.add_line('Total skipped info %s' % num_skipped_infos)
            self.logger.add_line('Total samples for DENSE dataset: %d' %
                             (len(dense_infos)))

    def get_lidar(self, idx):
        lidar_file = self.root_data_path / self.lidar_folder / ('%s.bin' % idx)
        assert lidar_file.exists(), f'{lidar_file} not found'
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

    # def get_calib(self):
    #     calib_file = self.root_data_path / f'calib_{self.sensor_type}.txt'
    #     assert calib_file.exists(), f'{calib_file} not found'
    #     return calibration_kitti.Calibration(calib_file)

    def __len__(self):
        return len(self.dense_infos)

    def __getitem__(self, index):
        # index = 563                               # this VLP32 index does not have a single point in the camera FOV

        info = copy.deepcopy(self.dense_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        
        data_dict = {}
        
        # Drop gt_names == DontCare, gt_boxes already don't have dontcare boxes
        keep_indices = [i for i,x in enumerate(info['annos']['name']) if x != 'DontCare']
        data_dict['gt_names'] = info['annos']['name'][keep_indices]
        data_dict['gt_boxes_lidar'] = info['annos']['gt_boxes_lidar']
        
        # Drop gt_names and boxes with negative h,w,l i.e. not visible in lidar
        keep_indices = [i for i in range(data_dict['gt_boxes_lidar'].shape[0]) if data_dict['gt_boxes_lidar'][i, 3] > 0]
        data_dict['gt_names'] = data_dict['gt_names'][keep_indices]
        data_dict['gt_boxes_lidar'] = data_dict['gt_boxes_lidar'][keep_indices]

        # if np.any(data_dict['gt_boxes_lidar'][:,3:6] < 0):
        #     print(f"Printing h,w,l of gt boxes: {data_dict['gt_boxes_lidar'][:,3:6].shape}")
        #     print(data_dict['gt_boxes_lidar'][:,3:6])
        #     b=1
        # if data_dict['gt_names'].shape[0] != data_dict['gt_boxes_lidar'].shape[0]:
        #     b=1
        assert data_dict['gt_names'].shape[0] == data_dict['gt_boxes_lidar'].shape[0]

        # Change Vehicle or Obstacle class to PassengerCar
        for i, name in enumerate(data_dict['gt_names']):
            if name in ['Vehicle', 'Obstacle']:
                data_dict['gt_names'][i] = 'PassengerCar'

        # calib = self.get_calib()        

        # if 'annos' in info:
        #     annos = info['annos']

        #     loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
        #     gt_boxes_camera = np.concatenate(
        #         [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        #     gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
        #         gt_boxes_camera, calib)

        #     gt_names = annos['name']
        #     b=1

        # Prepare points and Transform 
        data_dict['data'] = points[:,:4]
        data_dict = self.prepare_data(data_dict, index)


        return data_dict
