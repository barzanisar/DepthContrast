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

    def __init__(self, cfg, mode='train', logger=None):
        self.mode = mode

        self.logger = logger
        self.cfg = cfg
        self.batchsize_per_replica = cfg["BATCHSIZE_PER_REPLICA"]
        self.dataset_names = cfg["DATASET_NAMES"]
        self.root_path = (Path(__file__) / '../..').resolve()  # DepthContrast

        # Define voxelizer here
        if ("Lidar" in cfg) and cfg["VOX"]:
            self.VOXEL_SIZE = [0.1, 0.1, 0.2]

            if "waymo" in self.dataset_names:
                self.point_cloud_range = WAYMO_POINT_RANGE
            elif "dense" in self.dataset_names:
                self.point_cloud_range = DENSE_POINT_RANGE

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

    def prepare_data(self, points, idx):

        #Crop given point cloud range
        upper_idx = np.sum((points[:, 0:3] <= self.point_cloud_range[3:6]).astype(np.int32), 1) == 3
        lower_idx = np.sum((points[:, 0:3] >= self.point_cloud_range[0:3]).astype(np.int32), 1) == 3

        new_pointidx = (upper_idx) & (lower_idx)
        points = points[new_pointidx, :]

        cfg = self.cfg
        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        if cfg["DATA_TYPE"] == "point_vox":
            item = {"data": [], "data_valid": [],
                    "data_moco": [], "vox": [], "vox_moco": []}

            item["data"].append(points)
            item["data_moco"].append(np.copy(points))
            item["vox"].append(np.copy(points))
            item["vox_moco"].append(np.copy(points))
            item["data_valid"].append(1)
        else:
            item = {"data": [], "data_moco": [],
                    "data_valid": [], "data_idx": []}

            item["data"].append(points)
            item["data_moco"].append(np.copy(points))
            item["data_valid"].append(1)

        # Make copies for moco setting
        item["label"] = []
        item["label"].append(idx)

        # Apply the transformation here
        if (cfg["DATA_TYPE"] == "point_vox"):
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data"] = tempdata["data"]

            tempitem = {"data": item["data_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data_moco"] = tempdata["data"]

            tempitem = {"data": item["vox"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:, :3]
            feats = tempdata["data"][0][:, 3:6] * 255.0  # np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox"] = [self.toVox(coords, feats, labels)]

            tempitem = {"data": item["vox_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:, :3]
            feats = tempdata["data"][0][:, 3:6] * 255.0  # np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox_moco"] = [self.toVox(coords, feats, labels)]
        else:
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
            if cfg["VOX"]:
                coords = tempdata["data"][0][:, :3]
                feats = tempdata["data"][0][:, 3:6]*255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)
                item["data"] = [self.toVox(coords, feats, labels)]
            else:
                item["data"] = tempdata["data"]

            tempitem = {"data": item["data_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
            if cfg["VOX"]:
                coords = tempdata["data"][0][:, :3]
                feats = tempdata["data"][0][:, 3:6] * 255.0  # np.ones(coords.shape)*255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)
                item["data_moco"] = [self.toVox(coords, feats, labels)]
            else:
                item["data_moco"] = tempdata["data"]
        return item

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

    def __init__(self, cfg, mode='train', logger=None):
        super().__init__(cfg, mode=mode, logger=logger)
        self.root_data_path = self.root_path / 'data' / 'dense'  # DepthContrast/data/waymo
        
        self.sensor_type = self.cfg["SENSOR_TYPE"]
        self.signal_type = self.cfg["SIGNAL_TYPE"]
        self.lidar_folder = f'lidar_{self.sensor_type}_{self.signal_type}'

        self.dense_infos = []
        self.include_dense_data()

        # To create infos
        # self.split = self.cfg["DATA_PATHS"][self.mode]
        # split_dir = self.root_data_path / 'ImageSets' / f'{self.split}.txt'

        # if split_dir.exists():
        #     self.sample_id_list = ['_'.join(x.strip().split(',')) for x in open(split_dir).readlines()]
        # else:
        #     self.sample_id_list = None

    def include_dense_data(self):
        if self.logger is not None:
            self.logger.info('Loading DENSE dataset')
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
            self.logger.info('Total skipped info %s' % num_skipped_infos)
            self.logger.info('Total samples for DENSE dataset: %d' %
                             (len(dense_infos)))

    def get_lidar(self, idx):
        lidar_file = self.root_data_path / self.lidar_folder / ('%s.bin' % idx)
        assert lidar_file.exists(), f'{lidar_file} not found'
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

    def get_calib(self):
        calib_file = self.root_data_path / f'calib_{self.sensor_type}.txt'
        assert calib_file.exists(), f'{calib_file} not found'
        return calibration_kitti.Calibration(calib_file)

    def __len__(self):
        return len(self.dense_infos)

    def __getitem__(self, index):
        # index = 563                               # this VLP32 index does not have a single point in the camera FOV

        info = copy.deepcopy(self.dense_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)

        data_dict = self.prepare_data(points[:,:4], index)

        calib = self.get_calib()

        data_dict.update({'frame_id': sample_idx})

        if 'annos' in info:
            annos = info['annos']

            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_boxes_camera = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                gt_boxes_camera, calib)

            gt_names = annos['name']

            data_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        return data_dict
