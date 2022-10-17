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

from pcdet.utils import calibration_kitti, box_utils
from lib.LiDAR_snow_sim.tools.snowfall.sampling import snowfall_rate_to_rainfall_rate
from utils.pcd_preprocess import *
from utils.data_map import *


#from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V

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
#np.random.seed(1024)

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
            self.class_names = ['Car', 'Pedestrian', 'Cyclist', 'Van'] #, 'LargeVehicle' is now grouped under PassengerCar

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
        
        cfg = self.cfg
        points = data_dict['data']
        if not self.linear_probe:
            points_moco = data_dict['data_moco']
        
        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        if False: #cfg["DATA_TYPE"] == "point_vox":
            # Across format
            item = {"data": [], "vox": []}

            item["data"].append(points)
            item["vox"].append(np.copy(points))
            
            if not self.linear_probe:
                item["data_moco"] = []
                item["vox_moco"] = []
                item["data_moco"].append(points_moco)
                item["vox_moco"].append(np.copy(points_moco))

            #item["data_valid"].append(1)
        else:
            # Within format: data is either points or later is voxelized
            item = {"data": []}
            item["data"].append(points)

            if not self.linear_probe:
                item["data_moco"] = []
                item["data_moco"].append(points_moco)

        # Apply the transformation here
        if False: #(cfg["DATA_TYPE"] == "point_vox"):
            # Points
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data"] = tempdata["data"]
            
            if not self.linear_probe:
                # Points MoCo
                tempitem = {"data": item["data_moco"]}
                tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
                item["data_moco"] = tempdata["data"]

            # Vox
            tempitem = {"data": item["vox"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:, :3]
            feats = tempdata["data"][0][:, 3:6] #* 255.0  # np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox"] = [self.toVox(coords, feats, labels)]

            if not self.linear_probe:
                # Vox MoCo
                tempitem = {"data": item["vox_moco"]}
                tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
                coords = tempdata["data"][0][:, :3]
                feats = tempdata["data"][0][:, 3:6] #* 255.0  # np.ones(coords.shape)*255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)
                item["vox_moco"] = [self.toVox(coords, feats, labels)]
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
    def __init__(self, cfg, cluster, linear_probe=False, mode='train', logger=None):
        super().__init__(cfg, linear_probe=linear_probe, mode=mode, logger=logger)

        self.cluster = cluster

        self.num_dense_features = 5 # [x,y,z,i,channel]
        if self.cluster:
            self.num_dense_features = 6 # [x,y,z,i,cluster_id]
        
        self.root_dense_path = self.root_path / 'data' / 'dense'
        self.root_kitti_path = self.root_path / 'data' / 'kitti'

        # Dense
        include_dense_infos = False
        if self.cfg.get("INFO_PATHS", None) is not None:
            for info_path in self.cfg["INFO_PATHS"][self.mode]:
                if 'dense_infos' in info_path:
                    include_dense_infos = True

        if include_dense_infos:
            self.sensor_type = self.cfg["SENSOR_TYPE"]
            self.signal_type = self.cfg["SIGNAL_TYPE"]
            self.dense_lidar_folder = f'lidar_{self.sensor_type}_{self.signal_type}_FOV_clustered_train_all_60' if self.cluster else f'lidar_{self.sensor_type}_{self.signal_type}'
            self.dense_calib = self.get_dense_calib(self.sensor_type)
        

        self.infos = []
        if self.cfg.get("INFO_PATHS", None) is not None:
            self.include_dense_kitti_data()

        # Semantic Kitti
        if "SEMANTIC_KITTI" in self.cfg:
            self.root_semkitti_path = self.root_path / 'data' / 'semantic_kitti'
            self.include_sem_kitti_infos()


        #SnowFall augmentation
        self.snowfall_rates = [0.5, 0.5, 1.0, 2.0, 2.5, 1.5]      # mm/h
        self.terminal_velocities = [2.0, 1.2, 1.6, 2.0, 1.6, 0.6] # m/s

        self.rainfall_rates = []

        for i in range(len(self.snowfall_rates)):

            self.rainfall_rates.append(snowfall_rate_to_rainfall_rate(self.snowfall_rates[i],  self.terminal_velocities[i]))
    
    def include_sem_kitti_infos(self):
        sem_kitti_points_datapath = []
        sem_kitti_labels_datapath = []
        sem_kitti_infos = []
        if self.logger is not None:
            self.logger.add_line('Loading Semantic Kitti dataset')

        for seq in self.cfg["SEMANTIC_KITTI"][self.mode]:
            point_seq_path = os.path.join(self.root_semkitti_path, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            sem_kitti_points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

            try:
                label_seq_path = os.path.join(self.root_semkitti_path, 'dataset', 'sequences', seq, 'labels')
                point_seq_label = os.listdir(label_seq_path)
                point_seq_label.sort()
                sem_kitti_labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]
            except:
                pass

        for point_path, label_path in zip(sem_kitti_points_datapath, sem_kitti_labels_datapath):
            info = {'point_cloud': {'lidar_idx': point_path}, 'dataset': 'semantic_kitti', 'label_path': label_path}
            #TODO add annos for linear probe
            sem_kitti_infos.append(info)
        
        if self.linear_probe:
            # Only linear probe on 10% of sem kitti data to save time
            perm_idx = np.random.permutation(len(sem_kitti_infos))
            idx = perm_idx[:int(len(sem_kitti_infos)/10)]
            self.infos.extend(np.array(sem_kitti_infos)[idx].tolist())
        else:
            self.infos.extend(sem_kitti_infos)
        
        
        # shuffle infos
        perm_idx = np.random.permutation(len(self.infos))
        self.infos = np.array(self.infos)[perm_idx].tolist()
        if self.logger is not None:
            self.logger.add_line('Total Semantic Kitti samples loaded: %d' % (len(sem_kitti_infos)))
            self.logger.add_line('Total samples loaded: %d' % (len(self.infos)))

    def include_dense_kitti_data(self):
        if self.logger is not None:
            self.logger.add_line('Loading DENSE and Kitti dataset')
        dense_kitti_infos = []

        num_skipped_infos = 0
        for info_path in self.cfg["INFO_PATHS"][self.mode]:
            if self.logger is not None:
                self.logger.add_line(f'Loading info path {info_path}')
            if 'dense_infos' in info_path:
                info_path = self.root_dense_path / info_path
            elif 'kitti_infos' in info_path:
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
        # To work on a subset of dense_infos for debugging, 
        #perm_idx = perm_idx[:int(len(dense_kitti_infos)/10)]
        self.infos.extend(np.array(dense_kitti_infos)[perm_idx].tolist())

        if self.logger is not None:
            self.logger.add_line('Total skipped info %s' % num_skipped_infos)
            self.logger.add_line('Total samples for DENSE and Kitti dataset: %d' %
                            (len(dense_kitti_infos)))
    
    def get_dense_calib(self, sensor: str = 'hdl64'):
        calib_file = self.root_dense_path / f'calib_{sensor}.txt'
        assert calib_file.exists(), f'{calib_file} not found'
        return calibration_kitti.Calibration(calib_file)
    
    def get_kitti_calib(self, velo_parent_dir, idx):
        calib_file = self.root_kitti_path / velo_parent_dir / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_dense_lidar(self, idx):
        lidar_file = self.root_dense_path / self.dense_lidar_folder / ('%s.bin' % idx)
        assert lidar_file.exists(), f'{lidar_file} not found'
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, self.num_dense_features)
        return pc

    def get_kitti_lidar(self, velo_parent_dir, idx):
        lidar_file = self.root_kitti_path / velo_parent_dir / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists(), f'{lidar_file} not found'
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        pc[:,3] = np.round(pc[:,3] * 255)
        if self.cluster:
            cluster_id_file = Path(str(lidar_file).replace(velo_parent_dir, f'{velo_parent_dir}_clustered'))
            assert cluster_id_file.exists(), f'{cluster_id_file} not found'
            cluster_ids = np.fromfile(str(cluster_id_file), dtype=np.int16).reshape(-1, 1)
            cluster_ids = cluster_ids.astype(np.float32)
            pc = np.hstack((pc, cluster_ids))
        return pc

    def get_sem_kitti_lidar(self, idx):
        lidar_file = Path(idx)
        assert lidar_file.exists(), f'{lidar_file} not found'
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        pc[:,3] = np.round(pc[:,3] * 255)

        if self.cluster:
            cluster_id_file = Path(idx.replace('dataset', 'dataset_clustered'))
            assert cluster_id_file.exists(), f'{cluster_id_file} not found'
            cluster_ids = np.fromfile(str(cluster_id_file), dtype=np.int16).reshape(-1, 1)
            cluster_ids = cluster_ids.astype(np.float32)
            pc = np.hstack((pc, cluster_ids))
        return pc

    def __len__(self):
        return len(self.infos)

    def crop_pc(self, pc, calib=None, img_shape=None):
        upper_idx = np.sum((pc[:, 0:3] <= self.point_cloud_range[3:6]).astype(np.int32), 1) == 3
        lower_idx = np.sum((pc[:, 0:3] >= self.point_cloud_range[0:3]).astype(np.int32), 1) == 3

        new_pointidx = (upper_idx) & (lower_idx)
        pc = pc[new_pointidx, :]

        # Extract FOV points
        if self.cfg['FOV_POINTS_ONLY'] and calib is not None:
            pts_rect = calib.lidar_to_rect(pc[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, img_shape, calib)
            pc = pc[fov_flag]

        return pc
   
    def __getitem__(self, index):
        
        info = copy.deepcopy(self.infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        dataset = info.get('dataset', 'dense')
        calib = None
        img_shape = None

        clustered = False
        if dataset == 'kitti':
            calib = self.get_kitti_calib(info['velodyne_parent_dir'], sample_idx)
            img_shape = info['image']['image_shape']
            points = self.get_kitti_lidar(info['velodyne_parent_dir'], sample_idx) #xyzi or xyzi,cluster_id
            clustered = points.shape[1] > 4
        elif dataset == 'semantic_kitti':
            points = self.get_sem_kitti_lidar(sample_idx)
            clustered = points.shape[1] > 4
        else:
            calib = self.dense_calib
            img_shape = info['image']['image_shape']
            points = self.get_dense_lidar(sample_idx)
            clustered = points.shape[1] > 5

        if self.cluster:
            assert clustered, 'PC is unclustered! This is segContrast'
        
                # Add object or class label for every point for linear probing
        if self.linear_probe:
            if "SEMANTIC_KITTI" not in self.cfg:
                if False: #self.cfg['LABEL_TYPE']  == 'class_names':
                    # Select gt names and boxes that are in self.class_names
                    selected = [i for i, x in enumerate(info['annos']['name']) if x != 'DontCare'] #in self.class_names
                    selected = np.array(selected, dtype=np.int64)
                    info['annos']['name'] = info['annos']['name'][selected]

                gt_boxes_lidar = info['annos']['gt_boxes_lidar']
                num_objects = gt_boxes_lidar.shape[0]
                corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar[:,:7]) #nboxes, 8 corners, 3coords
                labels = np.zeros(points.shape[0])
                for k in range(num_objects):
                    flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                    labels[flag] = 1 #if self.cfg['LABEL_TYPE']  == 'objects' else self.class_names.index(info['annos']['name'][k]) + 1
                points = np.hstack((points[:,:4], labels.reshape((-1,1)))) #x,y,z,i,label
        
            else:
                labels = np.fromfile(info['label_path'], dtype=np.uint32)
                labels = labels.reshape((-1))
                labels = labels & 0xFFFF

                #remap labels to learning values
                labels = np.vectorize(learning_map.get)(labels)
                labels = np.expand_dims(labels, axis=-1)
                unlabeled = labels[:,0] == 0

                # remove unlabeled points
                labels = np.delete(labels, unlabeled, axis=0)
                points = np.delete(points, unlabeled, axis=0)
                points = np.hstack((points[:,:4], labels.reshape((-1,1)))) #x,y,z,i,label 1-19

        # Crop and extract FOV points
        points = self.crop_pc(points, calib, img_shape)

        points_moco = np.copy(points) #points_weather
        
        weather = 'clear'
        if dataset == 'dense':
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
                points_moco, upsample_applied = nn_upsample(self.cfg, points_moco, self.cluster)

            # Snowfall Augmentation
            snowfall_augmentation_applied = False
            apply_snow = weather == 'clear' and 'SNOW' in self.cfg and dataset == 'dense'
            if apply_snow:
                points_moco, snowfall_augmentation_applied = snow_sim(self.cfg, self.logger, self.rainfall_rates, sample_idx, self.root_dense_path, points_moco, self.cluster)
            
            # Wet Surface Augmentation
            wet_surface_applied = False
            apply_wet_surface = weather == 'clear' and 'WET_SURFACE' in self.cfg and dataset == 'dense'
            if apply_wet_surface:
                points_moco, wet_surface_applied = wet_surface_sim(self.cfg, snowfall_augmentation_applied, points_moco, self.logger)
            
            #Fog augmentation
            fog_applied = False
            apply_fog = weather == 'clear' and not snowfall_augmentation_applied and 'FOG_AUGMENTATION' in self.cfg
            if apply_fog:
                points_moco, fog_applied = fog_sim(self.cfg, points_moco, self.cluster) 

        #print(f'dataset: {dataset}, weather: {weather}, dror: {dror_applied}, up: {upsample_applied}, snow: {snowfall_augmentation_applied}, wet: {wet_surface_applied}, fog: {fog_applied}')

        
        if not self.linear_probe:
            points_moco = self.crop_pc(points_moco, calib, img_shape)
            if self.cluster:
                # SegContrast
                data_dict['data'] = np.hstack((points[:,:4], points[:,-1].reshape((-1,1))))
                data_dict['data_moco'] = np.hstack((points_moco[:,:4], points_moco[:, -1].reshape((-1,1))))  
            else:
                # DepthContrast
                data_dict['data'] = points[:,:4] #x,y,z,i #drop channel or label, add cluster id
                data_dict['data_moco'] = points_moco[:,:4] #x,y,z,i #drop channel or label , add cluster id
        else:
            data_dict['data'] = points


        #V.draw_scenes(points=points, color_feature='intensity')
        #V.draw_scenes(points=points_moco, color_feature='intensity')
        # if fog_applied or snowfall_augmentation_applied:
        # if dataset == 'semantic_kitti' and fog_applied:
        #     visualize_pcd_clusters(points)
        #     visualize_pcd_clusters(points_moco)
        # Prepare points and Transform 
        data_dict = self.prepare_data(data_dict, index)
        # if dataset == 'semantic_kitti' and fog_applied:
        #     visualize_pcd_clusters(np.asarray(data_dict["data"][0]))
        #     visualize_pcd_clusters(np.asarray(data_dict["data_moco"][0]))

        return data_dict