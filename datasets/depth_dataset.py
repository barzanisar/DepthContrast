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

from datasets.transforms.augment3d import get_transform3d

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'OpenPCDet'))
#sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'LiDAR_snow_sim'))
import scipy.stats as stats
import pcl
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from lib.LiDAR_snow_sim.tools.snowfall.simulation import augment
from lib.LiDAR_snow_sim.tools.wet_ground.augmentation import ground_water_augmentation
from lib.LiDAR_snow_sim.tools.snowfall.sampling import snowfall_rate_to_rainfall_rate, compute_occupancy

from lib.LiDAR_fog_sim.fog_simulation import *
from lib.LiDAR_fog_sim.SeeingThroughFog.tools.DatasetFoggification.beta_modification import BetaRadomization
from lib.LiDAR_fog_sim.SeeingThroughFog.tools.DatasetFoggification.lidar_foggification import haze_point_cloud


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
        elif "DenseDataset" in self.dataset_names:
            self.point_cloud_range = DENSE_POINT_RANGE
            self.class_names = ['Car', 'Pedestrian', 'Cyclist'] #, 'LargeVehicle' is now grouped under PassengerCar

        #### Add the voxelizer here
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
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
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

        #Crop given point cloud range
        points = data_dict['data']
        upper_idx = np.sum((points[:, 0:3] <= self.point_cloud_range[3:6]).astype(np.int32), 1) == 3
        lower_idx = np.sum((points[:, 0:3] >= self.point_cloud_range[0:3]).astype(np.int32), 1) == 3

        new_pointidx = (upper_idx) & (lower_idx)
        points = points[new_pointidx, :]

        #Crop given point cloud range
        points_weather = data_dict['data_moco']
        upper_idx = np.sum((points_weather[:, 0:3] <= self.point_cloud_range[3:6]).astype(np.int32), 1) == 3
        lower_idx = np.sum((points_weather[:, 0:3] >= self.point_cloud_range[0:3]).astype(np.int32), 1) == 3

        new_pointidx = (upper_idx) & (lower_idx)
        points_weather = points_weather[new_pointidx, :]


        cfg = self.cfg
        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        if cfg["DATA_TYPE"] == "point_vox":
            # Across format
            item = {"data": [], "data_aug_matrix": [], 
            "data_moco": [], "data_moco_aug_matrix": [], 
            "vox": [], "vox_aug_matrix": [], 
            "vox_moco": [], "vox_moco_aug_matrix": []}

            item["data"].append(points)
            item["data_moco"].append(points_weather)
            item["vox"].append(np.copy(points))
            item["vox_moco"].append(np.copy(points_weather))

            #item["data_valid"].append(1)
        else:
            # Within format: data is either points or later is voxelized
            item = {"data": [], "data_aug_matrix": [], 
            "data_moco": [], "data_moco_aug_matrix": []}

            item["data"].append(points)
            item["data_moco"].append(points_weather)

        # Apply the transformation here
        if (cfg["DATA_TYPE"] == "point_vox"):
            # Points
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data"] = tempdata["data"]
            item["data_aug_matrix"] = tempdata['aug_trans_matrix']

            # Points MoCo
            tempitem = {"data": item["data_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data_moco"] = tempdata["data"]
            item["data_moco_aug_matrix"] = tempdata['aug_trans_matrix']

            # Vox
            tempitem = {"data": item["vox"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:, :3]
            feats = tempdata["data"][0][:, 3:6] * 255.0  # np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox"] = [self.toVox(coords, feats, labels)]
            item["vox_aug_matrix"] = tempdata['aug_trans_matrix']

            # Vox MoCo
            tempitem = {"data": item["vox_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:, :3]
            feats = tempdata["data"][0][:, 3:6] * 255.0  # np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox_moco"] = [self.toVox(coords, feats, labels)]
            item["vox_moco_matrix"] = tempdata['aug_trans_matrix']
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
            item["data_aug_matrix"] = tempdata['aug_trans_matrix']
            
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
            item["data_moco_aug_matrix"] = tempdata['aug_trans_matrix']
            
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


class DenseDataset(DepthContrastDataset):

    def __init__(self, cfg, linear_probe=False, mode='train', logger=None):
        super().__init__(cfg, linear_probe=linear_probe, mode=mode, logger=logger)
        self.root_data_path = self.root_path / 'data' / 'dense'  # DepthContrast/data/waymo
        
        self.sensor_type = self.cfg["SENSOR_TYPE"]
        self.signal_type = self.cfg["SIGNAL_TYPE"]
        self.lidar_folder = f'lidar_{self.sensor_type}_{self.signal_type}'
        self.calib = self.get_calib(self.sensor_type)

        self.dense_infos = []
        self.include_dense_data()

        #SnowFall augmentation
        self.snowfall_rates = [0.5, 0.5, 1.0, 2.0, 2.5, 1.5]      # mm/h
        self.terminal_velocities = [2.0, 1.2, 1.6, 2.0, 1.6, 0.6] # m/s

        self.rainfall_rates = []
        #self.occupancy_ratios = []

        for i in range(len(self.snowfall_rates)):

            self.rainfall_rates.append(snowfall_rate_to_rainfall_rate(self.snowfall_rates[i],
                                                                      self.terminal_velocities[i]))

            #self.occupancy_ratios.append(compute_occupancy(self.snowfall_rates[i], self.terminal_velocities[i]))

        #self.combos = np.column_stack((self.rainfall_rates, self.occupancy_ratios))


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
    
    def get_calib(self, sensor: str = 'hdl64'):
        calib_file = self.root_data_path / f'calib_{sensor}.txt'
        assert calib_file.exists(), f'{calib_file} not found'
        return calibration_kitti.Calibration(calib_file)

    def get_lidar(self, idx):
        lidar_file = self.root_data_path / self.lidar_folder / ('%s.bin' % idx)
        assert lidar_file.exists(), f'{lidar_file} not found'
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

    def __len__(self):
        return len(self.dense_infos)

    def foggify(self, points, sample_idx, alpha, augmentation_method, curriculum_stage, on_the_fly=False):

        if augmentation_method == 'DENSE' and alpha != '0.000' and not on_the_fly:          # load from disk

            curriculum_folder = f'{self.lidar_folder}_{augmentation_method}_beta_{alpha}'

            lidar_file = self.root_data_path / 'fog_simulation' / curriculum_folder / ('%s.bin' % sample_idx)
            assert lidar_file.exists(), f'could not find {lidar_file}'
            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

        if augmentation_method == 'DENSE' and alpha != '0.000' and on_the_fly:

            B = BetaRadomization(beta=float(alpha), seed=0)
            B.propagate_in_time(10)

            arguments = Namespace(sensor_type='Velodyne HDL-64E S3D', fraction_random=0.05)
            n_features = points.shape[1]
            points = haze_point_cloud(points, B, arguments)
            points = points[:, :n_features]

        if augmentation_method == 'CVL' and alpha != '0.000':

            p = ParameterSet(alpha=float(alpha), gamma=0.000001)

            gain = self.cfg.get('FOG_GAIN', False)
            fog_noise_variant = self.cfg.get('FOG_NOISE_VARIANT', 'v1')
            soft = self.cfg.get('FOG_SOFT', True)
            hard = self.cfg.get('FOG_HARD', True)

            points, _, _ = simulate_fog(p, pc=points, noise=10, gain=gain, noise_variant=fog_noise_variant,
                                        soft=soft, hard=hard)

        #self.curriculum_stage = curriculum_stage
        #self.current_iteration += self.iteration_increment

        return points

    # adapted from https://github.com/mpitropov/cadc_devkit/blob/master/other/filter_pointcloud.py#L13-L50
    def dynamic_radius_outlier_filter(self, pc: np.ndarray, alpha: float = 0.16, beta: float = 3.0,
                                    k_min: int = 3, sr_min: float = 0.04) -> np.ndarray:
        """
        :param pc:      pointcloud
        :param alpha:   horizontal angular resolution of the lidar
        :param beta:    multiplication factor
        :param k_min:   minimum number of neighbors
        :param sr_min:  minumum search radius

        :return:        mask [False = snow, True = no snow]
        """

        pc = pcl.PointCloud(pc[:, :3])

        num_points = pc.size

        # initialize mask with False
        mask = np.zeros(num_points, dtype=bool)

        k = k_min + 1

        kd_tree = pc.make_kdtree_flann()

        for i in range(num_points):

            x = pc[i][0]
            y = pc[i][1]

            r = np.linalg.norm([x, y], axis=0)

            sr = alpha * beta * np.pi / 180 * r

            if sr < sr_min:
                sr = sr_min

            [_, sqdist] = kd_tree.nearest_k_search_for_point(pc, i, k)

            neighbors = -1      # start at -1 since it will always be its own neighbour

            for val in sqdist:
                if np.sqrt(val) < sr:
                    neighbors += 1

            if neighbors >= k_min:
                mask[i] = True  # no snow -> keep

        return mask

    def __getitem__(self, index):
        
        # TODO: remove
        index = 1259
        info = copy.deepcopy(self.dense_infos[index])
        sample_idx = '2018-02-16_17-35-08_00130' #info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        points_weather = np.copy(points)
        
        # # TODO: remove
        # for i, val in enumerate(self.dense_infos):
        #     if val['point_cloud']['lidar_idx'] == sample_idx:
        #         info = copy.deepcopy(self.dense_infos[i])
        #         break

        
        data_dict = {}
        apply_dror = info['annos']['weather'] != 'clear' and not self.linear_probe and 'DROR' in self.cfg
        if apply_dror:

            alpha = self.cfg['DROR']

            dror_path = self.root_data_path / 'DROR' / f'alpha_{alpha}' / \
                        'all' / self.sensor_type / self.signal_type / 'full' / f'{sample_idx}.pkl'

            with open(str(dror_path), 'rb') as f:
                snow_indices = pickle.load(f)

            keep_indices = np.ones(len(points), dtype=bool)
            keep_indices[snow_indices] = False

            points_weather = points[keep_indices]

        # Extract FOV points
        if self.cfg['FOV_POINTS_ONLY']:
            pts_rect = self.calib.lidar_to_rect(points[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, (1024, 1920), self.calib)
            points = points[fov_flag]

            pts_rect = self.calib.lidar_to_rect(points_weather[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, (1024, 1920), self.calib)
            points_weather = points_weather[fov_flag]

        
        # Snowfall Augmentation
        snowfall_augmentation_applied = False
        apply_snow = False #info['annos']['weather'] == 'clear' and not self.linear_probe and 'SNOW' in self.cfg
        if apply_snow:
            parameters = self.cfg['SNOW'].split('_')

            sampling = parameters[0]        # e.g. uniform
            mode = parameters[1]            # gunn or sekhon
            chance = parameters[2]          # e.g. 8in9

            choices = [0]

            if chance == '8in9':
                choices = [1, 1, 1, 1, 1, 1, 1, 1, 0]
            elif chance == '4in5':
                choices = [1, 1, 1, 1, 0]
            elif chance == '1in2':
                choices = [1, 0]
            elif chance == '1in1':
                choices = [1]
            elif chance == '1in4':
                choices = [1, 0, 0, 0]
            elif chance == '1in10': #recommended by lidar snow sim paper
                choices = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            if np.random.choice(choices): 

                rainfall_rate = 0

                if sampling == 'uniform':
                    rainfall_rate = int(np.random.choice(self.rainfall_rates))

                lidar_file = self.root_data_path / 'snowfall_simulation' / mode / \
                             f'{self.lidar_folder}_rainrate_{rainfall_rate}' / f'{sample_idx}.bin'

                try:
                    points_weather = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)
                    snowfall_augmentation_applied = True
                except FileNotFoundError:
                    print(f'\n{lidar_file} not found')
                    pass
        
        wet_surface_applied = False
        apply_wet_surface = info['annos']['weather'] == 'clear' and not self.linear_probe and 'WET_SURFACE' in self.cfg
        if apply_wet_surface:

            method = self.cfg['WET_SURFACE']

            choices = [0]

            if '1in2' in method:
                choices = [0, 1]                            # pointcloud gets augmented with 50% chance
            
            elif '1in1' in method:
                choices = [1]

            elif '1in4' in method:
                choices = [0, 0, 0, 1]                      # pointcloud gets augmented with 25% chance

            elif '1in10' in method:
                choices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]    # pointcloud gets augmented with 10% chance

            apply_coupled = 'COUPLED' in self.cfg and snowfall_augmentation_applied

            if 'COUPLED' in self.cfg:
                choices = [0]                   # make sure we only apply coupled when coupled is enabled

            if np.random.choice(choices) or apply_coupled:

                if 'norm' in method:

                    lower, upper = 0.05, 0.5
                    mu, sigma = 0.2, 0.1
                    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

                    water_height = X.rvs(1)

                else:

                    elements = np.linspace(0.1, 1.2, 12)
                    probabilities = 5 * np.ones_like(elements)  # each element initially 5% chance

                    probabilities[0] = 15   # 0.1
                    probabilities[1] = 25   # 0.2
                    probabilities[2] = 15   # 0.3

                    probabilities = probabilities / 100

                    water_height = np.random.choice(elements, 1, p=probabilities)

                try:
                    points_weather = ground_water_augmentation(points_weather, water_height=water_height, debug=False)
                    wet_surface_applied = True
                except (TypeError, ValueError):
                    pass
        
        #Fog augmentation
        apply_fog = info['annos']['weather'] == 'clear' and not snowfall_augmentation_applied and not self.linear_probe and 'FOG_AUGMENTATION' in self.cfg
        if apply_fog:
            alphas = ['0.005', '0.010', '0.020', '0.030', '0.060']
            curriculum_stage = int(np.random.randint(low=0, high=len(alphas)))
            alpha = alphas[curriculum_stage]

            if alpha == '0.000':    # to prevent division by zero
                mor = np.inf
            else:
                mor = np.log(20) / float(alpha)

            augmentation_method = self.cfg['FOG_AUGMENTATION'].split('_')[0]
            #augmentation_schedule = self.cfg['FOG_AUGMENTATION'].split('_')[-1]

            points_weather = self.foggify(points_weather, sample_idx, alpha, augmentation_method, curriculum_stage)

        
        # Add gt_boxes for linear probing
        if self.linear_probe:
            data_dict['gt_names'] = info['annos']['name']
            data_dict['gt_boxes_lidar'] = info['annos']['gt_boxes_lidar']
            assert data_dict['gt_names'].shape[0] == data_dict['gt_boxes_lidar'].shape[0]

            limit_by_mor = self.cfg.get('LIMIT_BY_MOR', False)

            if limit_by_mor:
                distances = np.linalg.norm(data_dict['gt_boxes_lidar'][:, 0:3], axis=1)
                mor_mask = distances < mor

                data_dict['gt_names'] = data_dict['gt_names'][mor_mask]
                data_dict['gt_boxes_lidar'] = data_dict['gt_boxes_lidar'][mor_mask]
            
            filter_out_of_mor_boxes = self.cfg.get('FILTER_OUT_OF_MOR_BOXES', False)

            # filter out empty bounding boxes that are outside of MOR
            if filter_out_of_mor_boxes:

                max_point_dist = max(np.linalg.norm(points[:, 0:3], axis=1))
                box_distances = np.linalg.norm(data_dict['gt_boxes_lidar'][:, 0:3], axis=1)

                box_mask = box_distances < max_point_dist
                data_dict['gt_boxes_lidar'] = data_dict['gt_boxes_lidar'][box_mask]

            # TODO: mask boxes outside point cloud range
            
            # # TODO: remove
            # # Drop gt_names == DontCare, gt_boxes already don't have dontcare boxes
            # keep_indices = [i for i,x in enumerate(info['annos']['name']) if x != 'DontCare']
            # data_dict['gt_names'] = info['annos']['name'][keep_indices]
            # data_dict['gt_boxes_lidar'] = info['annos']['gt_boxes_lidar']
            
            # # Drop gt_names and boxes with negative h,w,l i.e. not visible in lidar
            # keep_indices = [i for i in range(data_dict['gt_boxes_lidar'].shape[0]) if data_dict['gt_boxes_lidar'][i, 3] > 0]
            # data_dict['gt_names'] = data_dict['gt_names'][keep_indices]
            # data_dict['gt_boxes_lidar'] = data_dict['gt_boxes_lidar'][keep_indices]

            # assert data_dict['gt_names'].shape[0] == data_dict['gt_boxes_lidar'].shape[0]
            
            # # # what happens if gt_boxes_lidar is empty?
            # # if data_dict['gt_boxes_lidar'].shape[0] == 0:
            # #     if self.logger is not None:
            # #         self.logger.add_line(f'No gt_boxes_lidar in infos Index: {index}, sample_idx: {sample_idx}!')
            # #     new_index = np.random.randint(self.__len__())
            # #     return self.__getitem__(new_index)
            
            # # assert data_dict['gt_boxes_lidar'].shape[0] > 0

            # # TODO: remove
            # # Change Vehicle or Obstacle class to PassengerCar
            # for i, name in enumerate(data_dict['gt_names']):
            #     if name in ['Vehicle', 'Obstacle', 'LargeVehicle']:
            #         data_dict['gt_names'][i] = 'PassengerCar'

        # Prepare points and Transform 
        data_dict['data'] = points[:,:4] #x,y,z,i #drop channel or label
        data_dict['data_moco'] = points_weather[:,:4] #x,y,z,i #drop channel or label 

        V.draw_scenes(points=points, color_feature='intensity')
        V.draw_scenes(points=points_weather, color_feature='intensity')
        data_dict = self.prepare_data(data_dict, index)

        return data_dict
