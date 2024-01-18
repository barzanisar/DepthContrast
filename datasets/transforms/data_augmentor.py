from functools import partial

import numpy as np
from third_party.OpenPCDet.pcdet.utils import common_utils
from third_party.OpenPCDet.pcdet.datasets.augmentor import augmentor_utils

def check_aspect2D(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2])/np.max(crop_range[:2])
    return (xy_aspect >= aspect_min)

class DataAugmentor(object):
    def __init__(self, augmentor_configs) -> None:        
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs["POINT_TRANSFORMS"]
        
        for cur_cfg in aug_config_list:
            cur_augmentor = getattr(self, cur_cfg["NAME"])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
    
    def random_cuboid_lidar(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_cuboid_lidar, config=config)
        points = data_dict['points']
        
        # not compatible with voxel wise contrastive loss!
        range_xyz = np.max(points[:,0:2], axis=0) - np.min(points[:,0:2], axis=0)
        crop_range = 0.5 + (np.random.rand(2) * 0.5)
        loop_count = 0
        while not check_aspect2D(crop_range, 0.75):
            loop_count += 1
            crop_range = 0.5 + (np.random.rand(2) * 0.5)
            if loop_count > 100:
                break

        loop_count = 0
        while True:
            loop_count += 1
            
            sample_center = points[np.random.choice(len(points)), 0:3]

            new_range = range_xyz * crop_range / 2.0

            max_xyz = sample_center[0:2] + new_range
            min_xyz = sample_center[0:2] - new_range

            upper_idx = np.sum((points[:,0:2] <= max_xyz).astype(np.int32), 1) == 2
            lower_idx = np.sum((points[:,0:2] >= min_xyz).astype(np.int32), 1) == 2

            new_pointidx = (upper_idx) & (lower_idx)
        
            if (loop_count > 100) or (np.sum(new_pointidx) > float(config['npoints'])):
                break
        
        data_dict["points"] = points[new_pointidx,:]
        return data_dict

    def random_drop_n_cuboids(self, data_dict=None, config=None):
        """ Randomly drop N cuboids from the point cloud.
            Input:
                BxNx3 array, original batch of point clouds
            Return:
                BxNx3 array, dropped batch of point clouds
        """
        if data_dict is None:
            return partial(self.random_drop_n_cuboids, config=config)
        data_dict = self.random_drop(data_dict)
        cuboids_count = 1
        while cuboids_count < 5 and np.random.uniform(0., 1.) > 0.3:
            data_dict = self.random_drop(data_dict)
            cuboids_count += 1

        return data_dict

    def random_drop(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_drop, config=config)
        points = data_dict['points']

        range_xyz = np.max(points[:,0:3], axis=0) - np.min(points[:,0:3], axis=0)

        crop_range = np.random.uniform(0.1, 0.15)
        new_range = range_xyz * crop_range / 2.0
        
        sample_center = points[np.random.choice(len(points)), 0:3]
        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range

        upper_idx = np.sum((points[:,0:3] < max_xyz).astype(np.int32), 1) == 3
        lower_idx = np.sum((points[:,0:3] > min_xyz).astype(np.int32), 1) == 3

        new_pointidx = ~((upper_idx) & (lower_idx))
        data_dict["points"] = points[new_pointidx,:]

        return data_dict
    
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )
        
        data_dict['gt_boxes'] = gt_boxes #TODO: not needed
        data_dict['points'] = points
        return data_dict
    
    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range, data_dict['gt_box_cluster_ids']
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range, data_dict['gt_box_cluster_ids']
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE'], data_dict['gt_box_cluster_ids']
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def forward(self, points, gt_boxes=None, gt_box_cluster_ids=None):
        """
        Args:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
        Returns:
        """
        data_dict = {'points': points, 'gt_boxes': gt_boxes, 'gt_box_cluster_ids': gt_box_cluster_ids}
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        
        if gt_boxes is not None:
            gt_boxes[:, 6] = common_utils.limit_period(
                gt_boxes[:, 6], offset=0.5, period=2 * np.pi
            )#[-2pi=-360deg, 2pi=360 deg] --> [-pi=-180, pi=180]

        return data_dict['points'], data_dict['gt_boxes']