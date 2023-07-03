import copy
import pickle
from pathlib import Path

import numpy as np

from datasets.depth_dataset import DepthContrastDataset

def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


class WaymoDataset(DepthContrastDataset):
    def __init__(self, cfg, cluster, linear_probe=False, mode='train', logger=None):
        super().__init__(cfg, linear_probe=linear_probe, mode=mode, logger=logger)
        self.data_path = self.root_path / 'waymo_processed_data_10'
        self.infos_pkl_path = self.root_path / ('waymo_processed_data_10_infos_%s.pkl' % (self.mode)) #TODO: add DATA_SPLIT in cfg

        self.infos = []
        self.include_waymo_data() # read tfrecords in sample_seq_list and then find its pkl in waymo_processed_data_10 and include the pkl infos in waymo infos

    def include_waymo_data(self):
        self.logger.add_line('Loading Waymo dataset')

        with open(self.infos_pkl_path, 'rb') as f:
            infos = pickle.load(f) # loads all infos
            self.infos.extend(infos[:]) # each info is one frame

        self.logger.add_line('Total samples for Waymo dataset: %d' % (len(self.infos))) # total frames
    
    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all = point_features[:, 0:5] #points_all: x,y,z,i,elongation
        points_all[:, 3] = np.tanh(points_all[:, 3]) * 255.0  #TODO:
        return points_all #only get xyzi

    
    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']


        points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            annos = drop_info_with_name(annos, name='unknown')

            gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict