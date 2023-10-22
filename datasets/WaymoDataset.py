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
        self.data_root_path =  self.root_path / cfg["DATA_PATH"]
        self.lidar_data_path = self.data_root_path / cfg.PROCESSED_DATA_TAG 
        #self.infos_pkl_path = self.data_root_path / f'{cfg.PROCESSED_DATA_TAG}_infos_{cfg.DATA_SPLIT[self.mode]}.pkl'
        self.split = cfg.DATA_SPLIT[self.mode]
        self.cluster_root_path = self.root_path / 'output' / f'{cfg.PROCESSED_DATA_TAG}_clustered'
        self.mean_box_sizes = cfg.get('MEAN_SIZES', None)

        self.infos = []
        self.include_waymo_data() # read tfrecords in sample_seq_list and then find its pkl in waymo_processed_data_10 and include the pkl infos in waymo infos
        
        if self.mean_box_sizes is not None:
            self.mean_box_sizes = np.array(self.mean_box_sizes)
            self.class_size_cnts = self.add_pseudo_classes()
            self.logger.add_line('Pseudo Class Counts:')
            for i, cls in enumerate(self.class_names):
                self.logger.add_line(f'{cls}: {self.class_size_cnts[i]}')

    def include_waymo_data(self):
        self.logger.add_line('Loading Waymo dataset')
        split_txt_file = self.data_root_path / 'ImageSets' / (self.split + '.txt')
        seq_list = [x.strip().split('.')[0] for x in open(split_txt_file).readlines()]

        num_skipped_infos=0
        for seq_name in seq_list:
            seq_info_path = self.cluster_root_path / seq_name /  'approx_boxes.pkl' #('%s.pkl' % seq_name)
            if not seq_info_path.exists():
                num_skipped_infos += 1
                continue
            with open(seq_info_path, 'rb') as f:
                infos = pickle.load(f) # loads 20 infos for one seq pkl i.e. 20 frames if seq pkl was formed by sampling every 10th frame
                self.infos.extend(infos) # each info is one frame


            #self.infos = self.infos[:32] # each info is one frame

        self.logger.add_line('Total frames for Waymo dataset: %d' % (len(self.infos))) # total frames
        self.logger.add_line('Total skipped sequences due to missing info pkls %s' % num_skipped_infos)
    
    def add_pseudo_classes(self):
        class_size_cnts=np.zeros(len(self.class_names))
        for info in self.infos:
            gt_boxes = info['approx_boxes_closeness_to_edge']
            
            lwh = gt_boxes[:, 3:6]
            l = np.max(lwh[:,:2], axis=1)
            w = np.min(lwh[:,:2], axis=1)
            lwh[:,0] = l
            lwh[:,1] = w
            
            dist = (((self.mean_box_sizes.reshape(1, -1, 3) - \
            lwh.reshape(-1, 1, 3)) ** 2).sum(axis=2))  # N=boxes x M=mean sizes 
            idx_matched_mean_sizes = dist.argmin(axis=1) # N gt boxes
            gt_names = np.array(self.class_names)[idx_matched_mean_sizes]
            info['approx_boxes_names'] = gt_names
            unique_idx, counts = np.unique(idx_matched_mean_sizes, return_counts=True)
            class_size_cnts[unique_idx] += counts
        
        return class_size_cnts



    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.lidar_data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all = point_features[:, 0:4] #points_all: x,y,z,i, skip elongation
        points_all[:, 3] = np.tanh(points_all[:, 3]) * 255.0  #TODO:
        return points_all #only get xyzi

    def get_cluster_labels(self, sequence_name, sample_idx):
        label_file = self.cluster_root_path / sequence_name / ('%04d.npy' % sample_idx)
        labels = np.fromfile(label_file, dtype=np.float16)
        return labels

    
    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        frame_id = info['frame_id']

        # print('frame_id: ', info['frame_id'])

        points = self.get_lidar(sequence_name, sample_idx)
        pt_cluster_labels = self.get_cluster_labels(sequence_name, sample_idx)

        assert points.shape[0] == pt_cluster_labels.shape[0], f'Missing labels for {frame_id}!!!!!!!!'

        points = np.hstack([points, pt_cluster_labels.reshape(-1, 1)]) #xyzi
        gt_classes = np.array([self.class_names.index(n) + 1 for n in info['approx_boxes_names']], dtype=np.int32) # 1: Vehicle, 2: Ped, 3: Cycl, 4: OtherSmall...

        #append class id as 8th entry in gt boxes and cluster label as 9th
        #gt_boxes = np.concatenate((data_dict['approx_boxes_closeness_to_edge'][:,:7], gt_classes.reshape(-1, 1).astype(np.float32), info['cluster_labels_boxes'].reshape(-1, 1)), axis=1)
        gt_boxes = np.hstack([info['approx_boxes_closeness_to_edge'][:,:7], gt_classes.reshape(-1, 1).astype(np.float32), info['cluster_labels_boxes'].reshape(-1, 1)])
        
        input_dict = {
            'points': points,
            # 'frame_id': info['frame_id'],
            'gt_boxes':  gt_boxes
            # 'gt_names': info['approx_boxes_names']
            }

        # if 'annos' in info:
        #     annos = info['annos']
        #     annos = drop_info_with_name(annos, name='unknown')

        #     gt_boxes_lidar = annos['gt_boxes_lidar'] #TODO: approx boxes

        #     if self.cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
        #         mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
        #         annos['name'] = annos['name'][mask]  #TODO: approx boxes
        #         gt_boxes_lidar = gt_boxes_lidar[mask]
        #         annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

        #     input_dict.update({
        #         'gt_names': annos['name'],
        #         'gt_boxes': gt_boxes_lidar,
        #         'num_points_in_gt': annos.get('num_points_in_gt', None)
        #     })

        data_dict = self.prepare_data(data_dict=input_dict)
        #data_dict['metadata'] = info.get('metadata', info['frame_id'])
        #data_dict.pop('num_points_in_gt', None)
        return data_dict