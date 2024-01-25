import copy
import pickle
import os
from pathlib import Path

import numpy as np
from utils.data_map import *
from datasets.depth_dataset import DepthContrastDataset

class SemanticKittiDataset(DepthContrastDataset):
    def __init__(self, cfg, pretraining=True, mode='train', logger=None):
        super().__init__(cfg, pretraining=pretraining, mode=mode, logger=logger)
        self.data_root_path =  self.root_path / 'data/semantic_kitti' 
        seq = {'train': [ '00', '01', '02', '03', '04', '05', '06', '07', '09', '10' ],
               'val': ['08']}
        self.seq_ids = seq[self.mode]
        self.frame_sampling_interval= cfg["FRAME_SAMPLING_INTERVAL"][self.mode]

        self.points_data_path = []
        self.labels_data_path = []
        self.include_data() 
        
    def include_data(self):
        points_datapath = []
        labels_datapath = []
        self.logger.add_line('Loading Semantic Kitti dataset')

        for seq in self.seq_ids:
            point_seq_path = os.path.join(self.data_root_path, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

            label_seq_path = os.path.join(self.data_root_path, 'dataset', 'sequences', seq, 'labels')
            point_seq_label = os.listdir(label_seq_path)
            point_seq_label.sort()
            labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]

        if self.frame_sampling_interval > 1:
            for k in range(0, len(points_datapath), self.frame_sampling_interval):
                self.points_data_path.append(points_datapath[k])
                self.labels_data_path.append(labels_datapath[k])
        else:
            self.points_data_path = points_datapath
            self.labels_data_path = labels_datapath

        self.logger.add_line(f'Total Semantic Kitti samples loaded: {len(self.points_data_path)} / {len(points_datapath)}')
        b=1
    
    
    def get_lidar(self, sample_path):
        lidar_file = Path(sample_path)
        assert lidar_file.exists(), f'{lidar_file} not found'
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        #TODO: check this if necessary
        # pc[:,3] = np.round(pc[:,3] * 255)
        pc[:,3] = pc[:,3] * 255.0
        #only get xyzi
        return pc 

    def get_seglabels(self, sample_path):
        labels = np.fromfile(sample_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF

        #remap labels to learning values
        labels = np.vectorize(semantic_kitti_labels_map.get)(labels)
        labels = np.expand_dims(labels, axis=-1)
        # unlabeled = labels[:,0] == 0

        # # remove unlabeled points
        # labels = np.delete(labels, unlabeled, axis=0)
        # points = np.delete(points, unlabeled, axis=0)

        
        return labels

    
    def __len__(self):
        return len(self.points_data_path)
    
    def get_item_downstream(self, index):
        points_path = self.points_data_path[index]
        labels_path = self.labels_data_path[index]

        points = self.get_lidar(points_path)
        pt_seg_labels = self.get_seglabels(labels_path)
        assert points.shape[0] == pt_seg_labels.shape[0], f'Missing labels for {labels_path}!!!!!!!!'
        points = np.hstack([points[:,:4], pt_seg_labels.reshape(-1, 1)]) #xyzi, seglabel

        input_dict = {
            'points': points, #xyzi, seglabel
            'frame_id': labels_path
            }

        data_dict = self.prepare_data_downstream(data_dict=input_dict)

        return data_dict

    def __getitem__(self, index):
        data_dict = self.get_item_downstream(index)
        return data_dict