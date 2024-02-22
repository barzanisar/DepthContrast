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

def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

class WaymoDataset(DepthContrastDataset):
    def __init__(self, cfg, pretraining=True, mode='train', logger=None):
        super().__init__(cfg, pretraining=pretraining, mode=mode, logger=logger)
        self.data_root_path =  self.root_path / cfg["DATA_PATH"] #root_path: DepthContrast, DATA_PATH: 'data/waymo'
        self.lidar_data_path = self.data_root_path / cfg.PROCESSED_DATA_TAG 
        self.cluster_root_path = self.data_root_path / f'{cfg.PROCESSED_DATA_TAG}_clustered'
        self.seglabels_root_path = self.data_root_path/ f'{cfg.PROCESSED_DATA_TAG}_labels'
        self.split = cfg.DATA_SPLIT[self.mode]

        self.mean_box_sizes = cfg.get('MEAN_SIZES', None)
        self.frame_sampling_interval= cfg["FRAME_SAMPLING_INTERVAL"][self.mode]

        self.infos = []
        # read tfrecords in sample_seq_list and then find its pkl in waymo_processed_data_10 and include the pkl infos in waymo infos
        self.include_waymo_data() 
        
        if self.pretraining and self.mean_box_sizes is not None:
            self.mean_box_sizes = np.array(self.mean_box_sizes)
            self.distance_thresh = cfg.get("DIST_THRESH", None)
            self.class_size_cnts = self.add_pseudo_classes()
            self.logger.add_line('Pseudo Class Counts:')
            for i, cls in enumerate(self.class_names):
                self.logger.add_line(f'{cls}: {self.class_size_cnts[i]}')

    def include_waymo_data(self):
        self.logger.add_line('Loading Waymo dataset')
        split_txt_file = self.data_root_path / 'ImageSets' / (self.split + '.txt')
        seq_list = [x.strip().split('.')[0] for x in open(split_txt_file).readlines()]

        waymo_infos=[]
        num_skipped_infos=0
        for seq_name in seq_list:
            if self.pretraining:
                seq_info_path = self.cluster_root_path / seq_name /  'approx_boxes.pkl' #('%s.pkl' % seq_name)
            else:
                seq_info_path = self.seglabels_root_path / seq_name /  ('%s.pkl' % seq_name)

            if not seq_info_path.exists():
                num_skipped_infos += 1
                continue
            with open(seq_info_path, 'rb') as f:
                infos = pickle.load(f) # loads 20 infos for one seq pkl i.e. 20 frames if seq pkl was formed by sampling every 10th frame
                # waymo_infos.extend(infos) # each info is one frame
            
            if self.pretraining and (self.use_gt_seg_labels or self.use_gt_dataset):
                with open(self.seglabels_root_path / seq_name /  ('%s.pkl' % seq_name), 'rb') as f:
                    seg_seq_infos = pickle.load(f) # loads 20 infos for one seq pkl i.e. 20 frames if seq pkl was formed by sampling every 10th frame
                sample_idx_with_seg_labels = [info['point_cloud']['sample_idx'] for info in seg_seq_infos]

                for info in infos:
                    if info['point_cloud']['sample_idx'] in sample_idx_with_seg_labels:
                        waymo_infos.append(info) # each info is one frame
            else:
                waymo_infos.extend(infos)


        self.infos.extend(waymo_infos[:])
        self.logger.add_line('Total skipped sequences due to missing info pkls %s' % num_skipped_infos)
        self.logger.add_line('Total samples(frames) for Waymo dataset: %d' % (len(waymo_infos)))

        if self.frame_sampling_interval > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.frame_sampling_interval):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.add_line('Total sampled samples(frames) for Waymo dataset: %d' % len(self.infos))


        #self.infos = self.infos[:32] # each info is one frame

    
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

            if self.distance_thresh is not None:
                valid_match_mask = dist.min(axis=1) < self.distance_thresh
                info['approx_boxes_closeness_to_edge'] = gt_boxes[valid_match_mask]
                info['approx_boxes_names'] = gt_names[valid_match_mask]
                info['cluster_labels_boxes'] = info['cluster_labels_boxes'][valid_match_mask]
            else:            
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
    
    def get_box_gt_seglabels(self, cluster_labels, seg_labels, info):
        class_ids=np.zeros(len(info['cluster_labels_boxes']))
        for i, lbl in enumerate(info['cluster_labels_boxes']):
            gt_pt_labels = seg_labels[cluster_labels==lbl]
            gt_labels, cnts = np.unique(gt_pt_labels, return_counts=True)
            majority_label = gt_labels[np.argmax(cnts)]
            class_ids[i] = majority_label
        
        return class_ids


    def get_seglabels(self, sequence_name, sample_idx, num_points):
        label_file = self.seglabels_root_path / sequence_name / ('%04d.npy' % sample_idx)
        # labels = np.fromfile(label_file, dtype=np.float16)
        # if labels.shape[0] == num_points and labels.max() < 23:
        #     return labels
        # else:
        labels = np.load(label_file)
        # if labels.shape[0] != num_points:
        #     print(label_file, f'lbls: {labels.shape[0]}, numpts: {num_points}')
        # if labels.max() >= 23:
        #     print(label_file)
        assert labels.shape[0] == num_points, label_file
        assert labels.max() < 23, label_file
        
        return labels

    
    def __len__(self):
        return len(self.infos)
    
    def get_item_downstream(self, index):
        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        frame_id = info['frame_id']

        # print('frame_id: ', info['frame_id'])

        points = self.get_lidar(sequence_name, sample_idx)
        pt_seg_labels = self.get_seglabels(sequence_name, sample_idx, points.shape[0])
        assert points.shape[0] == pt_seg_labels.shape[0], f'Missing labels for {frame_id}!!!!!!!!'
        points = np.hstack([points, pt_seg_labels.reshape(-1, 1)]) #xyzi, seglabel


        # annos = info['annos']
        # #filter unknown boxes
        # annos = drop_info_with_name(annos, name='unknown')
        # # filter empty boxes
        # mask = (annos['num_points_in_gt'] > 0) 
        # annos['name'] = annos['name'][mask]
        # annos['gt_boxes_lidar'] = annos['gt_boxes_lidar'][mask]
        # annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

        # #filer gt boxes not in self.class_names
        # selected = keep_arrays_by_name(annos['name'], self.class_names)
        # annos['name'] = annos['name'][selected]
        # annos['gt_boxes_lidar'] = annos['gt_boxes_lidar'][selected]
        # annos['num_points_in_gt'] = annos['num_points_in_gt'][selected]


        # gt_classes = np.array([self.class_names.index(n) + 1 for n in annos['name']], dtype=np.int32) # 1: Vehicle, 2: Ped, 3: Cycl, 4: OtherSmall...
        
        # #append class id as 8th entry in gt boxes 
        # gt_boxes = np.hstack([annos['gt_boxes_lidar'][:,:7], gt_classes.reshape(-1, 1).astype(np.float32)])
        

        # input_dict = {
        #     'points': points, #xyzi, seglabel
        #     'gt_boxes':  gt_boxes, #gtbox, class_indx
        #     'frame_id': frame_id
        #     }

        input_dict = {
            'points': points, #xyzi, seglabel
            'frame_id': frame_id
            }

        data_dict = self.prepare_data_downstream(data_dict=input_dict)

        return data_dict

    def get_item_pretrain(self, index):
        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        frame_id = info['frame_id']

        # print('frame_id: ', info['frame_id'])

        points = self.get_lidar(sequence_name, sample_idx)
        pt_cluster_labels = self.get_cluster_labels(sequence_name, sample_idx)
        assert points.shape[0] == pt_cluster_labels.shape[0], f'Missing cluster labels for {frame_id}!!!!!!!!'
        gt_cluster_ids =  info['cluster_labels_boxes']

        if self.use_gt_seg_labels:
            pt_seg_labels = self.get_seglabels(sequence_name, sample_idx, points.shape[0])
            assert points.shape[0] == pt_seg_labels.shape[0], f'Missing gt seg labels for {frame_id}!!!!!!!!'
            gt_classes = self.get_box_gt_seglabels(pt_cluster_labels, pt_seg_labels, info)
        elif self.mean_box_sizes is not None:
            gt_classes = np.array([self.class_names.index(n) + 1 for n in info['approx_boxes_names']], dtype=np.int32) # 1: Vehicle, 2: Ped, 3: Cycl, 4: OtherSmall...
        else:
            gt_classes = np.array([1]*gt_cluster_ids.shape[0])
        #append class id as 8th entry in gt boxes and cluster label as 9th
        gt_boxes = np.hstack([info['approx_boxes_closeness_to_edge'][:,:7], gt_classes.reshape(-1, 1).astype(np.float32), gt_cluster_ids.reshape(-1, 1)])
        
        # Set clusters as background if their groundtruth box is not available
        for lbl in np.unique(pt_cluster_labels):
            if lbl not in gt_cluster_ids:
                pt_cluster_labels[pt_cluster_labels==lbl] = -1

        points = np.hstack([points, pt_cluster_labels.reshape(-1, 1)]) #xyzil

        input_dict = {
            'points': points,
            'gt_boxes':  gt_boxes,
            'frame_id': frame_id
            }
        # cluster_ids, cnts = np.unique(input_dict['points'][:,-1], return_counts=True)
        # for cluster_id, cnt in zip(cluster_ids, cnts):
        #     if cluster_id == -1:
        #         continue
        #     frame_id = input_dict['frame_id']
        #     assert cluster_id in input_dict['gt_boxes'][:,-1], f'{frame_id}, cluster_label: {cluster_id}, cnts:{cnt}'

        data_dict = self.prepare_data_pretrain(data_dict=input_dict)

        return data_dict

    def __getitem__(self, index):
        if self.pretraining:
            data_dict = self.get_item_pretrain(index)
        else:
            data_dict = self.get_item_downstream(index)

        return data_dict