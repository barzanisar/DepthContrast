import argparse

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils.approx_bbox_utils import *
from utils.cluster_utils import *
from third_party.OpenPCDet.pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from third_party.OpenPCDet.pcdet.ops.iou3d_nms import iou3d_nms_utils
from utils.tracker import PubTracker as Tracker
from utils.estimate_ground import estimate_ground
from third_party.OpenPCDet.pcdet.utils import box_utils
import multiprocessing as mp
from functools import partial
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description='Cluster Waymo')
parser.add_argument('--mode', type=str, default='simple_cluster', help='simple_cluster or cluster_tracking (does not work atm)')
parser.add_argument('--split', type=str, default='train_short', help='specify the split of infos to cluster')
parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_10_short', help='specify the processed data tag')
parser.add_argument('--save_rejection_tag', action='store_true', default=False, help='if you want to save rejection tags for debugging')



# import open3d as o3d
import os
import torch
import glob
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V

np.random.seed(100)

class WaymoDataset():
    def __init__(self, split, processed_data_tag):
        parent_dir = (Path(__file__) / '../..').resolve() #DepthContrast
        self.root_path = parent_dir / 'data/waymo'
        self.processed_data_tag=processed_data_tag #'waymo_processed_data_10_short'
        self.split = split
        self.data_path = self.root_path / self.processed_data_tag
        self.infos_pkl_path = self.root_path / f'{self.processed_data_tag}_infos_{self.split}.pkl'

        self.save_label_path = parent_dir / 'output' / (self.processed_data_tag + '_clustered')

        self.infos_dict = {} # seq_name: [frame infos]
        self.include_waymo_data() # read tfrecords in sample_seq_list and then find its pkl in waymo_processed_data_10 and include the pkl infos in waymo infos

    def include_waymo_data(self):
        infos=[]
        with open(self.infos_pkl_path, 'rb') as f:
            infos = pickle.load(f) # loads all infos

        for info in infos:
            sequence_name = info['point_cloud']['lidar_sequence']
            if sequence_name not in self.infos_dict:
                self.infos_dict[sequence_name] = []
                #print(sequence_name)
            self.infos_dict[sequence_name].append(info)
    
    def get_lidar(self, seq_name, sample_idx):
        lidar_file = self.data_path / seq_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]
        point_features[:,3] = np.tanh(point_features[:, 3]) * 255.0
        return point_features[:,:4] #only get xyzi
    
    def get_cluster_labels(self, seq_name, sample_idx):
        label_file = self.save_label_path / seq_name / ('%04d.npy' % sample_idx)
        labels = np.fromfile(label_file, dtype=np.float16)
        return labels
    
    def get_ground_mask(self, seq_name, sample_idx):
        path = self.save_label_path / seq_name / 'ground' /('%04d.npy' % sample_idx)
        ground_mask = np.fromfile(path, dtype=np.bool_)
        return ground_mask
    
    def get_rejection_tag(self, seq_name, sample_idx):
        path = self.save_label_path / seq_name / 'rejection_tag' /('%04d.npy' % sample_idx)
        pt_wise_rejection_tag = np.fromfile(path, dtype=np.uint8)
        return pt_wise_rejection_tag
    
    
    def estimate_ground_seq(self, seq_name):
        save_seq_path = self.save_label_path / seq_name / 'ground'
        os.makedirs(save_seq_path.__str__(), exist_ok=True)
        infos = self.infos_dict[seq_name]
        for info in infos:
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']
            save_path = save_seq_path / ('%04d.npy' % sample_idx)
            if save_path.exists():
                print(f'Ground exists for sample idx: {sample_idx}')
                continue

            #points in current vehicle frame
            xyzi = self.get_lidar(seq_name, sample_idx)
            
            ground_mask = estimate_ground(xyzi)
            
            ground_mask.tofile(save_path)
        
        print('Estimating ground Done!')

def transform_pc_to_world(xyz, pose_world_from_vehicle):
    xyz = np.concatenate([xyz, np.ones([xyz.shape[0], 1])], axis=-1) #(N, xyz1)
    xyz = np.matmul(pose_world_from_vehicle, xyz.T).T[:,:3] #(N, xyz)

    return xyz

def simple_cluster(seq_name, dataset, show_plots=False, save_rejection_tag=False):
    save_seq_path = dataset.save_label_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    os.makedirs((save_seq_path / 'rejection_tag').__str__(), exist_ok=True)
    print(f'Clustering of sequence: {seq_name} started!')
    
    for i, info in enumerate(dataset.infos_dict[seq_name]):
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']
        save_path = save_seq_path / ('%04d.npy' % sample_idx)
        if save_path.exists():
            continue
        
        xyzi = dataset.get_lidar(seq_name, sample_idx)
        num_pts = xyzi.shape[0]
        ground_mask = dataset.get_ground_mask(seq_name, sample_idx)
        
        # Get new labels
        labels = cluster(xyzi[:,:3], np.logical_not(ground_mask), eps=0.2)
        assert labels.shape[0] == num_pts, f'After Clustering: Some labels missing for seq: {seq_name}, sample {sample_idx}!'
        print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')
        if show_plots:
            visualize_pcd_clusters(xyzi[:,:3], labels.reshape((-1,1)))


        new_labels, label_wise_rejection_tag  = filter_labels(xyzi[:,:3], labels,
                                    max_volume=None, min_volume=0.1, 
                                    max_height_for_lowest_point=1, 
                                    min_height_for_highest_point=0.5,
                                    ground_mask = ground_mask)
        
        assert new_labels.shape[0] == num_pts, f'After filtering: Some labels missing for seq: {seq_name}, sample {sample_idx}!'

        if show_plots:
            print(f'After filtering')
            visualize_pcd_clusters(xyzi[:,:3], new_labels.reshape((-1,1)))

        if save_rejection_tag:
            pt_wise_rejection_tag = np.zeros(xyzi.shape[0], dtype=np.uint8) #zero means not rejected
            for i in np.unique(labels):
                if i == -1:
                    continue
                pt_wise_rejection_tag[labels == i] = label_wise_rejection_tag[int(i)]
        
            if show_plots:
                for key, val in REJECT.items():
                    rejected_labels = np.where(label_wise_rejection_tag == REJECT[key])[0]
                    if len(rejected_labels):
                        print(f'rejected_labels: {rejected_labels}')
                        print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
                        visualize_selected_labels(xyzi[:,:3], labels.flatten(), rejected_labels)
        
        labels = remove_outliers_cluster(xyzi[:,:3], new_labels.flatten())
        assert labels.shape[0] == num_pts, f'After remove outliers: Some labels missing for seq: {seq_name}, sample {sample_idx}!'

        labels = get_continuous_labels(labels)
        assert labels.shape[0] == num_pts, f'After Continuous Labels: Some labels missing for seq: {seq_name}, sample {sample_idx}!'

        if show_plots:
            print(f'Removed cluster pts far away from main cluster')
            visualize_pcd_clusters(xyzi[:,:3], labels.reshape((-1,1)))


        if save_rejection_tag:
            pt_wise_rejection_tag[labels>-1] = 0
            pt_wise_rejection_tag.astype(np.uint8)

            # Save rejection tag for each pt
            save_path = save_seq_path / 'rejection_tag'/ ('%04d.npy' % sample_idx)
            pt_wise_rejection_tag.tofile(save_path.__str__())
   
        save_path = save_seq_path / ('%04d.npy' % sample_idx)
        labels = labels.astype(np.float16)
        assert labels.shape[0] == num_pts, f'Some labels missing for seq: {seq_name}, sample {sample_idx}!'
        labels.tofile(save_path.__str__())
        print(f'Saved sample: {sample_idx}')

def cluster_tracking(seq_name, dataset, num_frames_to_aggr = 3, initial_guess=False, show_plots=False):
    save_seq_path = dataset.save_label_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    os.makedirs((save_seq_path / 'rejection_tag').__str__(), exist_ok=True)
    print(f'Clustering of sequence: {seq_name} started!')
    
    if initial_guess:
        #get clusters from the aggregated pcs of whole seq
        path = save_seq_path / 'max_cluster_id.pkl'
        with open(path, 'rb') as f:
            max_cluster_id_dict = pickle.load(f)
        max_label = max_cluster_id_dict['max_cluster_id_aggregated']
    else:
        max_label = -1

    # path = save_seq_path / 'cluster2frame_id_dict.pkl'
    # with open(path, 'rb') as f:
    #     cluster2frame_id_dict = pickle.load(f)

    infos = dataset.infos_dict[seq_name]
    start_info_indices = np.arange(0, len(infos)-num_frames_to_aggr+1)

    for start_idx in start_info_indices:
        
        aggr_infos = infos[start_idx:start_idx+num_frames_to_aggr]
        aggr_pcs_in_world = np.zeros((0,3))
        aggr_ground_mask = np.zeros(0, dtype=np.bool_)
        old_labels = np.zeros(0)
        pc_lens = [] 
        for i, info in enumerate(aggr_infos):
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']
            print(f'sample idx: {sample_idx}')
            
            xyzi = dataset.get_lidar(seq_name, sample_idx)
            if initial_guess:
                labels = dataset.get_cluster_labels(seq_name, sample_idx)
            else:
                if start_idx == start_info_indices[0] or i == len(aggr_infos)-1:
                    labels = -1. * np.ones(xyzi.shape[0], dtype=np.float16)
                else:
                    labels = dataset.get_cluster_labels(seq_name, sample_idx)
            ground_mask = dataset.get_ground_mask(seq_name, sample_idx)
            xyzi[:,:3] = transform_pc_to_world(xyzi[:,:3], info['pose'])
            pc_lens.append(xyzi.shape[0])
            
            
            aggr_ground_mask = np.hstack([aggr_ground_mask, ground_mask])
            if initial_guess and start_idx != start_info_indices[0] and i == len(aggr_infos)-1:
                
                old_labeled_pcd = o3d.geometry.PointCloud()
                old_labeled_pcd.points = o3d.utility.Vector3dVector(aggr_pcs_in_world[old_labels>-1])

                old_labeled_tree = o3d.geometry.KDTreeFlann(old_labeled_pcd)
                new_labels = -1* np.ones_like(labels)
                for l in np.unique(labels):
                    if l == -1:
                        continue
                    points_new_lbl = xyzi[labels == l]
                    for i in range(points_new_lbl.shape[0]):
                        pt = points_new_lbl[i,:3]
                        #Find its neighbors with distance less than 0.2
                        [_, idx, _] = old_labeled_tree.search_radius_vector_3d(pt, 0.2)
                        if len(idx):
                            nearest_labels = old_labels[old_labels>-1][np.asarray(idx)]
                            label_of_majority = np.bincount(nearest_labels.astype(int)).argmax()
                            new_labels[labels == l] = label_of_majority
                            break
                
                labels = new_labels
            aggr_pcs_in_world = np.vstack([aggr_pcs_in_world, xyzi[:,:3]])
            old_labels = np.hstack([old_labels, labels])

        # Get new labels
        labels = cluster(aggr_pcs_in_world, np.logical_not(aggr_ground_mask), eps=0.2)
        print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')
        # if show_plots:
        #     print('showing old labels')
        #     visualize_pcd_clusters(aggr_pcs_in_world, old_labels.reshape((-1,1)))
        #     print('showing new clustered labels')
        #     visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))

        new_labels, label_wise_rejection_tag  = filter_labels(aggr_pcs_in_world, labels,
                                    max_volume=None, min_volume=0.1, 
                                    max_height_for_lowest_point=1, 
                                    min_height_for_highest_point=0.5,
                                    ground_mask = aggr_ground_mask)
        
        pt_wise_rejection_tag = np.zeros(aggr_pcs_in_world.shape[0], dtype=np.uint8) #zero means not rejected
        for i in np.unique(labels):
            if i == -1:
                continue
            pt_wise_rejection_tag[labels == i] = label_wise_rejection_tag[int(i)]


        # if show_plots:
        #     for key, val in REJECT.items():
        #         rejected_labels = np.where(label_wise_rejection_tag == REJECT[key])[0]
        #         if len(rejected_labels):
        #             print(f'rejected_labels: {rejected_labels}')
        #             print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
        #             visualize_selected_labels(aggr_pcs_in_world, labels.flatten(), rejected_labels)
        
        labels = remove_outliers_cluster(aggr_pcs_in_world, new_labels.flatten())
        labels = get_continuous_labels(labels)
        print(f'2nd Step Filtering Done. Labels: {np.unique(labels).shape[0]}')
        # if show_plots:
        #     #visualize_pcd_clusters(aggr_pcs_in_world, old_labels.reshape((-1,1)))
        #     print('showing new filtered labels')
        #     visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))

        lbls_only_in_new_labels = []
        label_new2old_dict = {}

        for i in np.unique(labels):
            if i == -1:
                continue
            old_lbls_for_lbl_i = old_labels[labels==i]
            if old_lbls_for_lbl_i.max() == -1:
                lbls_only_in_new_labels.append(i)
            else:
                obj_old_labels = old_lbls_for_lbl_i[old_lbls_for_lbl_i>-1]
                label_new2old_dict[i] = {'majority_old_label': -1,
                                         'old_labels': [],
                                         'old_labels_count': []}
                # old_labels_count = []
                for old_lbl in np.unique(obj_old_labels):
                    label_new2old_dict[i]['old_labels'].append(old_lbl)
                    label_new2old_dict[i]['old_labels_count'].append((obj_old_labels == old_lbl).sum()) #old_labels == old_lbl
                
                label_new2old_dict[i]['majority_old_label'] = label_new2old_dict[i]['old_labels'][np.argmax(label_new2old_dict[i]['old_labels_count'])]

        # Start fusing aggregated view and new frame view labels
        new_labels = old_labels.copy()
        
        #Add labels only in new labels
        if len(lbls_only_in_new_labels):
            label_mapping = {lbl:i+max_label+1 for i, lbl in enumerate(sorted(lbls_only_in_new_labels))}
            for lbl, new_lbl in label_mapping.items():
                new_labels[labels == lbl] = new_lbl
                #cluster2frame_id_dict[new_lbl] = [frame_idx]
            max_label = np.max(new_labels)
            #save max cluster id
            if initial_guess:
                path = save_seq_path / 'max_cluster_id.pkl'
                max_cluster_id_dict['max_cluster_id'] = max_label
                with open(path, 'wb') as f:
                    pickle.dump(max_cluster_id_dict,f)
            
            # # save new cluster2frame_id_dict
            # save_path = save_seq_path / 'cluster2frame_id_dict.pkl'
            # with open(save_path, 'wb') as f:
            #     pickle.dump(cluster2frame_id_dict, f)

        # if show_plots:
        #     visualize_pcd_clusters(aggr_pcs_in_world, new_labels.reshape((-1,1)))
        # Keep old labels if old2new label connection exists
        for new_lbl, value in label_new2old_dict.items():
            new_labels[labels == new_lbl] = value['majority_old_label']

        if show_plots:
            # print('showing old labels')
            # visualize_pcd_clusters(aggr_pcs_in_world, old_labels.reshape((-1,1)))
            # print('showing new_filtered labels')
            # visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))
            print('showing final labels')
            visualize_pcd_clusters(aggr_pcs_in_world, new_labels.reshape((-1,1)))

        i=0
        new_labels = new_labels.astype(np.float16)
        pt_wise_rejection_tag[new_labels>-1] = 0
        pt_wise_rejection_tag.astype(np.uint8)
        for info, pc_len in zip(aggr_infos, pc_lens):
            sample_idx = info['point_cloud']['sample_idx']
            print(f'Saving sample: {sample_idx}')
            save_path = save_seq_path / ('%04d.npy' % sample_idx)

            label_this_pc = new_labels[i:i+pc_len]
            label_this_pc.tofile(save_path.__str__())

            # Save rejection tag for each pt
            save_path = save_seq_path / 'rejection_tag'/ ('%04d.npy' % sample_idx)
            rej_tag_this_pc = pt_wise_rejection_tag[i:i+pc_len]
            rej_tag_this_pc.tofile(save_path.__str__())

            i+=pc_len

def merge_overlapping_boxes(seq_name, dataset, iou_thresh, method, show_plots=False):
    aggr_pcs_in_world = np.zeros((0,3))
    aggr_ground_mask = np.zeros(0, dtype=np.bool_)
    aggr_labels = np.zeros(0)
    pc_lens = [] 
    aggr_boxes_world_frame = np.zeros((0, 8)) #cxyz, lwh, heading, label

    #Load approx boxes
    approx_boxes_path = dataset.save_label_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f)

    # aggregate infos
    for i, info in enumerate(infos):
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']
        print(f'sample idx: {sample_idx}')
        
        xyzi = dataset.get_lidar(seq_name, sample_idx)
        xyzi[:,:3] = transform_pc_to_world(xyzi[:,:3], info['pose'])
        pc_lens.append(xyzi.shape[0])
        
        ground_mask = dataset.get_ground_mask(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx)

        aggr_ground_mask = np.hstack([aggr_ground_mask, ground_mask])
        aggr_pcs_in_world = np.vstack([aggr_pcs_in_world, xyzi[:,:3]])
        aggr_labels =  np.hstack([aggr_labels, labels])

        det_boxes_in_v = info[f'approx_boxes_{method}']
        cluster_labels_boxes = info['cluster_labels_boxes']
        # for i in range(det_boxes.shape[0]):
        #     box_label = cluster_labels_boxes[i]
        #     if box_label not in unique_cluster_boxes:
        #         unique_cluster_boxes[box_label] = [det_boxes[i]]
        #     else:
        #         unique_cluster_boxes[box_label].append(det_boxes[i])
        pose_v_to_w = info['pose']
        det_boxes_in_w = transform_box(det_boxes_in_v, pose_v_to_w)
        det_boxes_in_w = np.hstack([det_boxes_in_w, cluster_labels_boxes.reshape(-1,1)])
        aggr_boxes_world_frame = np.vstack([aggr_boxes_world_frame, det_boxes_in_w])
    
    
    iou3d_det_det, _, _ = iou3d_nms_utils.boxes_iou3d_gpu(torch.from_numpy(aggr_boxes_world_frame[:, 0:7]).float().cuda(), 
                                                         torch.from_numpy(aggr_boxes_world_frame[:, 0:7]).float().cuda())
    
    iou3d = iou3d_det_det.cpu().numpy()

    # max_scores = (iou3d_det_gt.max(dim=0)[0]).cpu().numpy() # for each gt col, find max iou3d across all dets/rows

    map_= {}
    aggr_boxes_unique_labels = np.unique(aggr_boxes_world_frame[:,-1])
    for box_lbl in aggr_boxes_unique_labels:
        boxes_this_lbl_mask = aggr_boxes_world_frame[:,-1] == box_lbl
        boxes_other_lbl_mask = np.logical_not(boxes_this_lbl_mask)

        iou3d_this_box_lbl = iou3d[boxes_this_lbl_mask][:, boxes_other_lbl_mask]
        indices = (iou3d_this_box_lbl > iou_thresh).nonzero()
        if indices[0].shape[0]:
            map_[box_lbl] = {'overlapping_box_cluster_labels': [],
                         'overlapping_box_ious': [],
                         'unique_labels': []}
            rows, cols = indices
            matching_box_indices = boxes_other_lbl_mask.nonzero()[0][cols]
            matching_box_labels = aggr_boxes_world_frame[matching_box_indices][:,-1]
            map_[box_lbl]['overlapping_box_cluster_labels'] = matching_box_labels
            map_[box_lbl]['unique_labels'] = np.unique(matching_box_labels)
            map_[box_lbl]['overlapping_box_ious'] = iou3d_this_box_lbl[rows, cols]
    
    final_map_=[]
    ids_done = set()
    box_lbls_with_multiple_matches = []
    for box_lbl, val in map_.items():
        ids_done.add(box_lbl)
        if len(val['unique_labels']) == 1:
            matching_lbl = val['unique_labels'][0]
            if matching_lbl not in ids_done and len(map_[matching_lbl]['unique_labels']) == 1:
                final_map_.append([box_lbl, matching_lbl])
                ids_done.add(matching_lbl)
        else:
            matching_lbls = val['unique_labels']
            for l in matching_lbls:
                if l not in ids_done:
                    matches = [box_lbl]
                    matches += matching_lbls.tolist()
                    box_lbls_with_multiple_matches.append(matches)
                    break

            for l in matching_lbls:
                ids_done.add(l)
        
        
    
    print(len(final_map_))
    print(len(box_lbls_with_multiple_matches))
    f = np.concatenate(final_map_)
    mm =  np.concatenate(box_lbls_with_multiple_matches)

    if show_plots:
        visualize_selected_labels(aggr_pcs_in_world, aggr_labels, np.concatenate([f, mm]))    
    # visualize_selected_labels(aggr_pcs_in_world, aggr_labels, f)
    # visualize_selected_labels(aggr_pcs_in_world, aggr_labels, mm)
    # print(len(box_lbls_with_multiple_matches))
    # for i in box_lbls_with_multiple_matches:
    #     visualize_selected_labels(aggr_pcs_in_world, aggr_labels, i)

    for m in final_map_:
        lbl = min(m)
        aggr_labels[aggr_labels==m[0]] = lbl
        aggr_labels[aggr_labels==m[1]] = lbl

    for m in box_lbls_with_multiple_matches:
        lbl = min(m)
        for i in m:
            aggr_labels[aggr_labels==i] = lbl

    aggr_labels = get_continuous_labels(aggr_labels)
    if show_plots:
        visualize_pcd_clusters(aggr_pcs_in_world, aggr_labels.reshape(-1,1))
    
    save_seq_path = dataset.save_label_path / seq_name
    i=0
    aggr_labels = aggr_labels.astype(np.float16)
    for info, pc_len in zip(infos, pc_lens):
        sample_idx = info['point_cloud']['sample_idx']
        save_label_path = save_seq_path / ('%04d.npy' % sample_idx)
        print(f'Saving labels: {sample_idx}')

        label_this_pc = aggr_labels[i:i+pc_len]
        i+=pc_len

        label_this_pc.tofile(save_label_path.__str__())


def fit_approx_boxes_seq(seq_name, dataset, show_plots=False, method = 'closeness_to_edge', simple_cluster=False):
    #Load approx boxes
    approx_boxes_path = dataset.save_label_path / seq_name / 'approx_boxes.pkl'

    if simple_cluster and approx_boxes_path.exists():
        print(f'{seq_name} already clustered and bboxes fitted.')
        return

    try:
        with open(approx_boxes_path, 'rb') as f:
            infos = pickle.load(f)
    except:
        infos= dataset.infos_dict[seq_name]

    print(f'Fitting boxes for sequence: {seq_name}')
    for i, info in enumerate(infos):
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']

        #points in current vehicle frame
        pc = dataset.get_lidar(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx)

        approx_boxes_this_pc = np.empty((0, 18)) #cxyz, lwh, heading, bev_corners.flatten(), frame_info_idx, num_points in this box, label
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_pc = pc[labels==label, :]
            if cluster_pc.shape[0] < 10:
                continue
            box, corners, _ = fit_box(cluster_pc, fit_method=method)
            full_box = np.zeros((1, approx_boxes_this_pc.shape[-1]))
            full_box[0,:7] = box
            full_box[0,7:15] = corners.flatten()
            full_box[0,15] = i # info index
            full_box[0,16] = cluster_pc.shape[0] # num_points
            full_box[0,17] = label
            approx_boxes_this_pc = np.vstack([approx_boxes_this_pc, full_box])
            # [cxy[0], cxy[1], cz, l, w, h, rz, corner0_x, corner0_y, ..., corner3_x, corner3_y,
            # info index, num cluster pts, label]
            # corner0-3 are BEV box corners in lidar frame
        
        # Fitting boxes done for this pc
        #assert np.unique(labels).shape[0] - 1 == approx_boxes_this_pc.shape[0]
        info[f'approx_boxes_{method}'] = approx_boxes_this_pc.astype(np.float32)
        info['cluster_labels_boxes'] = approx_boxes_this_pc[:, -1]


        if show_plots:
            gt_boxes = info['annos']['gt_boxes_lidar']
            show_bev_boxes(pc[labels>-1], approx_boxes_this_pc, 'unrefined_approx_boxes')
            V.draw_scenes(pc, gt_boxes=gt_boxes, 
                                ref_boxes=approx_boxes_this_pc[:,:7], ref_labels=None, ref_scores=None, 
                                color_feature=None, draw_origin=True)
            
    print(f'Fitting boxes Done.')
    save_path = dataset.save_label_path / seq_name / 'approx_boxes.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)

def refine_boxes_seq(seq_name, dataset, method, show_plots=False):
    
    #Load approx boxes
    approx_boxes_path = dataset.save_label_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f)
    
    approx_boxes = np.empty((0, 18))
    num_boxes_per_pc = np.zeros(len(infos), dtype=int)
    for i, info in enumerate(infos):
        num_boxes_per_pc[i] = info[f'approx_boxes_{method}'].shape[0]
        approx_boxes= np.vstack([approx_boxes, info[f'approx_boxes_{method}']])
    
    #Refine boxes cxyz, lwh, heading, bev_corners.flatten(), label
    refined_boxes = refine_boxes(approx_boxes, approx_boxes_labels=approx_boxes[:,-1])
    print(f'Refining boxes Done.')

    if show_plots:
        ind = 0
        for i, info in enumerate(infos):
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']
            gt_boxes = info['annos']['gt_boxes_lidar']

            #points in current vehicle frame
            pc = dataset.get_lidar(seq_name, sample_idx)
            labels = dataset.get_cluster_labels(seq_name, sample_idx).flatten()
            num_boxes_this_pc = int(num_boxes_per_pc[i])
            approx_boxes_this_pc = approx_boxes[ind:ind+num_boxes_this_pc]
            refined_boxes_this_pc = refined_boxes[ind:ind+num_boxes_this_pc]
            ind += num_boxes_this_pc

            gt_boxes_corners = np.zeros((gt_boxes.shape[0], 8))
        
            for j in range(gt_boxes.shape[0]):
                corners = get_box_corners(gt_boxes[j, :3], gt_boxes[j, 3:6], gt_boxes[j, 6])
                gt_boxes_corners[j, :] = corners.flatten()

            gt_boxes = np.hstack([gt_boxes, gt_boxes_corners])
            
            savefig_path = dataset.save_label_path / seq_name/ ('%04d.png' % sample_idx)
            savefig_path = savefig_path.__str__() #None
            # show_bev_boxes(pc[labels>-1], approx_boxes_this_pc, f'approx_boxes_{method}', \
            #                refined_boxes_this_pc, f'refined_boxes_{method}', gt_boxes, 'gt_boxes',\
            #                 savefig_path=savefig_path)
            # V.draw_scenes(pc, gt_boxes=approx_boxes_this_pc[:,:7], 
            #                     ref_boxes=refined_boxes_this_pc[:,:7]) #gt_boxes=blue, ref_boxes=green
            V.draw_scenes(pc, gt_boxes=gt_boxes, 
                                ref_boxes=refined_boxes_this_pc[:,:7])
    
    #save sequence boxes
    ind = 0
    for i, info in enumerate(infos):
        num_boxes = num_boxes_per_pc[i]
        boxes_this_pc = refined_boxes[ind:ind+num_boxes, :]
        info[f'refined_boxes_{method}'] = boxes_this_pc.astype(np.float32)
        # info['cluster_labels_boxes'] = refined_boxes[ind:ind+num_boxes, -1]
        ind += num_boxes
    
    save_path = dataset.save_label_path / seq_name / 'approx_boxes.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)


def transform_box(box, pose):
    """Transforms 3d upright boxes from one frame to another.
    Args:
    box: [..., N, 7] boxes.
    from_frame_pose: [...,4, 4] origin frame poses.
    to_frame_pose: [...,4, 4] target frame poses.
    Returns:
    Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    transform = pose 
    heading = box[..., 6] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0]) # heading of obj x/front axis wrt world x axis = rz (angle b/w obj x and ego veh x) + angle between ego veh x and world x axis
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2) # box center wrt ego vehicle frame -> transform wrt world frame

    return np.concatenate([center, box[..., 3:6], heading[..., np.newaxis]], axis=-1)

def remove_outliers_cluster(xyz, labels):
    for i in np.unique(labels):
        if i == -1:
            continue
        cluster_indices = labels==i
        cluster_pc = xyz[cluster_indices]
        cluster_labels = labels[cluster_indices]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_pc)

        # set as background if point in cluster has less than 2 neighbors within 0.4 m distance
        tree = o3d.geometry.KDTreeFlann(pcd)
        for i in range(cluster_pc.shape[0]):
            [_, idx, _] = tree.search_radius_vector_3d(cluster_pc[i], 0.4)
            if len(idx) < 2:
                cluster_labels[i] = -1
        
        labels[cluster_indices] = cluster_labels
    
    return labels
             
def show_bev_boxes(pc, boxes1, label1, boxes2=None, label2=None, boxes3=None, label3=None, savefig_path=None, show_rot=False, iou3d=None):
    fig=plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(pc[:,0], pc[:,1], s=2)
    ax.arrow(0,0,2,0, facecolor='red', linewidth=1, width=0.5) #x-axis
    ax.arrow(0,0,0,2, facecolor='green', linewidth=1, width=0.5) #y-axis
    handles =[]

    bev_corners1 = boxes1[:, 7:15].reshape((-1,4,2))
    handles.append(Line2D([0], [0], label=label1, color='k'))
    for i in range(bev_corners1.shape[0]):
        draw2DRectangle(ax, bev_corners1[i].T, color='k')
        if show_rot:
            ax.text(boxes1[i, 0], boxes1[i, 1],  "{:.2f}".format(np.rad2deg(boxes1[i, 6])), color='black', fontsize = 10, bbox=dict(facecolor='yellow', alpha=0.5))

    if boxes2 is not None:
        handles.append(Line2D([0], [0], label=label2, color='m'))
        bev_corners2 = boxes2[:, 7:15].reshape((-1,4,2))
        for i in range(bev_corners2.shape[0]):   
            draw2DRectangle(ax, bev_corners2[i].T, color='m')
            if iou3d is not None:
                ax.text(boxes2[i, 0]+0.3, boxes2[i, 1]+0.3,  "{:.2f}".format(iou3d[i]), color='black', fontsize = 10, bbox=dict(facecolor='green', alpha=0.5))

    
    if boxes3 is not None:
        bev_corners3 = boxes3[:, 7:15].reshape((-1,4,2))
        handles.append(Line2D([0], [0], label=label3, color='g'))
        for i in range(bev_corners3.shape[0]):   
            draw2DRectangle(ax, bev_corners3[i].T, color='g')
            if show_rot:
                ax.text(boxes3[i, 0]+0.3, boxes3[i, 1]+0.3,  "{:.2f}".format(np.rad2deg(boxes3[i, 6])), color='black', fontsize = 10, bbox=dict(facecolor='green', alpha=0.5))


    ax.legend(handles=handles, fontsize='large', loc='upper right')
    ax.grid()
    if savefig_path is not None:
        plt.savefig(savefig_path)
    else:
        plt.show()


def run_simple_cluster(seq_name, dataset, show_plots=False, save_rejection_tag=False):
    dataset.estimate_ground_seq(seq_name)
    simple_cluster(seq_name, dataset, show_plots=show_plots, save_rejection_tag=save_rejection_tag)
    fit_approx_boxes_seq(seq_name, dataset, method='closeness_to_edge', show_plots=show_plots, simple_cluster=True) #fit using closeness or min max?

def run1(seq_name, dataset):
    dataset.estimate_ground_seq(seq_name)
    cluster_tracking(seq_name, dataset, initial_guess=False, show_plots=False)
    fit_approx_boxes_seq(seq_name, dataset, method='naive_min_max', show_plots=False) #fit using closeness or min max?

def run2(seq_name, dataset):
    merge_overlapping_boxes(seq_name, dataset, iou_thresh=0.5,method='naive_min_max', show_plots=False) # for some clusters it does not work, maybe try just using fitted boxes?

def run3(seq_name, dataset):
    # TODO: Some clusters have different/colourful ids although they belong to the same object. If you want all points of the same object to have one cluster id.
    # boxes of different cluster ids but high iou-> make all their cluster points the same id and then refit naive min max and refine
    fit_approx_boxes_seq(seq_name, dataset, method='naive_min_max', show_plots=False) #fit using closeness or min max?
    refine_boxes_seq(seq_name, dataset, method='naive_min_max', show_plots=False)
    fit_approx_boxes_seq(seq_name, dataset, method='closeness_to_edge', show_plots=False) #fit using closeness or min max?
    refine_boxes_seq(seq_name, dataset, method='closeness_to_edge', show_plots=False)



def main():
    args = parser.parse_args()
    dataset = WaymoDataset(split=args.split, processed_data_tag=args.processed_data_tag)
    num_workers = mp.cpu_count() - 2
    seq_name_list = [seq_name for seq_name in dataset.infos_dict]

    if args.mode == 'simple_cluster':
        run_func = partial(run_simple_cluster, dataset=dataset, save_rejection_tag=args.save_rejection_tag)
        with mp.Pool(num_workers) as p:
            results = list(tqdm(p.imap(run_func, seq_name_list), total=len(seq_name_list)))
        
        #for seq_name in dataset.infos_dict:
        # seq_name = 'segment-10023947602400723454_1120_000_1140_000_with_camera_labels'
        # run_simple_cluster(seq_name, dataset,  show_plots=False)

    else:
        run_func = partial(run1, dataset=dataset)

        with mp.Pool(num_workers) as p:
            results = list(tqdm(p.imap(run_func, seq_name_list), total=len(seq_name_list)))

        for seq_name in dataset.infos_dict:
            run2(seq_name, dataset)
        
        run_func = partial(run3, dataset=dataset)

        with mp.Pool(num_workers) as p:
            results = list(tqdm(p.imap(run_func, seq_name_list), total=len(seq_name_list)))



if __name__ == '__main__':
    main()


