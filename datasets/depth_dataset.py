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

from datasets.transforms import  data_augmentor, data_processor
from datasets.features import global_descriptors

from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
from utils.pcd_preprocess import visualize_selected_labels
import open3d as o3d
    
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

#np.random.seed(1024)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DepthContrastDataset(Dataset):
    """Base Self Supervised Learning Dataset Class."""

    def __init__(self, cfg, pretraining = True, mode='train', logger=None):
        self.mode = mode
        self.pretraining = pretraining
        self.logger = logger
        self.cfg = cfg
        self.root_path = (Path(__file__) / '../..').resolve()  # DepthContrast
        self.point_cloud_range = np.array(cfg["POINT_CLOUD_RANGE"], dtype=np.float32)
        self.class_names = cfg["CLASS_NAMES"]
        self.used_num_point_features  = 4

        self.data_augmentor = data_augmentor.DataAugmentor(self.cfg["POINT_TRANSFORMS"])


        #### Add the voxelizer here
        self.grid_size = None
        self.voxel_size = None
        self.depth_downsample_factor = None 
        if cfg["VOX"]:
            self.voxel_size = [0.1, 0.1, 0.15] #[0.05, 0.05, 0.1] #

            self.MAX_POINTS_PER_VOXEL = 5
            self.MAX_NUMBER_OF_VOXELS = 150000 #80000
            if SPCONV_VER == 1:
                self.voxel_generator = VoxelGenerator(
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range,
                    max_num_points=self.MAX_POINTS_PER_VOXEL,
                    max_voxels=self.MAX_NUMBER_OF_VOXELS
                )
            else:
                self.voxel_generator = VoxelGenerator(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features = 5,
                    max_num_points_per_voxel=self.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=self.MAX_NUMBER_OF_VOXELS
                )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
            self.grid_size = np.round(grid_size).astype(np.int64)

    def toVox(self, points):
        if SPCONV_VER==1:
            voxel_output = self.voxel_generator.generate(points)
        else:
            voxel_output = self.voxel_generator(torch.from_numpy(points).contiguous())
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            try:
                voxels, coordinates, num_points = voxel_output
            except: 
                voxels, coordinates, num_points, pc_voxel_id = voxel_output 

        data_dict = {}
        data_dict['voxels'] = voxels #dim =(num_voxels, max_points_per_voxel=5, coords+feats = 3+1=4), gives xyzi value for each point in each voxel, if num points in voxel < 5, fill in zeros
        data_dict['voxel_coords'] = coordinates #dim= (num voxels, 3), gives z,y,x grid coord of each voxel 
        data_dict['voxel_num_points'] = num_points # dim = num_voxels, gives num points in each voxel
        return data_dict
    
    def prepare_data_downstream(self, data_dict):
        #points: xyzi, seglabel
        #gt_boxes: xyz, dx, dy, dz, rz, class_lbl_idx i.e. 1 for vehicle etc
        cfg = self.cfg

        #remove points outside range
        data_dict['points'] = data_processor.mask_points_outside_range(data_dict['points'], self.point_cloud_range)

        # transform data_dict points and gt_boxes #TODO: check if augmentor assumes and returns 7 dim gt boxes
        if self.mode == 'train':
            gt_classes_idx = data_dict["gt_boxes"][:,-1].reshape(-1,1)
            data_dict["points"], data_dict["gt_boxes"] = self.data_augmentor.forward(data_dict["points"], data_dict["gt_boxes"][:,:7])
            #Reappend class_idx
            data_dict["gt_boxes"] = np.hstack([data_dict["gt_boxes"], gt_classes_idx])
        
        if not cfg["VOX"]:
            data_dict['points'] = data_processor.sample_points(data_dict['points'], self.cfg["SAMPLE_NUM_POINTS"] )

        if self.mode == 'train':
            data_dict['points'] = data_processor.shuffle_points(data_dict['points'])
            
            #Only needed if augmentation removes points
            # If augmentor removes a patch with gt box, remove its gt box 
            data_dict['points'], data_dict['gt_boxes'], data_dict['pt_wise_gtbox_idxs'] = data_processor.mask_boxes_with_few_points(data_dict['points'], data_dict['gt_boxes'])
        else:
            data_dict['pt_wise_gtbox_idxs'] = data_processor.find_pt_wise_gtbox_idx(data_dict['points'], data_dict['gt_boxes'])
        if cfg["VOX"]:
            vox_dict = self.toVox(data_dict["points"]) # xyzil=seg label 
            data_dict["vox"] = vox_dict


        return data_dict
    def prepare_data_pretrain(self, data_dict):
        
        PLOT= False
        cfg = self.cfg

        # remove points outside range
        data_dict['points'] = data_processor.mask_points_outside_range(data_dict['points'], self.point_cloud_range)
    

        if PLOT:
            # After cropping points outside range
            V.draw_scenes(points=data_dict["points"][:,:4], gt_boxes=data_dict["gt_boxes"][:,:7])

        assert len(data_dict['points']) > 0

        # #Create different views / augmentation
        data_dict['points_moco'] = np.copy(data_dict["points"])
        data_dict['gt_boxes_moco'] = np.copy(data_dict["gt_boxes"])
        gt_classes_idx = data_dict["gt_boxes"][:,-2].reshape(-1,1)
        gt_cluster_ids = data_dict["gt_boxes"][:,-1].reshape(-1,1)
        data_dict['unscaled_lwhz_cluster_id'] = np.hstack([data_dict["gt_boxes"][:,3:6], data_dict["gt_boxes"][:,2].reshape(-1,1), gt_cluster_ids])
        
        # l_a= torch.from_numpy(data_dict['unscaled_lwhz_cluster_id'][:,0]).reshape(-1, 1)
        # w_a= torch.from_numpy(data_dict['unscaled_lwhz_cluster_id'][:,1]).reshape(-1, 1)
        # h_a= torch.from_numpy(data_dict['unscaled_lwhz_cluster_id'][:,2]).reshape(-1, 1)
        # z_a= torch.from_numpy(data_dict['unscaled_lwhz_cluster_id'][:,3]).reshape(-1, 1)

        # l_b = l_a.reshape(1, -1)
        # w_b = w_a.reshape(1, -1)
        # h_b = h_a.reshape(1, -1)
        # z_b = z_a.reshape(1, -1)

        # boxes_a_height_max = (z_a + h_a / 2).reshape(-1, 1) #col
        # boxes_a_height_min = (z_a - h_a / 2).reshape(-1, 1) #col
        # boxes_b_height_max = boxes_a_height_max.reshape(1, -1) #row
        # boxes_b_height_min = boxes_a_height_min.reshape(1, -1) #row

        # max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min) #(Npos, Nneg)
        # min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max) #(Npos, Nneg)
        # overlaps_h = torch.clamp(min_of_max - max_of_min, min=0) # torch.min(h, h_neg)  #(Npos, Nneg) height overlaps between each pos and neg sample

        # vol_a = (l_a*w_a*h_a).view(-1, 1) # col: Nposx1
        # vol_b = (l_b * w_b * h_b).view(1, -1) # row: 1xK
        # overlap_vol = torch.min(l_a, l_b) *  torch.min(w_a, w_b) * overlaps_h # NxK
        # iou3d = overlap_vol / torch.clamp(vol_a + vol_b - overlap_vol, min=1e-6) # NxK
        # iou3d_dist = 1-iou3d


        # observe_cluster_id = 1
        # observe_cluster_idx=np.where(observe_cluster_id == gt_cluster_ids)[0]
        # #visualize_selected_labels(data_dict["points"], data_dict["points"][:, -1], [observe_cluster_id])
        # n_neighbors=50
        # iou_dist_vec = iou3d_dist[observe_cluster_idx].numpy().flatten()
        # # idx_knn = np.argsort(iou_dist_vec)[:n_neighbors]
        # # iou_dist_knn = np.sort(iou_dist_vec)[:n_neighbors]
        # # print(f'IOU with {n_neighbors}-nn:')
        # # print(1-iou_dist_knn)
        # # visualize_selected_labels(data_dict["points"], data_dict["points"][:, -1], gt_cluster_ids[idx_knn])

        # iou_dist_thresh = 0.8 #(0.6, 0.7, 0.8) record percentage removed
        # idx = np.where(iou_dist_vec <= iou_dist_thresh)[0]
        # iou_dist = iou_dist_vec[idx]
        # print(f'Num samples with iou dist <= {iou_dist_thresh}: {len(idx)}')
        # print(iou_dist)
        # visualize_selected_labels(data_dict["points"], data_dict["points"][:, -1], gt_cluster_ids[idx])


        # b=1

        if 'EXTRACT_SHAPE_DESCRIPTORS' in cfg:
            method=cfg['EXTRACT_SHAPE_DESCRIPTORS']
            shape_descs = [] 
            cluster_ids_for_shape_descs=[]
            for idx, i in enumerate(gt_cluster_ids):
                obj_points_mask = data_dict["points"][:, -1] == i
                obj_points = data_dict["points"][obj_points_mask,:3] - data_dict["gt_boxes"][idx,:3] # for vfh
                if len(obj_points) > 5:
                    shape_desc = global_descriptors.extract_feats(obj_points, method=method)
                    # visualize_selected_labels(obj_points, data_dict["points"][obj_points_mask, -1], [i])

                    if shape_desc is not None:
                        shape_descs.append(shape_desc)
                        cluster_ids_for_shape_descs.append(i)
                        #print(f'cluster_id: {i}')
                        #visualize_selected_labels(data_dict["points"], data_dict["points"][:, -1], [i])

            shape_descs = np.array(shape_descs)
            cluster_ids_for_shape_descs = np.array(cluster_ids_for_shape_descs).flatten()
            data_dict['shape_descs'] = shape_descs
            data_dict['shape_desc_cluster_ids']= cluster_ids_for_shape_descs

            
        #     observe_cluster_id = 1
        #     n_neighbors=50
        #     query_desc = shape_descs[cluster_ids_for_shape_descs==observe_cluster_id]

        #     # kd_tree = o3d.geometry.KDTreeFlann(shape_descs.T)
        #     # [k, idx, sqdist] = kd_tree.search_knn_vector_xd(query_desc.T, n_neighbors)

        #     rmse = np.linalg.norm(query_desc - shape_descs, axis=1) # (N objs)
        #     idx_knn = np.argsort(rmse)[:n_neighbors]
        #     rmse_knn = np.sort(rmse)[:n_neighbors]
        #     #sq_dist_knn = rmse_knn**2
        #     print(f'RMSE with {n_neighbors}-nn:')
        #     print(rmse_knn)

        #     visualize_selected_labels(data_dict["points"], data_dict["points"][:, -1], cluster_ids_for_shape_descs[idx_knn])

        #     b=1

        
        # transform data_dict points and gt_boxes
        data_dict["points"], data_dict["gt_boxes"] = self.data_augmentor.forward(data_dict["points"], data_dict["gt_boxes"][:,:7], gt_box_cluster_ids=gt_cluster_ids)

        # transform data_dict points_moco and gt_boxes_moco
        data_dict["points_moco"], data_dict["gt_boxes_moco"] = self.data_augmentor.forward(data_dict["points_moco"], data_dict["gt_boxes_moco"][:,:7], gt_box_cluster_ids=gt_cluster_ids)
        
        #reappend the gt class indexes and cluster ids
        data_dict["gt_boxes"] = np.hstack([data_dict["gt_boxes"], gt_classes_idx, gt_cluster_ids])
        data_dict["gt_boxes_moco"] = np.hstack([data_dict["gt_boxes_moco"], gt_classes_idx, gt_cluster_ids])
        
        # cluster_ids, cnts = np.unique(data_dict['points'][:,-1], return_counts=True)
        # for cluster_id, cnt in zip(cluster_ids, cnts):
        #     if cluster_id == -1:
        #         continue
        #     frame_id = data_dict['frame_id']
        #     assert cluster_id in data_dict['gt_boxes'][:,-1], f'{frame_id}, cluster_label: {cluster_id}, cnts:{cnt}'

        # cluster_ids, cnts = np.unique(data_dict['points_moco'][:,-1], return_counts=True)
        # for cluster_id, cnt in zip(cluster_ids, cnts):
        #     if cluster_id == -1:
        #         continue
        #     frame_id = data_dict['frame_id']
        #     assert cluster_id in data_dict['gt_boxes_moco'][:,-1], f'{frame_id}, cluster_label: {cluster_id}, cnts:{cnt}'
        
        if PLOT:
            # After augmenting both views
            V.draw_scenes(points=data_dict["points"][:,:4], gt_boxes=data_dict["gt_boxes"][:,:7])
            V.draw_scenes(points=data_dict["points_moco"][:,:4], gt_boxes=data_dict["gt_boxes_moco"][:,:7])

        # data processor
        # sample points if pointnet backbone
        if not cfg["VOX"]:
            data_dict['points'] = data_processor.sample_points(data_dict['points'], self.cfg["SAMPLE_NUM_POINTS"] )
            data_dict['points_moco'] = data_processor.sample_points(data_dict['points_moco'], self.cfg["SAMPLE_NUM_POINTS"])

        # # shuffle points
        if self.mode == 'train':
            data_dict['points'] = data_processor.shuffle_points(data_dict['points'])
            data_dict['points_moco'] = data_processor.shuffle_points(data_dict['points_moco'])

        # for per point fg,bg prediction
        if self.mode == 'train':
            # If augmentor removes a patch with gt box, remove its gt box and label its points as -1
            data_dict['points'], data_dict['gt_boxes'], _ = data_processor.mask_boxes_with_few_points(data_dict['points'], data_dict['gt_boxes'], pt_cluster_ids=data_dict['points'][:, -1])
            data_dict['points_moco'], data_dict['gt_boxes_moco'], _ = data_processor.mask_boxes_with_few_points(data_dict['points_moco'], data_dict['gt_boxes_moco'], pt_cluster_ids=data_dict['points_moco'][:, -1])

        
        if PLOT:
            # After sampling points and removing empty boxes
            V.draw_scenes(points=data_dict["points"][:,:4], gt_boxes=data_dict["gt_boxes"][:,:7])
            V.draw_scenes(points=data_dict["points_moco"][:,:4], gt_boxes=data_dict["gt_boxes_moco"][:,:7])
        # if vox then transform points to voxels else save points as tensor
        if cfg["VOX"]:
            vox_dict = self.toVox(data_dict["points"]) # xyzil=clusterlabel 
            data_dict["vox"] = vox_dict

            vox_dict = self.toVox(data_dict["points_moco"])
            data_dict["vox_moco"] = vox_dict

        data_dict['gt_boxes_cluster_ids'] = data_dict['gt_boxes'][:,-1]
        data_dict['gt_boxes'] = data_dict['gt_boxes'][:, :8] #xyz,lwh, rz, gt class_index i.e. 1: Vehicle, ...
        data_dict['gt_boxes_moco_cluster_ids'] = data_dict['gt_boxes_moco'][:,-1]
        data_dict['gt_boxes_moco'] = data_dict['gt_boxes_moco'][:, :8]
        
        # Assert that points contain fg points
        assert (data_dict["points"][:,-1] > -1).sum() > 0
        assert (data_dict["points_moco"][:,-1] > -1).sum() > 0

        #get unscaled_lwhz_cluster_id of gt_boxes remaining after augmentation and then keep only those unscaled lwhz
        box_idx=np.where(np.isin(data_dict['unscaled_lwhz_cluster_id'][:,-1], data_dict['gt_boxes_cluster_ids']))[0]
        data_dict['unscaled_lwhz_cluster_id'] = data_dict['unscaled_lwhz_cluster_id'][box_idx]
        assert (data_dict['unscaled_lwhz_cluster_id'][:,-1] -  data_dict['gt_boxes_cluster_ids']).sum() == 0
        
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