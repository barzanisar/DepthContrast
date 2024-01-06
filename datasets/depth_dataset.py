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
from datasets.transforms.LiDAR_augmentation import LiDAR_aug_manager
from datasets.features import global_descriptors
from datasets.collators.sparse_collator import point_set_to_coord_feats

from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
# from utils.pcd_preprocess import visualize_selected_labels
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_pcd_clusters(points, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    
    colors = np.zeros((len(labels), 4))
    flat_indices = np.unique(labels)
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("prism")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

    for idx in range(len(flat_indices)):
        colors[labels == flat_indices[int(idx)]] = colors_instance[int(idx)]

    colors[labels == -1] = [0.,0.,0.,0.]

    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
    geometries = [pcd]

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(axis_pcd)
    o3d.visualization.draw_geometries(geometries)
    
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

        if 'LIDAR_AUG' in cfg:
            self.lidar_aug = LiDAR_aug_manager(self.root_path, cfg['LIDAR_AUG'])
            # self.lidars = cfg['LIDAR_AUG']['lidars']
            # self.lidar_augs = {}
            # for lidar in self.lidars:
            #     aug_cfg_path = self.root_path / 'configs' / 'Lidar_configs' / f'LiDAR_config_{lidar}.yaml' 
            #     self.lidar_augs[lidar] = LiDAR_aug(aug_cfg_path)
    
        self.data_augmentor = data_augmentor.DataAugmentor(self.cfg["POINT_TRANSFORMS"])


        #### Add the voxelizer here
        self.grid_size = None
        self.voxel_size = None
        self.depth_downsample_factor = None 
        if cfg["INPUT"] == 'voxels':
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
            
            if self.cfg['downstream_task'] == 'detection':
                #Reappend class_idx
                data_dict["gt_boxes"] = np.hstack([data_dict["gt_boxes"], gt_classes_idx])
        
        if cfg['INPUT'] == 'points':
            data_dict['points'] = data_processor.sample_points(data_dict['points'], self.cfg["SAMPLE_NUM_POINTS"] )

        if self.mode == 'train':
            data_dict['points'] = data_processor.shuffle_points(data_dict['points'])
            
        if cfg['INPUT'] == 'voxels':
            vox_dict = self.toVox(data_dict["points"]) # xyzil=seg label 
            data_dict["vox"] = vox_dict

        if cfg['INPUT'] == 'sparse_tensor':
            data_dict['voxel_coords'], sampled_pts_feats_xyzi, seg_labels = point_set_to_coord_feats(data_dict["points"][:,:-1], data_dict["points"][:,-1], self.cfg["RESOLUTION"], self.cfg["SAMPLE_NUM_POINTS"])
            data_dict["points"] = np.hstack([sampled_pts_feats_xyzi, seg_labels[:,None]])

        
        #Only needed this when downstream task is detection
        # Remove gt box if not enough pts but points stay the same
        # If augmentor removes a patch with gt box, remove its gt box
        if self.cfg['downstream_task'] == 'detection':
            if self.mode == 'train':
                data_dict['points'], data_dict['gt_boxes'], data_dict['pt_wise_gtbox_idxs'] = data_processor.mask_boxes_with_few_points(data_dict['points'], data_dict['gt_boxes'],  numpts=20)
            else: 
                data_dict['pt_wise_gtbox_idxs'] = data_processor.find_pt_wise_gtbox_idx(data_dict['points'], data_dict['gt_boxes'])
        else:
             data_dict.pop('gt_boxes')   
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

        #Create different views / augmentation
        data_dict['points_moco'] = np.copy(data_dict["points"])
        data_dict['gt_boxes_moco'] = np.copy(data_dict["gt_boxes"])

        gt_classes_idx = data_dict["gt_boxes"][:,-2].reshape(-1,1)
        gt_cluster_ids = data_dict["gt_boxes"][:,-1].reshape(-1,1)
        data_dict['unscaled_lwhz_cluster_id'] = np.hstack([data_dict["gt_boxes"][:,3:6], data_dict["gt_boxes"][:,2].reshape(-1,1), gt_cluster_ids])
        
        ################################ Visualize iou knn and threshold filtering #######################
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

        ################################ Extract Shape descriptors #######################

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
            data_dict['shape_desc_cluster_ids'] = cluster_ids_for_shape_descs

        ################################ Visualize knn for shape desc #######################
    
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
         #######################################################

        # if 'LIDAR_AUG' in cfg and cfg['LIDAR_AUG']['method'] == 'before':
        #     x_src = data_dict["points"][:,0]
        #     y_src = data_dict["points"][:,1]
        #     z_src = data_dict["points"][:,2] - 1.73
        #     l_src = data_dict["points"][:,-1]

        #     visualize_pcd_clusters(np.vstack((x_src, y_src, z_src)).T, l_src)
            
        #     azimuth_bounds = np.array([[0, np.pi/4], [3*np.pi/4, np.pi]])#np.random.uniform(0, 2*np.pi, (cfg['LIDAR_AUG']['cuts'],2))
        #     delta_azimuth = azimuth_bounds.max(axis=1) - azimuth_bounds.min(axis=1)
        #     azimuth_bounds = azimuth_bounds[delta_azimuth>0]
        #     azimuth_src = np.arctan2(y_src, x_src)
        #     azimuth_src[azimuth_src < 0] += 2*np.pi

        #     sample_pts = []
        #     sample_labels = []
        #     for i in range(cfg['LIDAR_AUG']['samples']):
        #         lidar_pattern = 'o128' #self.lidars[i] #np.random.choice(self.lidars)
        #         print(lidar_pattern, ' chosen')
        #         pts, labels = self.lidar_augs[lidar_pattern].generate_frame(np.vstack((x_src, y_src, z_src)).T, l_src)
        #         visualize_pcd_clusters(pts, labels)
        #         sample_pts.append(pts)
        #         sample_labels.append(labels)
            
        #     for i in range(azimuth_bounds.shape[0]):
        #         lidar_ind = i%len(sample_pts)
        #         pts = sample_pts[lidar_ind]
        #         labels = sample_labels[lidar_ind]
        #         azimuth = np.arctan2(pts[:,1], pts[:,0])
        #         azimuth[azimuth < 0] += 2*np.pi
                
        #         min_azimuth = azimuth_bounds[i].min()
        #         max_azimuth = azimuth_bounds[i].max()
        #         delta_azimuth = max_azimuth - min_azimuth

        #         if delta_azimuth > np.pi:
        #             cond = np.logical_or(azimuth < min_azimuth, azimuth > max_azimuth)
        #             cond_src = np.logical_or(azimuth_src < min_azimuth, azimuth_src > max_azimuth)
        #         else:
        #             cond = np.logical_and(azimuth < max_azimuth, azimuth > min_azimuth)
        #             cond_src = np.logical_and(azimuth_src < max_azimuth, azimuth_src > min_azimuth)

        #         cond_src = np.logical_not(cond_src)
        #         x_src = np.concatenate((x_src[cond_src], pts[cond,0]))
        #         y_src = np.concatenate((y_src[cond_src], pts[cond,1]))
        #         z_src = np.concatenate((z_src[cond_src], pts[cond,2]))
        #         l_src = np.concatenate((l_src[cond_src], labels[cond]))
        #         azimuth_src = np.arctan2(y_src, x_src)
        #         azimuth_src[azimuth_src < 0] += 2*np.pi
        #         #b=1

        #         visualize_pcd_clusters(np.vstack((x_src, y_src, z_src)).T, l_src)

        #     visualize_pcd_clusters(np.vstack((x_src, y_src, z_src)).T, l_src)
        #     b=1


        if 'LIDAR_AUG' in cfg:
            # data_dict["points"][:,:3], data_dict["points"][:,-1] = self.lidar_aug.generate_frame(data_dict["points"][:,:3], data_dict["points"][:,-1], 
            #                             data_dict["gt_boxes"], gt_cluster_ids, lidar='v32')
            
            # data_dict["points_moco"][:,:3], data_dict["points_moco"][:,-1] = self.lidar_aug.generate_frame(data_dict["points_moco"][:,:3], data_dict["points_moco"][:,-1], 
            #                             data_dict["gt_boxes_moco"], gt_cluster_ids) #, lidar='v32'

            # visualize_pcd_clusters(data_dict["points_moco"][:,:3], data_dict["points_moco"][:,-1])
            pts, l, src_pts, src_l = self.lidar_aug.generate_frame(data_dict["points_moco"][:,:3], data_dict["points_moco"][:,-1], 
                                        data_dict["gt_boxes_moco"], gt_cluster_ids, lidar='v32') #, lidar='v32'
            
            # visualize_pcd_clusters(src_pts, src_l)
            # visualize_pcd_clusters(pts, l)
            visualize_pcd_clusters(np.vstack((pts, src_pts)),  
                                   np.concatenate((-1*np.ones(pts.shape[0]), np.zeros(src_pts.shape[0]))))

            b=1

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
        if cfg['INPUT'] == 'points':
            data_dict['points'] = data_processor.sample_points(data_dict['points'], self.cfg["SAMPLE_NUM_POINTS"] )
            data_dict['points_moco'] = data_processor.sample_points(data_dict['points_moco'], self.cfg["SAMPLE_NUM_POINTS"])

        # # shuffle points
        if self.mode == 'train':
            data_dict['points'] = data_processor.shuffle_points(data_dict['points'])
            data_dict['points_moco'] = data_processor.shuffle_points(data_dict['points_moco'])

        # for per point fg,bg prediction
        if self.mode == 'train':
            # If augmentor removes a patch with gt box, remove its gt box and label its points as -1
            data_dict['points'], data_dict['gt_boxes'], _ = data_processor.mask_boxes_with_few_points(data_dict['points'], data_dict['gt_boxes'], pt_cluster_ids=data_dict['points'][:, -1], numpts=20)
            data_dict['points_moco'], data_dict['gt_boxes_moco'], _ = data_processor.mask_boxes_with_few_points(data_dict['points_moco'], data_dict['gt_boxes_moco'], pt_cluster_ids=data_dict['points_moco'][:, -1], numpts=20)

        
        if PLOT:
            # After sampling points and removing empty boxes
            V.draw_scenes(points=data_dict["points"][:,:4], gt_boxes=data_dict["gt_boxes"][:,:7])
            V.draw_scenes(points=data_dict["points_moco"][:,:4], gt_boxes=data_dict["gt_boxes_moco"][:,:7])
        # if vox then transform points to voxels else save points as tensor
        if cfg['INPUT'] == 'voxels':
            vox_dict = self.toVox(data_dict["points"]) # xyzil=clusterlabel 
            data_dict["vox"] = vox_dict

            vox_dict = self.toVox(data_dict["points_moco"])
            data_dict["vox_moco"] = vox_dict

        if cfg['INPUT'] == 'sparse_tensor':
            data_dict['voxel_coords'], feats, cluster_p = point_set_to_coord_feats(data_dict["points"][:,:-1], data_dict["points"][:,-1], self.cfg["RESOLUTION"], self.cfg["SAMPLE_NUM_POINTS"], frame_id=data_dict['frame_id'])
            data_dict["points"] = np.hstack([feats, cluster_p[:,None]])

            data_dict['voxel_coords_moco'], feats_moco, cluster_p_moco = point_set_to_coord_feats(data_dict["points_moco"][:,:-1], data_dict["points_moco"][:,-1], self.cfg["RESOLUTION"], self.cfg["SAMPLE_NUM_POINTS"], frame_id=data_dict['frame_id'])
            data_dict["points_moco"] = np.hstack([feats_moco, cluster_p_moco[:,None]])
            assert data_dict["points"].dtype=='float32', f'points dtype is not float32, dtype is {data_dict["points"].dtype}'
            assert data_dict["points_moco"].dtype=='float32', f'points_moco dtype is not float32, dtype is {data_dict["points_moco"].dtype}'


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