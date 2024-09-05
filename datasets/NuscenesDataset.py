import copy
import pickle
import os
from pathlib import Path

import numpy as np
from utils.data_map import *
from datasets.depth_dataset import DepthContrastDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from functools import reduce
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]

def remove_ego_points(points, center_radius=1.0):
    mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
    return points[mask]

class NuscenesDataset(DepthContrastDataset):
    def __init__(self, cfg, pretraining=True, mode='train', logger=None):
        super().__init__(cfg, pretraining=pretraining, mode=mode, logger=logger)
        version = cfg['VERSION']
        self.data_root_path =  self.root_path / f'data/nuscenes/{version}' 

        self.nusc = NuScenes(
            version=version, dataroot=self.data_root_path, verbose=True
        )

        self.sweeps = cfg["SWEEPS"] if pretraining else 1
        self.sweep_dir = cfg["SWEEP_DIR"] if pretraining else None

        skip_ratio= cfg["DATA_SKIP_RATIO"][mode]

        if pretraining and mode == 'train':
            phase_scenes = list( set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT))
        elif pretraining and mode == 'val':
            phase_scenes = CUSTOM_SPLIT
        else:
            #downstream
            phase_scenes = create_splits_scenes()[mode]


        self.list_tokens = []
        
        # create a list of all keyframe scenes
        skip_counter = 0
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0: #skip whole scenes so it is scene_sampling_interval
                    self.create_list_of_tokens(scene)
        
        if len(self.list_tokens)==0:
            # add only one scene
            # scenes with all labels (parametrizing split) "scene-0392", "scene-0517", "scene-0656", "scene-0730", "scene-0738"
            for scene_idx in range(len(self.nusc.scene)):
                scene = self.nusc.scene[scene_idx]
                if scene["name"] in phase_scenes and scene["name"] in ["scene-0392"]:

                    current_sample_token = scene["first_sample_token"]

                    # Loop to get all successive keyframes
                    list_data = []
                    while current_sample_token != "":
                        current_sample = self.nusc.get("sample", current_sample_token)
                        list_data.append(current_sample["data"]["LIDAR_TOP"])
                        current_sample_token = current_sample["next"]

                    # Add new scans in the list
                    self.list_tokens.extend(list_data)
        
        self.logger.add_line(f'Total Nuscenes samples loaded: {len(self.list_tokens)} for split {mode}')

    
    def create_list_of_tokens(self, scene):
        # Get first in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            self.list_tokens.append(current_sample["data"]["LIDAR_TOP"])
            current_sample_token = current_sample["next"]  
    
    def __len__(self):
        return len(self.list_tokens)
    
    def get_cluster_labels(self, sequence_name, sample_idx):
        label_file = self.cluster_root_path / sequence_name / ('%04d.npy' % sample_idx)
        labels = np.fromfile(label_file, dtype=np.float16)
        return labels
    
    def get_item_pretrain(self, index):
        ref_lidar_token = self.list_tokens[index]
        ref_lidar_sd = self.nusc.get("sample_data", ref_lidar_token)

        pcl_path = os.path.join(self.nusc.dataroot, ref_lidar_sd["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        points = pc_original.points.T[:,:4]

        if self.sweeps > 1:
            ref_cs_rec = self.nusc.get('calibrated_sensor', ref_lidar_sd['calibrated_sensor_token']) # transl and rot of LIDAR_TOP wrt ego vehicle frame i.e. T_ego_lidar
            ref_pose_rec = self.nusc.get('ego_pose', ref_lidar_sd['ego_pose_token']) # pose of ego wrt global frame 
            
            # Homogeneous transform from ego car frame to reference frame
            ref_from_car = transform_matrix(
                ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True
            ) # Lidar_frame_from_car i.e. T_lidar_ego

            # Homogeneous transformation matrix from global to _current_ ego car frame
            car_from_global = transform_matrix(
                ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True,
            ) #T_ego_global

            sweeps_points = [points] # get current keyframe and 10 frames before with timelag to current keyframe and transform to ref/current lidar keyframe
            # ground_masks = [estimate_ground(pc_ref, show_plots=show_plots)]
            curr_lidar_sd = ref_lidar_sd

            while len(sweeps_points) < self.sweeps:
                if ref_lidar_sd['prev'] == '':
                    curr_lidar_sd = self.nusc.get('sample_data', curr_lidar_sd['next']) #get next frame data
                else:
                    curr_lidar_sd = self.nusc.get('sample_data', curr_lidar_sd['prev']) #get next frame data
                
                # Get past pose
                current_pose_rec = self.nusc.get('ego_pose', curr_lidar_sd['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                ) # T_global_prevego

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = self.nusc.get(
                    'calibrated_sensor', curr_lidar_sd['calibrated_sensor_token']
                )
                car_from_current = transform_matrix(
                    current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                ) #T_prevego_prevlidar

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current]) # transf ref LIDAR keyframe from prev lidar frame

                pcl_path = os.path.join(self.nusc.dataroot, curr_lidar_sd["filename"])
                pc_original = LidarPointCloud.from_file(pcl_path)
                pc_curr = pc_original.points.T[:,:4]

                # ground_mask = estimate_ground(pc_curr, show_plots=show_plots)
                # ground_masks.append(ground_mask)

                pc_curr = remove_ego_points(pc_curr).T
                num_points = pc_curr.shape[1]
                pc_curr[:3, :] = tm.dot(np.vstack((pc_curr[:3, :], np.ones(num_points))))[:3, :]
                sweeps_points.append(pc_curr.T)

            points = np.concatenate(sweeps_points, axis=0)


        #Get cluster labels
        cluster_label_file = self.data_root_path / f'clustered_sweep_{self.sweeps}_{self.sweep_dir}' / f'{ref_lidar_token}.npy'
        pt_cluster_labels = np.fromfile(cluster_label_file, dtype=np.float16)  
        assert points.shape[0] == pt_cluster_labels.shape[0], f'Missing cluster labels for {ref_lidar_token}!!!!!!!!'

        #Get approx bboxes
        approx_boxes_file = self.data_root_path / f'clustered_sweep_{self.sweeps}_{self.sweep_dir}' / f'approx_boxes_{ref_lidar_token}.npy'
        approx_boxes = np.fromfile(approx_boxes_file, dtype=np.float32).reshape((-1,16))

        #append class id as 8th entry in gt boxes and cluster label as 9th
        box_cluster_labels =  approx_boxes[:, -1]
        gt_classes = np.array([1]*approx_boxes.shape[0])
        gt_boxes = np.hstack([approx_boxes[:,:7], gt_classes.reshape(-1, 1).astype(np.float32), box_cluster_labels.reshape(-1, 1)])
        
        # Set clusters as background if their groundtruth box is not available
        for lbl in np.unique(pt_cluster_labels):
            if lbl == -1.0:
                continue
            if lbl not in box_cluster_labels:
                pt_cluster_labels[pt_cluster_labels==lbl] = -1

        points = np.hstack([points, pt_cluster_labels.reshape(-1, 1)]) #xyzil

        input_dict = {
            'points': points,
            'gt_boxes':  gt_boxes,
            'frame_id': ref_lidar_token
            }

        data_dict = self.prepare_data_pretrain(data_dict=input_dict)

        return data_dict

    def get_item_downstream(self, index):
        lidar_token = self.list_tokens[index]
        pointsensor = self.nusc.get("sample_data", lidar_token)
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        points = LidarPointCloud.from_file(pcl_path).points.T
        # get the points (4th coordinate is the point intensity)
        #TODO check if necessary
        lidarseg_labels_filename = os.path.join(
                self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"]
            )
        points_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
        points_labels = np.vectorize(nuscenes_labels_map.get)(points_labels)
        points = np.hstack([points[:,:4], points_labels.reshape(-1, 1)]) #xyzi, seglabel

        input_dict = {
            'points': points, #xyzi, seglabel
            'frame_id': lidar_token
            }

        data_dict = self.prepare_data_downstream(data_dict=input_dict)

        return data_dict

    def __getitem__(self, index):
        if self.pretraining:
            data_dict = self.get_item_pretrain(index)
        else:
            data_dict = self.get_item_downstream(index)
        return data_dict