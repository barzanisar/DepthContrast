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

class NuscenesDataset(DepthContrastDataset):
    def __init__(self, cfg, pretraining=True, mode='train', logger=None):
        super().__init__(cfg, pretraining=pretraining, mode=mode, logger=logger)
        version = cfg['VERSION']
        self.data_root_path =  self.root_path / f'data/nuscenes/{version}' 
        # if cached_dataset is not None:
        #     self.nusc = cached_nuscenes
        # else:
        self.nusc = NuScenes(
            version=version, dataroot=self.data_root_path, verbose=True
        )

        self.frame_sampling_interval= cfg["FRAME_SAMPLING_INTERVAL"][self.mode]

        self.list_tokens = []
        
        # create a list of all keyframe scenes
        skip_counter = 0
        phase_scenes = create_splits_scenes()[mode]
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % self.frame_sampling_interval == 0:
                    self.create_list_of_tokens(scene)
        
        self.logger.add_line(f'Total Nuscenes samples loaded: {len(self.list_tokens)}')

    
    def create_list_of_tokens(self, scene):
        # Get first in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            next_sample_token = current_sample["next"]
            self.list_tokens.append(current_sample["data"]["LIDAR_TOP"])
            current_sample_token = next_sample_token  
    
    def __len__(self):
        return len(self.list_tokens)
    
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
        data_dict = self.get_item_downstream(index)
        return data_dict