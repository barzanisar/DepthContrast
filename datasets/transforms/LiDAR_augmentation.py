
import yaml
import numpy as np
import torch
# import open3d as o3d
from torch_scatter import scatter_min
# from third_party.OpenPCDet.pcdet.utils import box_utils
    
class LiDAR_aug_manager:
    def __init__(self, root_path, lidar_aug_cfg):
        self.cfg = lidar_aug_cfg
        self.lidars = lidar_aug_cfg['lidars']
        self.lidar_augs = {}
        self.lidar_height = 1.73

        for lidar in self.lidars:
            aug_cfg_path = root_path / 'configs' / 'Lidar_configs' / f'LiDAR_config_{lidar}.yaml' 
            self.lidar_augs[lidar] = LiDAR_aug(aug_cfg_path)
    
    # def denoise_labels(pts, labels, src_pc, src_labels, gt_boxes_lidar, gt_box_labels):
    #     src_pcd = o3d.geometry.PointCloud()
    #     src_pcd.points = o3d.utility.Vector3dVector(src_pc)

    #     query_pts = pts[labels>-1]
    #     query_labels = labels[labels>-1]
    #     corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar[:,:7])
    #     #nboxes, 8 corners, 3coords

    #     labeled_tree = o3d.geometry.KDTreeFlann(src_pcd)
    #     for i in range(query_pts.shape[0]):
    #         pt = query_pts[i]
    #         #Find its neighbors with distance less than 0.2
    #         [_, idx, _] = labeled_tree.search_knn_vector_3d(pt, 10)
    #         if len(idx):
    #             nearest_labels = src_labels[np.asarray(idx)]
    #             label_of_majority = np.bincount(nearest_labels.astype(int)).argmax()
    #             gt_box_of_majority_label = corners_lidar[gt_box_labels == label_of_majority]
    #             in_box = box_utils.in_hull(pt.reshape(1, -1), gt_box_of_majority_label)
    #             if in_box:
    #                 query_labels[i] = label_of_majority
    #             else:
    #                 query_labels[i] = -1
        
    #     pts[labels>-1] = query_pts
    #     labels[labels>-1] = query_labels
    #     return pts, labels
    
    def generate_frame(self, src_pc, src_labels, gt_boxes_lidar, gt_box_labels, lidar=None, readjust_height=True):
        if lidar is None:
            lidar = np.random.choice(self.lidars)
        
        # shift pc down by the height of a car 
        # bcz waymo dataset has lidar origin on the ground
        shifted_pc = np.copy(src_pc)
        shifted_pc[:, 2] -= self.lidar_height
        pts, labels, src_p, src_l = self.lidar_augs[lidar].generate_frame(shifted_pc, src_labels)
        pts[:,2] += self.lidar_height
        src_p[:,2] += self.lidar_height

        # pts, labels = self.denoise_labels(pts, labels, src_pc, src_labels, gt_boxes_lidar, gt_box_labels)

        if not readjust_height:
            pts[:,2] -= self.lidar_height
            src_p[:,2] -= self.lidar_height

        return pts, labels, src_p, src_l
    
    def generate_lidar_mix_frame(self, src_pc, src_labels, gt_boxes_lidar, gt_box_labels, lidars=None):
        if lidars is None:
            lidars = self.lidars
        
        n_crops = self.cfg['n_crops']
        all_frames = []
        all_labels = []

        for lidar in self.lidars:
            pts, labels = self.generate_frame(src_pc, src_labels, gt_boxes_lidar, gt_box_labels, lidar, readjust_height=False)
            all_frames.append(pts)
            all_labels.append(labels)
        
        azimuth_bound_1 = np.random.uniform(0, 2*np.pi, n_crops)
        delta_azimuths = np.random.uniform(np.deg2rad(10), np.pi/2, n_crops)
        azimuth_bound_2 = (azimuth_bound_1 + delta_azimuths) % (2*np.pi)
        azimuth_max_bounds = np.max((azimuth_bound_1,azimuth_bound_2), axis=0)
        azimuth_min_bounds = np.min((azimuth_bound_1,azimuth_bound_2), axis=0)
        delta_azimuths = azimuth_max_bounds - azimuth_min_bounds

        x_src = src_pc[:,0]
        y_src = src_pc[:,1]
        z_src = src_pc[:,2] - self.lidar_height
        l_src = np.copy(src_labels)
        azimuth_src = np.arctan2(y_src, x_src)
        azimuth_src[azimuth_src < 0] += 2*np.pi

        for i in range(delta_azimuths.shape[0]):
            lidar_ind = i%len(lidars)
            pts = all_frames[lidar_ind]
            labels = all_labels[lidar_ind]
            azimuth = np.arctan2(pts[:,1], pts[:,0])
            azimuth[azimuth < 0] += 2*np.pi

            min_azimuth = azimuth_min_bounds[i]
            max_azimuth = azimuth_max_bounds[i]
            if delta_azimuths[i] > np.pi:
                cond = np.logical_or(azimuth < min_azimuth, azimuth > max_azimuth)
                cond_src = np.logical_or(azimuth_src < min_azimuth, azimuth_src > max_azimuth)
            else:
                cond = np.logical_and(azimuth < max_azimuth, azimuth > min_azimuth)
                cond_src = np.logical_and(azimuth_src < max_azimuth, azimuth_src > min_azimuth)

            cond_src = np.logical_not(cond_src)
            x_src = np.concatenate((x_src[cond_src], pts[cond,0]))
            y_src = np.concatenate((y_src[cond_src], pts[cond,1]))
            z_src = np.concatenate((z_src[cond_src], pts[cond,2]))
            l_src = np.concatenate((l_src[cond_src], labels[cond]))
            azimuth_src = np.arctan2(y_src, x_src)
            azimuth_src[azimuth_src < 0] += 2*np.pi

        pts = np.vstack((x_src, y_src, z_src)).T
        pts[:,2] += self.lidar_height

        return pts, l_src


class LiDAR_aug:

    def __init__(self, config_path):
        
        with open(config_path) as file:

            config = yaml.load(file,  Loader=yaml.FullLoader)

            lidar = config['lidar']

            self.channel = int(lidar['channel'])
            self.Horizontal_res = int(lidar['Horizontal_res'])
            self.V_fov_min = np.deg2rad(float(lidar['VFoV_min']))
            self.V_fov_max = np.deg2rad(float(lidar['VFoV_max']))
            
            self.range = [lidar['range_min'],lidar['range_max']]
        
    
    def spherical_projection(self, pc, labels):
        
        # Creates range image for the given Lidar config
        map_pcd = torch.as_tensor(pc, dtype=torch.float)

        map_label = torch.as_tensor(labels, dtype=torch.float)


        x_lidar = map_pcd[:,0]
        y_lidar = map_pcd[:,1]
        z_lidar = map_pcd[:,2]

        torch.cuda.empty_cache()

        d_lidar = torch.sqrt(x_lidar**2+y_lidar**2+z_lidar**2)

        fov = abs(self.V_fov_max) + abs(self.V_fov_min)

        yaw = 0.5*(1-torch.atan2(y_lidar,x_lidar)/np.pi) 

        pitch = (1-(torch.asin(z_lidar/d_lidar)-self.V_fov_min)/fov)
        
        u_ = torch.as_tensor(yaw*self.Horizontal_res, dtype= torch.long)
        v_ = torch.as_tensor(pitch*self.channel, dtype= torch.long)

        ind = v_*self.Horizontal_res+u_
        
        temp_distance = np.inf*torch.ones((self.channel,self.Horizontal_res))

        V_cond = torch.logical_and(v_ >= 0, v_ < self.channel)

        D_cond = torch.logical_and(d_lidar >= self.range[0], d_lidar < self.range[1])

        Cond = torch.logical_and(V_cond,D_cond)

        # valid_pt = torch.where(Cond)[0]
    
        temp_distance, dist_ind = scatter_min(d_lidar[Cond], ind[Cond], out=torch.flatten(temp_distance))
        #temp_distance = temp_distance.reshape(self.channel, self.Horizontal_res)
        filled_inds_dst = dist_ind!=d_lidar[Cond].shape[0] # or temp_distance != np.inf 
        temp_distance[temp_distance == np.inf] = 0
        temp_distance = temp_distance.cpu().numpy()
        filled_inds_src = dist_ind[filled_inds_dst]
        #a = d_lidar[Cond][filled_inds_src] - temp_distance[temp_distance != np.inf]

        temp_labels = -1*np.ones(temp_distance.shape[0])
        temp_labels[filled_inds_dst] = map_label[Cond][filled_inds_src].cpu().detach().numpy() #TODO: bug: should not be -1
        
        src_inds = np.arange(x_lidar.shape[0])
        selected_src_inds = src_inds[Cond][filled_inds_src]
        return temp_distance, temp_labels, filled_inds_dst, selected_src_inds

    
    def inverse_projection(self, dist, label):

        dist = torch.from_numpy(dist)

        fov = abs(self.V_fov_max) + abs(self.V_fov_min)

        w = torch.arange(self.Horizontal_res,  dtype=torch.long)
        
        h = torch.arange(self.channel,  dtype=torch.long)

        hh, ww = torch.meshgrid(h,w)
        
        proj_w = ww/self.Horizontal_res
        
        proj_h = hh/self.channel

        yaw = (proj_w*2-1)*np.pi

        pitch = np.pi/2-(1.0*fov-proj_h*fov-abs(self.V_fov_min))

        x = dist*torch.sin(pitch)*torch.cos(-yaw)

        y = dist*torch.sin(pitch)*torch.sin(-yaw)

        z = dist*torch.cos(pitch) 

        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)

        point = torch.stack((x,y,z), dim=1).detach().cpu().numpy()

        label = label.flatten()

        label = np.delete(label, np.where(np.logical_or(np.isnan(point)[:,0], np.isnan(point)[:,1], np.isnan(point)[:,2])), axis=0)
        
        point = np.delete(point, np.where(np.logical_or(np.isnan(point)[:,0], np.isnan(point)[:,1], np.isnan(point)[:,2])), axis=0)

        label = np.delete(label, np.where(np.logical_and(point[:,0] == 0 , point[:,1] == 0, point[:,2] == 0)), axis=0)
        
        point = np.delete(point, np.where(np.logical_and(point[:,0] == 0 , point[:,1] == 0, point[:,2] == 0)), axis=0)

        return point, label



    def generate_frame(self, pc, labels):

        dist, label, filled_inds_dst, selected_src_inds = self.spherical_projection(pc, labels)
        
        dist = dist.reshape(self.channel,self.Horizontal_res)
        label = label.reshape(self.channel,self.Horizontal_res)
        
        point, label = self.inverse_projection(dist, label)
        
        assert filled_inds_dst.sum() == label.shape[0]
        src_pc_selected = pc[selected_src_inds]
 
        return point, label, src_pc_selected, labels[selected_src_inds]


    
   
