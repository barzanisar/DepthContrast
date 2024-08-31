
import yaml
import numpy as np
import torch
from torch_scatter import scatter_min
    
class LiDAR_aug_manager:
    def __init__(self, root_path, lidar_aug_cfg):
        self.cfg = lidar_aug_cfg
        self.lidars = lidar_aug_cfg['lidars']
        self.lidar_augs = {}
        self.lidar_height = lidar_aug_cfg['lidar_height']  if 'lidar_height' in lidar_aug_cfg else 1.73
        self.random_lidar_height = lidar_aug_cfg.get('random_lidar_height', False)

        for lidar in self.lidars:
            aug_cfg_path = root_path / 'configs' / 'Lidar_configs' / f'LiDAR_config_{lidar}.yaml' 
            self.lidar_augs[lidar] = LiDAR_aug(aug_cfg_path)
    
    def generate_frame_with_gt_boxes(self, src_pc, gt_boxes_lidar, lidar=None, readjust_height=True, lidar_height=None):
        if lidar is None:
            lidar = np.random.choice(self.lidars, p=self.cfg['prob'])
        
        # shift pc down by the height of a car 
        # bcz waymo dataset has lidar origin on the ground
        if lidar_height is None:
            lidar_height = self.lidar_height 
            if self.random_lidar_height:
                lidar_height += np.random.uniform(-0.25, 0.02)
        shifted_pc = np.copy(src_pc)
        shifted_pc[:, 2] -= lidar_height

        shifted_gt_boxes = np.copy(gt_boxes_lidar)
        shifted_gt_boxes[:, 2] -= lidar_height

        
        pts, gt_boxes = self.lidar_augs[lidar].generate_frame_with_gt_boxes(shifted_pc, 
                                                                          shifted_gt_boxes)
            
        if readjust_height:
            pts[:,2] += lidar_height
            gt_boxes[:,2] += lidar_height

        return pts, gt_boxes
    
    # def generate_frame(self, src_pc, src_labels, lidar=None, readjust_height=True):
    #     if lidar is None:
    #         lidar = np.random.choice(self.lidars, p=self.cfg['prob'])
        
    #     # shift pc down by the height of a car 
    #     # bcz waymo dataset has lidar origin on the ground
    #     shifted_pc = np.copy(src_pc)
    #     shifted_pc[:, 2] -= self.lidar_height
    #     src_p, src_l = self.lidar_augs[lidar].generate_frame_src(shifted_pc, src_labels)

    #     if readjust_height:
    #         src_p[:,2] += self.lidar_height

    #     return src_p, src_l
    
    def generate_lidar_mix_frame_with_gt_boxes(self, src_pc, src_gt_boxes, lidars=None):
        if lidars is None:
            lidars = self.lidars
        
        lidar_height = self.lidar_height 
        if self.random_lidar_height:
            lidar_height += np.random.uniform(-0.25, 0.02)

        n_crops = self.cfg['n_crops']
        all_frames = []
        all_gt_boxes = []

        for lidar in self.lidars:
            pts, gt_boxes = self.generate_frame_with_gt_boxes(src_pc, src_gt_boxes, lidar, readjust_height=False, lidar_height=lidar_height)
            assert (src_gt_boxes[:,-1] - gt_boxes[:,-1]).sum() == 0
            all_frames.append(pts)
            all_gt_boxes.append(gt_boxes)
        
        azimuth_bound_1 = np.random.uniform(0, 2*np.pi, n_crops)
        delta_azimuths = np.random.uniform(np.deg2rad(10), np.pi/2, n_crops)
        azimuth_bound_2 = (azimuth_bound_1 + delta_azimuths) % (2*np.pi)
        azimuth_max_bounds = np.max((azimuth_bound_1,azimuth_bound_2), axis=0)
        azimuth_min_bounds = np.min((azimuth_bound_1,azimuth_bound_2), axis=0)
        delta_azimuths = azimuth_max_bounds - azimuth_min_bounds #[0, 2pi]

        pc = np.copy(src_pc)
        pc[:,2] -= lidar_height
        azimuth_src = np.arctan2(pc[:,1], pc[:,0])
        azimuth_src[azimuth_src < 0] += 2*np.pi #to make azimuth 0 to 2pi

        gtb = np.copy(src_gt_boxes)
        gtb[:,2] -= lidar_height
        # gtb_azimuth_src = np.arctan2(gtb[:,1], gtb[:,0])
        # gtb_azimuth_src[gtb_azimuth_src < 0] += 2*np.pi #to make azimuth 0 to 2pi

        for i in range(delta_azimuths.shape[0]):
            lidar_ind = i%len(lidars)
            pts = all_frames[lidar_ind]
            azimuth = np.arctan2(pts[:,1], pts[:,0])
            azimuth[azimuth < 0] += 2*np.pi

            gtb_this = all_gt_boxes[lidar_ind]
            gtb_azimuth = np.arctan2(gtb_this[:,1], gtb_this[:,0])
            gtb_azimuth[gtb_azimuth < 0] += 2*np.pi

            min_azimuth = azimuth_min_bounds[i]
            max_azimuth = azimuth_max_bounds[i]
            if delta_azimuths[i] > np.pi:
                cond = np.logical_or(azimuth < min_azimuth, azimuth > max_azimuth)
                cond_src = np.logical_or(azimuth_src < min_azimuth, azimuth_src > max_azimuth)

                cond_gtb = np.logical_or(gtb_azimuth < min_azimuth, gtb_azimuth > max_azimuth)
                # cond_gtb_src = np.logical_or(gtb_azimuth_src < min_azimuth, gtb_azimuth_src > max_azimuth)
            else:
                cond = np.logical_and(azimuth < max_azimuth, azimuth > min_azimuth)
                cond_src = np.logical_and(azimuth_src < max_azimuth, azimuth_src > min_azimuth)
                
                cond_gtb = np.logical_and(gtb_azimuth < max_azimuth, gtb_azimuth > min_azimuth)
                # cond_gtb_src = np.logical_and(gtb_azimuth_src < max_azimuth, gtb_azimuth_src > min_azimuth)


            cond_src = np.logical_not(cond_src)
            # cond_gtb_src = np.logical_not(cond_gtb_src)
            gtb[cond_gtb] = gtb_this[cond_gtb]


            # #################### Select unique gt boxes #########################
            # gtb_lbls_this = gtb_this[cond_gtb, -1]
            # gtb_lbls_src = gtb[cond_gtb_src, -1]
            # keep_mask_this = np.ones(gtb_lbls_this.shape[0], dtype=bool)
            # keep_mask_src = np.ones(gtb_lbls_src.shape[0], dtype=bool)
            
            # common_gtb_lbls = np.intersect1d(gtb_lbls_src, gtb_lbls_this)
            # indices_of_common_lbls_in_src = np.where(np.in1d(gtb_lbls_src, common_gtb_lbls))[0]
            # indices_of_common_lbls_in_this = np.where(np.in1d(gtb_lbls_this, common_gtb_lbls))[0]

            # for i, lbl in enumerate(common_gtb_lbls):
            #     num_src_pts = (pc[cond_src][:, -1] == lbl).sum()
            #     num_this_pts = (pts[cond][:, -1] == lbl).sum()
            #     if num_src_pts > num_this_pts:
            #         keep_mask_this[indices_of_common_lbls_in_this[i]] = False
            #     else:
            #         keep_mask_src[indices_of_common_lbls_in_src[i]] = False

            # ####################

            pc = np.vstack([pc[cond_src], pts[cond]])
            # gtb = np.vstack([gtb[cond_gtb_src][keep_mask_src], gtb_this[cond_gtb][keep_mask_this]])

            azimuth_src = np.arctan2(pc[:,1], pc[:,0])
            azimuth_src[azimuth_src < 0] += 2*np.pi
            # gtb_azimuth_src = np.arctan2(gtb[:,1], gtb[:,0])
            # gtb_azimuth_src[gtb_azimuth_src < 0] += 2*np.pi


        pc[:,2] += lidar_height
        gtb[:,2] += lidar_height

        return pc, gtb 
    
    # def generate_lidar_mix_frame(self, src_pc, src_labels, lidars=None):
    #     if lidars is None:
    #         lidars = self.lidars
        
    #     n_crops = self.cfg['n_crops']
    #     all_frames = []
    #     all_labels = []

    #     for lidar in self.lidars:
    #         pts, labels = self.generate_frame(src_pc, src_labels, lidar, readjust_height=False)
    #         all_frames.append(pts)
    #         all_labels.append(labels)
        
    #     azimuth_bound_1 = np.random.uniform(0, 2*np.pi, n_crops)
    #     delta_azimuths = np.random.uniform(np.deg2rad(10), np.pi/2, n_crops)
    #     azimuth_bound_2 = (azimuth_bound_1 + delta_azimuths) % (2*np.pi)
    #     azimuth_max_bounds = np.max((azimuth_bound_1,azimuth_bound_2), axis=0)
    #     azimuth_min_bounds = np.min((azimuth_bound_1,azimuth_bound_2), axis=0)
    #     delta_azimuths = azimuth_max_bounds - azimuth_min_bounds

    #     pc = np.copy(src_pc)
    #     pc[:,2] -= self.lidar_height
    #     l_src = np.copy(src_labels)
    #     azimuth_src = np.arctan2(pc[:,1], pc[:,0])
    #     azimuth_src[azimuth_src < 0] += 2*np.pi

    #     for i in range(delta_azimuths.shape[0]):
    #         lidar_ind = i%len(lidars)
    #         pts = all_frames[lidar_ind]
    #         labels = all_labels[lidar_ind]
    #         azimuth = np.arctan2(pts[:,1], pts[:,0])
    #         azimuth[azimuth < 0] += 2*np.pi

    #         min_azimuth = azimuth_min_bounds[i]
    #         max_azimuth = azimuth_max_bounds[i]
    #         if delta_azimuths[i] > np.pi:
    #             cond = np.logical_or(azimuth < min_azimuth, azimuth > max_azimuth)
    #             cond_src = np.logical_or(azimuth_src < min_azimuth, azimuth_src > max_azimuth)
    #         else:
    #             cond = np.logical_and(azimuth < max_azimuth, azimuth > min_azimuth)
    #             cond_src = np.logical_and(azimuth_src < max_azimuth, azimuth_src > min_azimuth)

    #         cond_src = np.logical_not(cond_src)
    #         pc = np.vstack([pc[cond_src], pts[cond]])
    #         l_src = np.concatenate((l_src[cond_src], labels[cond]))
    #         azimuth_src = np.arctan2(pc[:,1], pc[:,0])
    #         azimuth_src[azimuth_src < 0] += 2*np.pi

    #     pc[:,2] += self.lidar_height

    #     return pc, l_src


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
    
    def generate_gt_boxes(self, gt_boxes):
        
        # Creates range image for the given Lidar config
        map_pcd = torch.as_tensor(gt_boxes, dtype=torch.float)

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

        proj_w = u_/self.Horizontal_res
        proj_h = v_/self.channel

        yaw = (proj_w*2-1)*np.pi

        pitch = np.pi/2-(1.0*fov-proj_h*fov-abs(self.V_fov_min))

        x = d_lidar*torch.sin(pitch)*torch.cos(-yaw)

        y = d_lidar*torch.sin(pitch)*torch.sin(-yaw)

        z = d_lidar*torch.cos(pitch) 

        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)

        point = torch.stack((x,y,z), dim=1).detach().cpu().numpy()

        gt_boxes_new = np.zeros(gt_boxes.shape)
        gt_boxes_new[:, :3] = point
        gt_boxes_new[:,3:] = gt_boxes[:,3:]

        return gt_boxes_new


    
    def spherical_projection(self, pc):
        
        # Creates range image for the given Lidar config
        map_pcd = torch.as_tensor(pc, dtype=torch.float)

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

        # temp_labels = -1*np.ones(temp_distance.shape[0])
        # temp_labels[filled_inds_dst] = map_label[Cond][filled_inds_src].cpu().detach().numpy() 
        
        src_inds = np.arange(x_lidar.shape[0])
        selected_src_inds = src_inds[Cond][filled_inds_src]
        assert filled_inds_dst.sum() == filled_inds_src.shape[0]
        return temp_distance, filled_inds_dst, selected_src_inds

    
    def inverse_projection(self, dist, feats):

        dist = dist.reshape(self.channel,self.Horizontal_res)
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
        

        cond = np.where(np.logical_or(np.isnan(point)[:,0], np.isnan(point)[:,1], np.isnan(point)[:,2]))
        point = np.delete(point, cond, axis=0)

        cond = np.where(np.logical_and(point[:,0] == 0 , point[:,1] == 0, point[:,2] == 0))
        point = np.delete(point, cond, axis=0)
        point = np.hstack([point, feats])

        return point

    def generate_frame_with_gt_boxes(self, pc, gt_boxes):

        new_pc, src_pc_selected = self.generate_frame(pc)
        
        new_gt_boxes = self.generate_gt_boxes(gt_boxes)
        return new_pc, new_gt_boxes

    def generate_frame(self, pc):

        dist, _, selected_src_inds = self.spherical_projection(pc)
        src_pc_selected = pc[selected_src_inds]
        num_feats = src_pc_selected.shape[1] - 3

        new_pc = self.inverse_projection(dist, src_pc_selected[:,3:].reshape((-1, num_feats)))
         
        return new_pc, src_pc_selected
    
    # def generate_frame_src(self, pc, labels):

    #     _, _, _, selected_src_inds = self.spherical_projection(pc, labels)
         
    #     return pc[selected_src_inds], labels[selected_src_inds]


    
   
