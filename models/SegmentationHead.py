import torch
import torch.nn as nn
from models.trunks.mlp import MLP
from criterions.lovasz_softmax import lovasz_softmax
from criterions.focal_loss import FocalLoss
from third_party.OpenPCDet.pcdet.models.backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch

   
class SegmentationClassifierHead(nn.Module):
    def __init__(self, cfg, point_cloud_range, voxel_size):
        nn.Module.__init__(self)

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.in_channels = cfg["CLS_FC"][0]
        self.out_channels = cfg["CLS_FC"][-1]
        self.ignore_index = 0
        self.fc = MLP(cfg["CLS_FC"])
        self.loss_types = cfg['loss_types']
        self.loss_weights =  cfg['loss_weights']
        
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            # weight=cfg['class_weights'],
            # label_smoothing=cfg['label_smoothing']
        )

        self.lov_loss = lovasz_softmax

        self.focalloss = FocalLoss(
            gamma=0.5,
            ignore_index=self.ignore_index,
        )

    def interpolate_from_bev_features(self, keypoints, bev_features, bev_stride):
        """
        Args:
            keypoints: (N 3=xyz)
            bev_features: (C=256, H=188, W=188)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0] # x coord in 1504x1504 grid
        y_idxs = (keypoints[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        # Getting keypoints xy bev grid coord
        x_idxs = x_idxs / bev_stride # x coord in 188x188 BEV grid
        y_idxs = y_idxs / bev_stride

        #bev_features.permute(1, 2, 0)  # (H=188, W=188, C=256)
        point_bev_features = bilinear_interpolate_torch(bev_features.permute(1, 2, 0), x_idxs, y_idxs) #(4096 keypts, bevfeat_dim=256)

        # point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C) (B x 4096, 256)
        return point_bev_features
    
    def forward(self, batch_dict):

        target = batch_dict['seg_labels']

        # Get pt-wise features
        if 'point_features' in batch_dict:
            x=batch_dict['point_features'].permute(0, 2, 1).reshape(-1, self.in_channels)
        elif 'sparse_point_feats' in batch_dict:
            x=batch_dict['sparse_point_feats'].F #(80k x bs = numpts,  96)
        else:
            backbone_3d_bev_feats = batch_dict['spatial_features'] # (6,256,188,188) # unshuffled (B=2, C=128, N num points = 20000)
            backbone_2d_bev_feats = batch_dict['spatial_features_2d'] # (6,512,188,188)
            bev_features = torch.cat([backbone_3d_bev_feats, backbone_2d_bev_feats], dim=1)  # (6,256+512,188,188)
            points = batch_dict['points'] #(N, bxyzi)
            batch_size = batch_dict["batch_size"] # unshuffled

            batch_feats=[]
            target_fg_pts=[]
            for pc_idx in range(batch_size):
                b_mask = points[:,0] == pc_idx
                pc = points[b_mask][:,1:4]
                fg_pts_mask = target[b_mask] > self.ignore_index
                fg_points = pc[fg_pts_mask][:,0:3] #(num fg pts, 3=xyz)
                target_fg_pts.append(target[b_mask][fg_pts_mask])
                
                #interpolate from bev features at the FPS sampled fg keypts
                keypoint_bev_features = self.interpolate_from_bev_features(
                    fg_points, bev_features[pc_idx],
                    bev_stride=batch_dict['spatial_features_stride']
                ) #(num keypts, 512+256=768)
                batch_feats.append(keypoint_bev_features)

            target = torch.cat(target_fg_pts) #(num fg pts)
            x = torch.vstack(batch_feats) #(num fg pts, 768)


        ######### FC layer + segmentation loss #########
        assert target.shape[0] == x.shape[0] #num pts
        assert x.shape[1] == self.in_channels
        class_score_pred = self.fc(x)
        
        target = target.long()
        loss_dict = {}
        
        if 'CELoss' in self.loss_types:
            loss_dict.update(
                CELoss=self.ce_loss(class_score_pred, target) * \
                    self.loss_weights[self.loss_types.index('CELoss')])

        if 'LovLoss' in self.loss_types:
            loss_dict.update(
                LovLoss=self.lov_loss(class_score_pred.softmax(dim=1), target, 
                                        ignore=self.ignore_index) * \
                    self.loss_weights[self.loss_types.index('LovLoss')])

        if 'FocalLoss' in self.loss_types:
            loss_dict.update(
                FocalLoss=self.focalloss(class_score_pred, target) * \
                    self.loss_weights[self.loss_types.index('FocalLoss')])
            
        pred_labels = class_score_pred.max(dim=1)[1] #pred_y 

        return sum(loss_dict.values()), loss_dict, pred_labels, target