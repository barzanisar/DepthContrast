import numpy as np
import torch
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils

# np.random.seed(100)

def mask_points_outside_range(points, point_cloud_range):
    keep_mask = common_utils.mask_points_by_range(points, point_cloud_range)
    return points[keep_mask]


def mask_boxes_outside_range(gt_boxes, point_cloud_range):
    # Only do this for training
    # return mask of inside range boxes
    keep_mask = box_utils.mask_boxes_outside_range_numpy(
        gt_boxes, point_cloud_range, min_num_corners=1
    )
    return keep_mask

# def remove_points_in_gt_boxes(points, gt_boxes):
#     # TODO: check for empty gt_boxes
#     box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
#                 torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
#                 torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
#             ).long().squeeze(dim=0).cpu().numpy()
#     num_obj = gt_boxes.shape[0]

#     keep_mask = np.ones(len(points), dtype=np.bool_)
#     for i in range(num_obj):
#         keep_mask[box_idxs_of_pts == i] = False
    
#     return points[keep_mask]
    
def mask_boxes_with_few_points(points, gt_boxes):
    # Only do this for training, this can happen after dropping patches
    # return mask for selecting valid boxes
    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
    
    num_obj = gt_boxes.shape[0]
    box_keep_mask = [] #[len(points[box_idxs_of_pts == i]) > 5 for i in range(num_obj)]

    for i in range(num_obj):
        obj_points_mask = box_idxs_of_pts == i
        if len(points[obj_points_mask]) <= 5:
            box_keep_mask.append(False)

            # remove cluster ids for these points
            box_idxs_of_pts[obj_points_mask] = -1 
        else:
            box_keep_mask.append(True)

    return box_keep_mask, box_idxs_of_pts

def shuffle_points(points):
        shuffle_idx = np.random.permutation(points.shape[0])
        return points[shuffle_idx]

def sample_points(points, num_points):
    if num_points < len(points):
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        choice = []
        # if num_far_points < (num_points = 16384) i.e. sparse point cloud at greater distance then take all far_idx and the remainder numpoints take from near ids  
        if num_points > len(far_idxs_choice):
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
        else: 
            # if point cloud has atleast 16384 points in far distances then randomly choose 16384 points
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, num_points, replace=False)
        np.random.shuffle(choice)
    else:
        # if pc is super sparse that it has less than 16384 points to begin with
        choice = np.arange(0, len(points), dtype=np.int32)
        if num_points > len(points):
            if num_points - len(points) > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=True)
            else:
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
     
    return points[choice]