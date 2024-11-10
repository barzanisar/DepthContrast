from pathlib import Path
import os
import numpy as np
from functools import reduce
from utils.estimate_ground import estimate_ground
from utils.cluster_utils import cluster, filter_labels, get_continuous_labels, REJECT
from utils.pcd_preprocess import visualize_pcd_clusters, visualize_selected_labels
from utils.approx_bbox_utils import fit_box, show_bev_boxes
#from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging


parser = argparse.ArgumentParser(description='Cluster Semkitti')
parser.add_argument('--start_seq_idx', type=int, default=2, help='[0-9]')
parser.add_argument('--end_seq_idx', type=int, default=9, help='[0-9]')
parser.add_argument('--eps', type=float, default=0.25, help='dbscan eps')


def get_lidar(sample_path):
    lidar_file = Path(sample_path)
    assert lidar_file.exists(), f'{lidar_file} not found'
    pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    return pc 


def run(seq_id, args, root_data_path, root_save_dir, show_plots=False):
    save_dir = root_save_dir / seq_id / 'velodyne'
    os.makedirs(save_dir, exist_ok=True)
    
    point_seq_path = os.path.join(root_data_path, seq_id, 'velodyne')
    point_seq_bin = os.listdir(point_seq_path)
    point_seq_bin.sort()
    points_datapath = [os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

    for points_path in points_datapath:
        xyzi = get_lidar(points_path)

        #check if already processed 
        frame_id = points_path.split('/')[-1].split('.')[0]
        saved_path = save_dir / f'approx_boxes_{frame_id}.npy'
        if saved_path.exists():
            try:
                boxes = np.fromfile(saved_path, dtype=np.float32).reshape((-1,16))
                if boxes.shape[0]:
                    logging.info(f"Already processed seq {seq_id}: {frame_id} \n\n")
                    continue
            except:
                pass

        logging.info(f"Already processed seq {seq_id}: {frame_id} \n\n")

        ground_mask = estimate_ground(xyzi, sensor_height=1.723, show_plots=show_plots)
        num_pts = xyzi.shape[0]

        #Cluster
        labels = cluster(xyzi[:,:3], np.logical_not(ground_mask), eps=args.eps)
        assert labels.shape[0] == num_pts
        # print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')
        if show_plots:
            visualize_pcd_clusters(xyzi[:,:3], labels.reshape((-1,1)))

        #Filter
        new_labels, label_wise_rejection_tag  = filter_labels(xyzi[:,:3], labels,
                                    max_volume=400, min_volume=0.03, 
                                    ground_mask = None)
                                    # max_height_for_lowest_point=2.0, 
                                    # min_height_for_highest_point=0.5,
        
        assert new_labels.shape[0] == num_pts

        if show_plots:
            print(f'After filtering Labels: {np.unique(new_labels).shape[0]}')
            visualize_pcd_clusters(xyzi[:,:3], new_labels.reshape((-1,1)))
        if show_plots:
            for key, val in REJECT.items():
                rejected_labels = np.where(label_wise_rejection_tag == REJECT[key])[0]
                if len(rejected_labels):
                    print(f'rejected_labels: {rejected_labels}')
                    print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
                    visualize_selected_labels(xyzi[:,:3], labels.flatten(), rejected_labels)


        # #Get continous labels
        labels = get_continuous_labels(new_labels)
        assert labels.shape[0] == num_pts
        # print(f' Final Labels found: {np.unique(labels).shape[0]}')

        if show_plots:
            print(f'Final clusters')
            visualize_pcd_clusters(xyzi[:,:3], labels.reshape((-1,1)))
        
        save_path = save_dir / f'{frame_id}.npy'
        labels = labels.astype(np.float16)
        assert labels.shape[0] == num_pts
        labels.tofile(save_path.__str__())
        # print(f'Saved sample: {ref_lidar_token}')

        #Fit boxes
        approx_boxes_this_pc = np.empty((0, 16)) #cxyz, lwh, heading, bev_corners.flatten(), cluster_label
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_pts_mask = labels==label
            cluster_pc = xyzi[cluster_pts_mask, :]
            assert cluster_pc.shape[0] >= 10

            box, corners, _ = fit_box(cluster_pc, fit_method='closeness_to_edge')
            full_box = np.zeros((1, approx_boxes_this_pc.shape[-1]))
            full_box[0,:7] = box
            full_box[0,7:15] = corners.flatten()
            full_box[0,15] = label

            approx_boxes_this_pc = np.vstack([approx_boxes_this_pc, full_box])
            # [cxy[0], cxy[1], cz, l, w, h, rz, corner0_x, corner0_y, ..., corner3_x, corner3_y, cluster label]
            # corner0-3 are BEV box corners in lidar frame
        
        # Save bboxes for this keyframe
        save_path = save_dir / f'approx_boxes_{frame_id}.npy'
        approx_boxes_this_pc.astype(np.float32).tofile(save_path.__str__())
        # print(f'Saved approx boxes: {ref_lidar_token}')

        if show_plots:
            # show_bev_boxes(xyzi[labels>-1], approx_boxes_this_pc, 'unrefined_approx_boxes')
            V.draw_scenes(xyzi, gt_boxes=None, 
                                ref_boxes=approx_boxes_this_pc[:,:7], ref_labels=None, ref_scores=None, 
                                color_feature=None, draw_origin=True)


def main():
    args = parser.parse_args()
    parent_dir = (Path(__file__) / '../..').resolve() #DepthContrast
    root = parent_dir / 'data/semantic_kitti' 

    seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10' ]
    new_seq_list = [seq_list[i] for i in range(args.start_seq_idx, args.end_seq_idx+1)]


    show_plots = False
    eps_name = str(args.eps).replace('.', 'p')
    save_dir = root / f'dataset_clustered_eps{eps_name}' / 'sequences'
    os.makedirs(save_dir, exist_ok=True)

    run_func = partial(run, args=args, root_data_path = root / 'dataset/sequences',  root_save_dir=save_dir, show_plots=show_plots)
    for seq_id in tqdm(new_seq_list):
        run_func(seq_id=seq_id)


if __name__ == '__main__':
    main()