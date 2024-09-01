from nuscenes import NuScenes
from pathlib import Path
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import os
import numpy as np
from functools import reduce
from utils.estimate_ground import estimate_ground
from utils.cluster_utils import cluster, filter_labels, get_continuous_labels, REJECT
from utils.pcd_preprocess import visualize_pcd_clusters, visualize_selected_labels
from utils.approx_bbox_utils import fit_box, show_bev_boxes
# from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging

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

parser = argparse.ArgumentParser(description='Cluster Nuscenes')
parser.add_argument('--start_scene_idx', type=int, default=0, help='[0-600]')
parser.add_argument('--end_scene_idx', type=int, default=100, help='[0-600]')
parser.add_argument('--sweeps', type=int, default=1, help='sweeps')
parser.add_argument('--version', type=str, default='v1.0-trainval', help='version')
parser.add_argument('--eps', type=float, default=0.7, help='dbscan eps')
parser.add_argument('--num_workers', type=int, default=2, help='num workers')


def remove_ego_points(points, center_radius=1.0):
    mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
    return points[mask]

def get_sweep_points_ground_masks(nusc, max_sweeps, ref_lidar_sd, show_plots):
        pcl_path = os.path.join(nusc.dataroot, ref_lidar_sd["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc_ref = pc_original.points.T[:,:4]

        ref_cs_rec = nusc.get('calibrated_sensor', ref_lidar_sd['calibrated_sensor_token']) # transl and rot of LIDAR_TOP wrt ego vehicle frame i.e. T_ego_lidar
        ref_pose_rec = nusc.get('ego_pose', ref_lidar_sd['ego_pose_token']) # pose of ego wrt global frame 
        
        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True
        ) # Lidar_frame_from_car i.e. T_lidar_ego

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True,
        ) #T_ego_global

        sweeps = [pc_ref] # get current keyframe and 10 frames before with timelag to current keyframe and transform to ref/current lidar keyframe
        # ground_masks = [estimate_ground(pc_ref, show_plots=show_plots)]
        curr_lidar_sd = ref_lidar_sd

        while len(sweeps) < max_sweeps:
            if ref_lidar_sd['prev'] == '':
                curr_lidar_sd = nusc.get('sample_data', curr_lidar_sd['next']) #get next frame data
            else:
                curr_lidar_sd = nusc.get('sample_data', curr_lidar_sd['prev']) #get next frame data
            
            # Get past pose
            current_pose_rec = nusc.get('ego_pose', curr_lidar_sd['ego_pose_token'])
            global_from_car = transform_matrix(
                current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
            ) # T_global_prevego

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get(
                'calibrated_sensor', curr_lidar_sd['calibrated_sensor_token']
            )
            car_from_current = transform_matrix(
                current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
            ) #T_prevego_prevlidar

            tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current]) # transf ref LIDAR keyframe from prev lidar frame

            pcl_path = os.path.join(nusc.dataroot, curr_lidar_sd["filename"])
            pc_original = LidarPointCloud.from_file(pcl_path)
            pc_curr = pc_original.points.T[:,:4]

            # ground_mask = estimate_ground(pc_curr, show_plots=show_plots)
            # ground_masks.append(ground_mask)

            pc_curr = remove_ego_points(pc_curr).T
            num_points = pc_curr.shape[1]
            pc_curr[:3, :] = tm.dot(np.vstack((pc_curr[:3, :], np.ones(num_points))))[:3, :]
            sweeps.append(pc_curr.T)

        
        points = np.concatenate(sweeps, axis=0)
        # ground_masks = np.concatenate(ground_masks, axis=0)
        ground_masks = estimate_ground(points, show_plots=show_plots)

        return points, ground_masks


def run(scene_idx, nusc, phase_scenes, args, save_dir, show_plots=False):
    scene = nusc.scene[scene_idx]
    if scene["name"] in phase_scenes:
        logging.info(f"Processing Scene {scene_idx}: {scene['name']} \n\n")
        # loop over all keyframes in one scene
        current_sample_token = scene["first_sample_token"]
        while current_sample_token != "":
            current_sample = nusc.get("sample", current_sample_token)
            current_sample_data = current_sample["data"]
            ref_lidar_sd = nusc.get("sample_data", current_sample_data["LIDAR_TOP"])
            ref_lidar_token = ref_lidar_sd['token']
            
            # Get next keyframe for next iter of while
            current_sample_token = current_sample["next"]

            #check if already processed 
            saved_path = save_dir / f'approx_boxes_{ref_lidar_token}.npy'
            if saved_path.exists():
                try:
                    boxes = np.fromfile(saved_path, dtype=np.float32).reshape((-1,16))
                    if boxes.shape[0]:
                        logging.info(f"Already processed scene {scene_idx}: {scene['name']}, Keyframe: {ref_lidar_token} \n\n")
                        continue
                except:
                    pass

            logging.info(f"Processing scene {scene_idx}: {scene['name']}, Keyframe: {ref_lidar_token} \n\n")
            

            # get points and ground masks for one keyframe
            xyzi, ground_mask = get_sweep_points_ground_masks(nusc, args.sweeps, ref_lidar_sd, show_plots)
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

            # if show_plots:
            #     print(f'Final clusters')
            #     visualize_pcd_clusters(xyzi[:,:3], labels.reshape((-1,1)))
            
            save_path = save_dir / f'{ref_lidar_token}.npy'
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
            save_path = save_dir / f'approx_boxes_{ref_lidar_token}.npy'
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
    root = parent_dir / 'data/nuscenes'
    # start_scene_idx = 0, 100, 200, 300, 400, 500
    # end_scene_idx = 100, 200, 300, 400, 500, 600
    show_plots = False
    eps_name = str(args.eps).replace('.', 'p')
    save_dir = parent_dir / 'data/nuscenes' / args.version / f'clustered_sweep_{args.sweeps}_eps{eps_name}'
    os.makedirs(save_dir, exist_ok=True)
    if args.num_workers == -1:
        args.num_workers = mp.cpu_count() - 3

    nusc = NuScenes(version=args.version, dataroot=f'{root}/{args.version}', verbose=True)

    if args.version == 'v1.0-mini':
        phase_scenes = create_splits_scenes()["mini_train"]
        scene_list = [scene_idx for scene_idx in range(len(nusc.scene))]
    else:
        phase_scenes = list(set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT))
        scene_list = [scene_idx for scene_idx in range(args.start_scene_idx, args.end_scene_idx)]

    logging.info(f'Length of nuscenes scene: {len(nusc.scene)}')

    # if args.num_workers > 0:
    #     run_func = partial(run, nusc=nusc, phase_scenes=phase_scenes, args=args, save_dir=save_dir, show_plots=show_plots)
    #     with mp.Pool(args.num_workers) as p:
    #         res = list(tqdm(p.imap(run_func, scene_list), total=len(scene_list)))
    # else:
    run_func = partial(run, nusc=nusc, phase_scenes=phase_scenes, args=args, save_dir=save_dir, show_plots=show_plots)
    for scene_idx in tqdm(scene_list): #len(nusc.scene)
        run_func(scene_idx=scene_idx)








    

    
































if __name__ == '__main__':
    main()