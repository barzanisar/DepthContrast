from logging import raiseExceptions
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from pathlib import Path
import pickle
import argparse
import os

import numpy as np
from tqdm import tqdm
from utils.pcd_preprocess import *
from pcdet.utils import box_utils, calibration_kitti


ROOT_PATH = (Path(__file__) / '../../../../..').resolve() #DepthContrast
DENSE_ROOT = ROOT_PATH / 'data' / 'dense'
KITTI_ROOT = ROOT_PATH / 'data' / 'kitti'
SEMANTIC_KITTI_ROOT = ROOT_PATH / 'data' / 'semantic_kitti'
alpha = 0.45
sensor_type = 'hdl64'
signal_type = 'strongest'

samples = ['2019-01-09_14-54-03,03700',
'2018-12-10_11-38-09,02600',
'2018-02-12_15-55-09,00100',
'2019-01-09_10-52-08,00100',
'2018-02-06_14-40-36,00100',
'2018-12-11_12-53-37,01540',
'2019-01-09_11-33-12,01200',
'2018-12-19_10-59-51,02200',
'2019-01-09_12-55-44,01000',
'2018-02-04_12-53-35,00000',
'2018-02-06_14-21-31,00000',
'2018-02-07_11-56-57,00560']

def o3d_dynamic_radius_outlier_filter(pc: np.ndarray, alpha: float = 0.45, beta: float = 3.0,
                                    k_min: int = 3, sr_min: float = 0.04) -> np.ndarray:
        """
        :param pc:      pointcloud
        :param alpha:   horizontal angular resolution of the lidar
        :param beta:    multiplication factor
        :param k_min:   minimum number of neighbors
        :param sr_min:  minumum search radius

        :return:        mask [False = snow, True = no snow]
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        num_points = len(pcd.points)

        # initialize mask with False
        mask = np.zeros(num_points, dtype=bool)

        k = k_min + 1

        kd_tree = o3d.geometry.KDTreeFlann(pcd)

        for i in range(num_points):

            x = pc[i,0]
            y = pc[i,1]

            r = np.linalg.norm([x, y], axis=0)

            sr = alpha * beta * np.pi / 180 * r

            if sr < sr_min:
                sr = sr_min

            [_, _, sqdist] = kd_tree.search_knn_vector_3d(pcd.points[i], k)

            neighbors = -1      # start at -1 since it will always be its own neighbour

            for val in sqdist:
                if np.sqrt(val) < sr:
                    neighbors += 1

            if neighbors >= k_min:
                mask[i] = True  # no snow -> keep

        return mask
def get_dense_calib(sensor: str = 'hdl64'):
        calib_file = DENSE_ROOT / f'calib_{sensor}.txt'
        assert calib_file.exists(), f'{calib_file} not found'
        return calibration_kitti.Calibration(calib_file)

def get_fov_flag(pts_rect, img_shape, calib):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

def crop_pc(pc, calib=None, img_shape=None):
    point_cloud_range = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)
    upper_idx = np.sum((pc[:, 0:3] <= point_cloud_range[3:6]).astype(np.int32), 1) == 3
    lower_idx = np.sum((pc[:, 0:3] >= point_cloud_range[0:3]).astype(np.int32), 1) == 3

    new_pointidx = (upper_idx) & (lower_idx)
    pc = pc[new_pointidx, :]

    # Extract FOV points
    if calib is not None:
        pts_rect = calib.lidar_to_rect(pc[:, 0:3])
        fov_flag = get_fov_flag(pts_rect, img_shape, calib)
        pc = pc[fov_flag]

    return pc

def dror_all():
    samples_txt_path = DENSE_ROOT / 'ImageSets' / 'all.txt'
    save_dir = DENSE_ROOT / 'DROR' / 'alpha_0.45' / 'all' / 'hdl64' / 'strongest' / 'crop_fov'
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_id_list = ['_'.join(x.strip().split(',')) for x in open(samples_txt_path).readlines()]
    calib = get_dense_calib()
    img_shape = np.array([1024, 1920], dtype=np.int32)
    
    for sample_idx in tqdm(sample_id_list):
        print(sample_idx)
        save_path = save_dir / f'{sample_idx}.pkl'

        if save_path.exists():
            continue
        lidar_file = DENSE_ROOT / 'lidar_hdl64_strongest' / f'{sample_idx}.bin'
        pc = np.fromfile(lidar_file, dtype=np.float32).reshape((-1,5))
        pc = crop_pc(pc, calib, img_shape)
        keep_mask = o3d_dynamic_radius_outlier_filter(pc=pc, alpha=0.45)
        snow_indices = (keep_mask == 0).nonzero()[0]#.astype(np.int16)
        with open(save_path, 'wb') as f:
            pickle.dump(snow_indices, f, protocol=pickle.HIGHEST_PROTOCOL)
            

def view_clustered_pc_adverse():
    info_path = DENSE_ROOT / 'dense_infos_train_all_FOV3000_60.pkl'
    lidar_dir = DENSE_ROOT / 'lidar_hdl64_strongest_FOV_clustered'
    dense_infos = []

    with open(info_path, 'rb') as i:
        infos = pickle.load(i)
        dense_infos.extend(infos)
    
    for info in tqdm(dense_infos):
        sample_idx = info['point_cloud']['lidar_idx']
        weather = info['annos']['weather']
        print(sample_idx)

        if weather == 'clear':
            continue
        
        lidar_file = lidar_dir / f'{sample_idx}.bin'
        pc = np.fromfile(lidar_file, dtype=np.float32).reshape((-1,6))
        visualize_pcd_clusters(pc)

def check_in():
    info_path1 = '/home/barza/OpenPCDet/data/dense/FOV3000_Infos/dense_infos_train_all_FOV3000_60.pkl' # ##DENSE_ROOT / 'dense_infos_train_all_FOV3000_60.pkl'
    info_path2 = '/home/barza/OpenPCDet/data/dense/FOV3000_Infos_allobjects/dense_infos_train_all_FOV3000_60.pkl'
    dense_infos1 = []
    dense_infos2 = []
    s1 ={}
    clear = 0
    adverse = 0

    with open(info_path1, 'rb') as i:
        infos = pickle.load(i)
        dense_infos1.extend(infos)
    
    with open(info_path2, 'rb') as i:
        infos = pickle.load(i)
        dense_infos2.extend(infos)
    
    for info in dense_infos1:
        sample_idx = info['point_cloud']['lidar_idx']
        s1[sample_idx] = 1
    
    count = 0
    for info in dense_infos2:
        sample_idx = info['point_cloud']['lidar_idx']
        if sample_idx not in s1:
            print(f'{sample_idx} not in s1')
            count +=1
    print(f'missing: {count}')

def count_infos():
    info_path = '/home/barza/OpenPCDet/data/dense/FOV3000_Infos/dense_infos_train_all_FOV3000_60.pkl' #'/home/barza/OpenPCDet/data/dense/FOV3000_Infos_allobjects/dense_infos_train_all_FOV3000_60.pkl' ##DENSE_ROOT / 'dense_infos_train_all_FOV3000_60.pkl'
    dense_infos = []
    clear = 0
    adverse = 0

    with open(info_path, 'rb') as i:
        infos = pickle.load(i)
        dense_infos.extend(infos)
    
    print(f'Total infos in {info_path}: {len(dense_infos)}')
    for info in tqdm(dense_infos):
        sample_idx = info['point_cloud']['lidar_idx']
        weather = info['annos']['weather']

        if weather == 'clear':
            clear +=1
        else:
            adverse +=1
        
    print(f'clear: {clear}, adverse: {adverse}')

def cluster_all():
    samples_txt_path = DENSE_ROOT / 'ImageSets' / 'all.txt'
    sample_id_list = ['_'.join(x.strip().split(',')) for x in open(samples_txt_path).readlines()]
    calib = get_dense_calib()
    img_shape = np.array([1024, 1920], dtype=np.int32)
    save_dir = DENSE_ROOT / 'lidar_hdl64_strongest_FOV_clustered_train_all_60'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_idx in tqdm(sample_id_list):
        save_path = save_dir / f'{sample_idx}.bin'
        if save_path.exists():
            continue
        
        lidar_file = DENSE_ROOT / 'lidar_hdl64_strongest' / f'{sample_idx}.bin'
        pc = np.fromfile(lidar_file, dtype=np.float32).reshape((-1,5))
        pc = crop_pc(pc, calib, img_shape)

        pc, num_clusters_found = clusterize_pcd(pc, 1000, dist_thresh=0.15, eps=1.0)
        print(f'{sample_idx}: num clusters: {num_clusters_found}')
        #visualize_pcd_clusters(pc)

        if num_clusters_found > 1:
            pc.astype(np.float32).tofile(save_path)
        
        else:
            print(f'{sample_idx} has no clusters!')
            V.draw_scenes(points=pc[:,:4], color_feature=3)
            visualize_pcd_clusters(pc)

def test():
    sample = '/home/barza/DepthContrast/data/semantic_kitti/dataset/sequences/07/velodyne/000695.bin'
    cluster_id_file = '/home/barza/DepthContrast/data/semantic_kitti/dataset_clustered/sequences/07/velodyne/000695.bin'
    pc = np.fromfile(str(sample), dtype=np.float32).reshape(-1, 4)

    pc, num_clusters_found = clusterize_pcd(pc, 100, dist_thresh=0.25, eps=0.25)
    #visualize_pcd_clusters(pc)
    cluster_id = pc[:,-1]
    cluster_id = cluster_id.astype(np.int16)
    cluster_id = cluster_id.astype(np.float32)
    cluster_ids = np.fromfile(str(cluster_id_file), dtype=np.int16).reshape(-1, 1)
    cluster_ids = cluster_ids.astype(np.float32)
    b=1

def cluster_semantic_kitti():
    train_seq = [ '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'] 

    for seq in train_seq:
        print(f'Processing Seq: {seq}')
        point_seq_path = os.path.join(SEMANTIC_KITTI_ROOT, 'dataset', 'sequences', seq, 'velodyne')
        point_seq_bin = os.listdir(point_seq_path)
        point_seq_bin.sort()
        save_dir = SEMANTIC_KITTI_ROOT / 'dataset_clustered' / 'sequences' / seq / 'velodyne'
        save_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in tqdm(point_seq_bin):
            points_path = f'{point_seq_path}/{sample_idx}'
            save_path = save_dir / sample_idx

            if save_path.exists():
                continue

            pc = np.fromfile(str(points_path), dtype=np.float32).reshape(-1, 4)

            pc, num_clusters_found = clusterize_pcd(pc, 100, dist_thresh=0.25, eps=0.25)
            #visualize_pcd_clusters(pc)

            if num_clusters_found > 1:
                cluster_id = pc[:,-1]
                assert cluster_id.max() < np.iinfo(np.int16).max
                cluster_id.astype(np.int16).tofile(save_path)
            else:
                print(f'{points_path} has no clusters!')
                V.draw_scenes(points=pc[:,:4], color_feature=3)
                visualize_pcd_clusters(pc)

def cluster_kitti():
    info_path =  KITTI_ROOT / 'kitti_infos_all.pkl'
    train_save_dir = KITTI_ROOT / 'training_clustered' / 'velodyne'
    train_save_dir.mkdir(parents=True, exist_ok=True)

    test_save_dir = KITTI_ROOT / 'testing_clustered' / 'velodyne'
    test_save_dir.mkdir(parents=True, exist_ok=True)

    kitti_infos = []

    with open(info_path, 'rb') as i:
        infos = pickle.load(i)
        kitti_infos.extend(infos)
    
    for info in tqdm(kitti_infos):
        sample_idx = info['point_cloud']['lidar_idx']
        if info['velodyne_parent_dir'] == 'testing':
            save_path = test_save_dir / f'{sample_idx}.bin'
        elif info['velodyne_parent_dir'] == 'training':
            save_path = train_save_dir / f'{sample_idx}.bin'
        else:
            raise ValueError('Only training and testing dirs allowed')

        if save_path.exists():
            continue

        lidar_file = KITTI_ROOT / info['velodyne_parent_dir'] / 'velodyne' / ('%s.bin' % sample_idx)
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

        pc, num_clusters_found = clusterize_pcd(pc, 100, dist_thresh=0.25, eps=0.5)
        #print(f'{sample_idx}: num clusters: {num_clusters_found}')
        #visualize_pcd_clusters(pc)

        if num_clusters_found > 1:
            cluster_id = pc[:,-1]
            assert cluster_id.max() < np.iinfo(np.int16).max
            cluster_id.astype(np.int16).tofile(save_path)
        else:
            print(f'{sample_idx} from has no clusters!')
            V.draw_scenes(points=pc[:,:4], color_feature=3)
            visualize_pcd_clusters(pc)

def cluster_adverse():
    info_path =  DENSE_ROOT / 'dense_infos_train_all_FOV3000_60.pkl'
    save_dir = DENSE_ROOT / 'lidar_hdl64_strongest_FOV_clustered_train_all_60'
    save_dir.mkdir(parents=True, exist_ok=True)
    calib = get_dense_calib()
    dense_infos = []

    with open(info_path, 'rb') as i:
        infos = pickle.load(i)
        dense_infos.extend(infos)
    
    for info in tqdm(dense_infos):
        sample_idx = info['point_cloud']['lidar_idx']
        save_path = save_dir / f'{sample_idx}.bin'

        if save_path.exists():
            continue

        weather = info['annos']['weather']
        print(sample_idx)

        if weather == 'clear':
            continue
        
        img_shape = info['image']['image_shape']

        lidar_file = DENSE_ROOT / 'lidar_hdl64_strongest' / f'{sample_idx}.bin'
        pc = np.fromfile(lidar_file, dtype=np.float32).reshape((-1,5))
        pc = crop_pc(pc, calib, img_shape)

        pc, num_clusters_found = clusterize_pcd(pc, 1000, dist_thresh=0.15, eps=1.0)
        print(f'{sample_idx}: num clusters: {num_clusters_found}')
        #visualize_pcd_clusters(pc)

        if num_clusters_found > 1:
            pc.astype(np.float32).tofile(save_path)
        
        else:
            print(f'{sample_idx} from {weather} has no clusters!')
            V.draw_scenes(points=pc[:,:4], color_feature=3)
            visualize_pcd_clusters(pc)


        # keep_mask = o3d_dynamic_radius_outlier_filter(pc=pc, alpha=0.45)
        # snow_indices = (keep_mask == 0).nonzero()[0]#.astype(np.int16)
        # range_snow_indices = np.linalg.norm(pc[snow_indices][:,:3], axis=1)
        # snow_indices = snow_indices[range_snow_indices < 30]
        # keep_indices = np.ones(len(pc), dtype=bool)
        # keep_indices[snow_indices] = False
        # pc = pc[keep_indices]
        # pc = clusterize_pcd(pc[:,:5], 1000, dist_thresh=0.25, eps=0.25)
        # visualize_pcd_clusters(pc)

if __name__ == '__main__':
    cluster_kitti()
   