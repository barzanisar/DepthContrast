import numpy as np
import open3d as o3d
#import scipy.stats as stats
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V


from lib.LiDAR_fog_sim.fog_simulation import *
from lib.LiDAR_snow_sim.tools.wet_ground.augmentation import ground_water_augmentation

def foggify(cfg, points, alpha, augmentation_method, last_col_cluster_id):

    if augmentation_method == 'CVL' and alpha != '0.000':

        p = ParameterSet(alpha=float(alpha), gamma=0.000001)

        gain = cfg.get('FOG_GAIN', False)
        fog_noise_variant = cfg.get('FOG_NOISE_VARIANT', 'v1')
        soft = cfg.get('FOG_SOFT', True)
        hard = cfg.get('FOG_HARD', True)

        points, _, _ = simulate_fog(p, pc=points, noise=10, gain=gain, noise_variant=fog_noise_variant,
                                    soft=soft, hard=hard, last_col_cluster_id=last_col_cluster_id)

    return points

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

def dror(cfg, points_moco, sample_idx, logger, root_dense_path, sensor_type='hdl64', signal_type= 'strongest'):
    dror_applied=False
    try:
        alpha = cfg['DROR']

        dror_path = root_dense_path / 'DROR' / f'alpha_{alpha}' / \
                    'all' / sensor_type / signal_type / 'crop_fov' / f'{sample_idx}.pkl'
        
        if dror_path.exists():
            with open(str(dror_path), 'rb') as f:
                snow_indices = pickle.load(f) #gives snow indices on full pc not fov pc
        else:
            logger.add_line(f'DROR path does not exist!, {sample_idx}')
            keep_mask = o3d_dynamic_radius_outlier_filter(points_moco, alpha=alpha)
            snow_indices = (keep_mask == 0).nonzero()[0]#.astype(np.int16)

        range_snow_indices = np.linalg.norm(points_moco[snow_indices][:,:3], axis=1)
        snow_indices = snow_indices[range_snow_indices < 30]
        keep_indices = np.ones(len(points_moco), dtype=bool)
        keep_indices[snow_indices] = False

        points_moco = points_moco[keep_indices]
        #V.draw_scenes(points=points_moco, color_feature='intensity')
        # points_moco = clusterize_pcd(points_moco, 5, dist_thresh=0.15, eps=0.1)
        # visualize_pcd_clusters(points_moco)
        dror_applied = True
    except:
        logger.add_line(f'ERROR: Could not apply DROR!, {sample_idx}')
        pass

    return points_moco, dror_applied

def fog_sim(cfg, points_moco, last_col_cluster_id):
    fog_applied = False
    augmentation_method = cfg['FOG_AUGMENTATION'].split('_')[0]
    chance = cfg['FOG_AUGMENTATION'].split('_')[-1]
    choices = [0]

    if chance == '8in9':
        choices = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    elif chance == '4in5':
        choices = [1, 1, 1, 1, 0]
    elif chance == '1in2':
        choices = [1, 0]
    elif chance == '1in1':
        choices = [1]
    elif chance == '1in4':
        choices = [1, 0, 0, 0]
    elif chance == '1in10': #recommended by lidar snow sim paper
        choices = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if np.random.choice(choices):
        alphas = ['0.005', '0.010', '0.020', '0.030', '0.060']
        curriculum_stage = int(np.random.randint(low=0, high=len(alphas)))
        alpha = alphas[curriculum_stage]

        # if alpha != '0.000': 
        #     mor = np.log(20) / float(alpha)

        points_moco = foggify(cfg, points_moco, alpha, augmentation_method, last_col_cluster_id)
        fog_applied=True
    
    return points_moco, fog_applied

def wet_surface_sim(cfg, snowfall_augmentation_applied, points_moco, logger):
    wet_surface_applied = False
    method = cfg['WET_SURFACE']

    choices = [0]

    if '1in2' in method:
        choices = [0, 1]                            # pointcloud gets augmented with 50% chance
    
    elif '1in1' in method:
        choices = [1]

    elif '1in4' in method:
        choices = [0, 0, 0, 1]                      # pointcloud gets augmented with 25% chance

    elif '1in10' in method:
        choices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]    # pointcloud gets augmented with 10% chance

    apply_coupled = cfg.get('COUPLED', False) and snowfall_augmentation_applied

    if np.random.choice(choices) or apply_coupled: 

        elements = np.linspace(0.001, 0.012, 12) #np.linspace(0.1, 1.2, 12)
        probabilities = 5 * np.ones_like(elements)  # each element initially 5% chance

        probabilities[0] = 15   # 0.1
        probabilities[1] = 25   # 0.2
        probabilities[2] = 15   # 0.3

        probabilities = probabilities / 100

        water_height = np.random.choice(elements, 1, p=probabilities)

        try:
            points_moco = ground_water_augmentation(points_moco, water_height=water_height, debug=False)
            wet_surface_applied = True
        except (TypeError, ValueError):
            logger.add_line(f'ERROR: Could not apply wet surface!')
            pass
    
    return points_moco, wet_surface_applied

def snow_sim(cfg, logger, rainfall_rates, sample_idx, root_dense_path, points_moco, num_feat, last_col_cluster_id):
    snowfall_augmentation_applied=False
    parameters = cfg['SNOW'].split('_')

    sampling = parameters[0]        # e.g. uniform
    mode = parameters[1]            # gunn or sekhon
    chance = parameters[2]          # e.g. 8in9

    choices = [0]

    if chance == '8in9':
        choices = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    elif chance == '4in5':
        choices = [1, 1, 1, 1, 0]
    elif chance == '1in2':
        choices = [1, 0]
    elif chance == '1in1':
        choices = [1]
    elif chance == '1in4':
        choices = [1, 0, 0, 0]
    elif chance == '1in10': #recommended by lidar snow sim paper
        choices = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if np.random.choice(choices): 

        rainfall_rate = 0

        if sampling == 'uniform':
            rainfall_rate = int(np.random.choice(rainfall_rates))

        if cfg['FOV_POINTS_ONLY']:
            snow_sim_dir = 'snowfall_simulation_FOV_clustered' if last_col_cluster_id else 'snowfall_simulation_FOV'
        else:
            snow_sim_dir = 'snowfall_simulation'
        lidar_file = root_dense_path / snow_sim_dir / mode / \
                    f'lidar_hdl64_strongest_rainrate_{rainfall_rate}' / f'{sample_idx}.bin'

        try:
            points_moco = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, num_feat)
            snowfall_augmentation_applied = True
        except FileNotFoundError:
            logger.add_line(f'\n{lidar_file} not found')
            pass
    
    return points_moco, snowfall_augmentation_applied


def nn_upsample(cfg, points_moco, last_col_cluster_id):
    upsampling_applied=False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_moco[:, :3])

    method = cfg['UPSAMPLE']

    choices = [0]

    if '1in2' in method:
        choices = [0, 1]                            # pointcloud gets augmented with 50% chance
    
    elif '1in1' in method:
        choices = [1]

    elif '1in4' in method:
        choices = [0, 0, 0, 1]                      # pointcloud gets augmented with 25% chance

    elif '1in10' in method:
        choices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]    # pointcloud gets augmented with 10% chance
    
    if np.random.choice(choices): 
        upsampling_applied=True
        iters = 1
        k_neighbours = 10

        for iter in range(iters):
            #start_time = time.time()
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            num_pts = len(pcd.points)
            new_pc = np.zeros(points_moco.shape)
            for i in range(num_pts):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k_neighbours)
                new_pc[i, :4] = np.mean(points_moco[idx][:,:4], axis=0)
                if last_col_cluster_id:
                    neighbour_cluster_ids, counts = np.unique(points_moco[idx][:,-1], return_counts=True)
                    new_pc[i, -1] = neighbour_cluster_ids[counts.argmax()]
            #print(f'Iter: {iter}, Time: {time.time()-start_time} ')

            points_moco = np.concatenate((points_moco, new_pc))
    
    return points_moco, upsampling_applied

