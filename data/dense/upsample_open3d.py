import pcl
import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
import time

from sklearn import neighbors

ROOT_PATH = (Path(__file__) / '../../..').resolve() #DepthContrast
DENSE_ROOT = ROOT_PATH /'data' / 'dense'
SPLIT_PATH = DENSE_ROOT / 'ImageSets' / 'train_clear.txt'

def get_colors(pc, color_feature=None):
    # create colormap
    if color_feature == 'x':
        feature = pc[:, 0]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 'y':

        feature = pc[:, 1]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 'z':
        feature = pc[:, 2]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 'intensity':
        feature = pc[:, 3]
        min_value = np.min(feature)
        max_value = 255

    elif color_feature == 'label' or color_feature == 'channel':
        feature = pc[:, 4]
        min_value = np.min(feature)
        max_value = np.max(feature)

    else:
        #color_feature == 'range'
        feature = np.linalg.norm(pc[:, 0:3], axis=1)
        min_value = np.min(feature)
        max_value = np.max(feature)


    norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)


    cmap = cm.jet  # sequential

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = m.to_rgba(feature)
    colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
    colors[:, 3] = 0.5

    return colors[:, :3]


def draw_pts(pcd1, pcd2=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if True:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    
    vis.add_geometry(pcd1)
    if pcd2 is not None:
        vis.add_geometry(pcd2)
    vis.run()
    vis.destroy_window()

def load_dense_pointcloud(file: str, sensor: str ='hdl64', signal: str = 'strongest') -> np.ndarray:

    filename = DENSE_ROOT / f'lidar_{sensor}_{signal}' / f'{file}.bin'

    pc = np.fromfile(filename, dtype=np.float32)
    pc = pc.reshape((-1, 5))

    return pc

def crop_pc(pc):
    point_cloud_range = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)
    upper_idx = np.sum((pc[:, 0:3] <= point_cloud_range[3:6]).astype(np.int32), 1) == 3
    lower_idx = np.sum((pc[:, 0:3] >= point_cloud_range[0:3]).astype(np.int32), 1) == 3

    new_pointidx = (upper_idx) & (lower_idx)
    pc = pc[new_pointidx, :]
    return pc

sample_id_list = sorted(['_'.join(x.strip().split(',')) for x in open(SPLIT_PATH).readlines()])

for sample_idx in sample_id_list:
    pc = load_dense_pointcloud(sample_idx)
    pc = crop_pc(pc)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normal
    pcd.colors = o3d.utility.Vector3dVector(get_colors(pc, 'intensity'))
    # pcd.estimate_normals()
    # draw_pts(pcd)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # pcd.orient_normals_consistent_tangent_plane(10)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # radii = [0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.3]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    # pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([pcd, mesh])


    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=20)
    # o3d.visualization.draw_geometries([mesh],
    #                               zoom=0.664,
    #                               front=[-0.4761, -0.4698, -0.7434],
    #                               lookat=[1.8900, 3.2596, 0.9284],
    #                               up=[0.2304, -0.8825, 0.4101])

    # pcd = mesh.sample_points_uniformly(number_of_points=2*pc.shape[0])
    # o3d.visualization.draw_geometries([pcd])
    # pcd = mesh.sample_points_poisson_disk(number_of_points=2*pc.shape[0], init_factor=5)
    # draw_pts(pcd)
    # o3d.visualization.draw_geometries([pcd])
    b=1
    iters = 1
    neighbours = 10

    for iter in range(iters):
        start_time = time.time()
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        num_pts = len(pcd.points)
        new_pc = np.zeros(pc.shape)
        for i in range(num_pts):
            [k, idx, sqdist] = pcd_tree.search_knn_vector_3d(pcd.points[i], neighbours)
            new_pc[i, :] = np.mean(pc[idx], axis=0)
            # print("Find its neighbors with distance less than 0.2, and paint them green.")
            # [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
        print(f'Iter: {iter}, Time: {time.time()-start_time} ')
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_pc[:, :3])
        #new_pcd.colors = o3d.utility.Vector3dVector(get_colors(new_pc, 'intensity'))
        # #Visualize separately
        # draw_pts(pcd)
        # draw_pts(new_pcd)

        #Visualize jointly
        pcd.colors = o3d.utility.Vector3dVector( [0,0,1] * np.ones((pc.shape[0], 3))) #blue
        new_pcd.colors = o3d.utility.Vector3dVector( [1,0,0] * np.ones((pc.shape[0], 3))) #red
        draw_pts(pcd, new_pcd)

        # #Visualize augmented
        # pc = np.concatenate((pc, new_pc))
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(get_colors(pc, 'intensity'))
        # draw_pts(pcd)
    

