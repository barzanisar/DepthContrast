import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datasets.features import global_descriptors
from torch import nn


quantile_thresh = np.array([0.01, 0.03, 0.05, 0.1, 1, 5])/100 #0.2, 0.3, 0.4, 0.5, 1,2,3,4,5
shape_dim_dict = {'esf':640, 'vfh': 308, 'gasd': 512}
desc_types = ['esf']
shape_dist_methods = ['cosine']
num_samples = 15000
min_num_pts = [5, 20, 50, 100]


split_txt_file = '/home/barza/DepthContrast/data/waymo/ImageSets/train_ten.txt'
lidar_data_path = Path('/home/barza/DepthContrast/data/waymo/waymo_processed_data_v_1_2_0')
cluster_root_path = Path('/home/barza/DepthContrast/data/waymo/waymo_processed_data_v_1_2_0_clustered')
seglabels_root_path = Path('/home/barza/DepthContrast/data/waymo/waymo_processed_data_v_1_2_0_labels')

WAYMO_LABELS = ['UNDEFINED', 'CAR', 'TRUCK', 'BUS', 'OTHER_VEHICLE', 'MOTORCYCLIST', 'BICYCLIST', 'PEDESTRIAN', 'SIGN',
                  'TRAFFIC_LIGHT', 'POLE', 'CONSTRUCTION_CONE', 'BICYCLE', 'MOTORCYCLE', 'BUILDING', 'VEGETATION',
                  'TREE_TRUNK', 'CURB', 'ROAD', 'LANE_MARKER', 'OTHER_GROUND', 'WALKABLE', 'SIDEWALK']

def get_lidar(sequence_name, sample_idx):
    lidar_file = lidar_data_path / sequence_name / ('%04d.npy' % sample_idx)
    point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

    points_all = point_features[:, 0:4] #points_all: x,y,z,i, skip elongation
    points_all[:, 3] = np.tanh(points_all[:, 3]) * 255.0  #TODO:
    return points_all #only get xyzi

def get_cluster_labels(sequence_name, sample_idx):
    label_file = cluster_root_path / sequence_name / ('%04d.npy' % sample_idx)
    labels = np.fromfile(label_file, dtype=np.float16)
    return labels

def get_seglabels(sequence_name, sample_idx, num_points):
    label_file = seglabels_root_path / sequence_name / ('%04d.npy' % sample_idx)
    labels = np.load(label_file)
    assert labels.shape[0] == num_points, label_file
    assert labels.max() < 23, label_file
    
    return labels


seq_list = [x.strip().split('.')[0] for x in open(split_txt_file).readlines()]

# Get all infos that have seg labels
waymo_infos_cluster=[]
waymo_infos_seg_labels=[]

num_skipped_infos=0
for seq_name in seq_list:
    seq_info_path = seglabels_root_path / seq_name /  ('%s.pkl' % seq_name)
    # if not seq_info_path.exists():
    #     num_skipped_infos += 1
    #     continue
    with open(seq_info_path, 'rb') as f:
        seq_infos_seg = pickle.load(f) # loads 20 infos for one seq pkl i.e. 20 frames if seq pkl was formed by sampling every 10th frame
        waymo_infos_seg_labels.extend(seq_infos_seg)
    
    sample_idx_with_seg_labels = [info['point_cloud']['sample_idx'] for info in seq_infos_seg]

    seq_info_path_cluster = cluster_root_path / seq_name /  'approx_boxes.pkl'
    with open(seq_info_path_cluster, 'rb') as f:
        seq_infos_cluster = pickle.load(f) # loads 20 infos for one seq pkl i.e. 20 frames if seq pkl was formed by sampling every 10th frame

    for info in seq_infos_cluster:
        if info['point_cloud']['sample_idx'] in sample_idx_with_seg_labels:
            waymo_infos_cluster.append(info) # each info is one frame

print('Total frames: ',len(waymo_infos_cluster))
assert len(waymo_infos_cluster) == len(waymo_infos_seg_labels)

def get_lwhz(info_cluster):
    z = info_cluster['approx_boxes_closeness_to_edge'][:,2]
    l = info_cluster['approx_boxes_closeness_to_edge'][:,3:5].max(axis=1)
    w = info_cluster['approx_boxes_closeness_to_edge'][:,3:5].min(axis=1)
    h = info_cluster['approx_boxes_closeness_to_edge'][:,5]
    lwhz_mat = np.zeros((l.shape[0], 4))
    lwhz_mat[:, 0] = l
    lwhz_mat[:, 1] = w
    lwhz_mat[:, 2] = h
    lwhz_mat[:, 3] = z
    return lwhz_mat

def get_class_ids(cluster_labels, seg_labels, info):
    class_ids=np.zeros(len(info['cluster_labels_boxes']))
    for i, lbl in enumerate(info['cluster_labels_boxes']):
        gt_pt_labels = seg_labels[cluster_labels==lbl]
        gt_labels, cnts = np.unique(gt_pt_labels, return_counts=True)
        majority_label = gt_labels[np.argmax(cnts)]
        class_ids[i] = majority_label

    
    return class_ids

def compute_iou_mat(lwhz_mat, iou_z=False):
    l_a= lwhz_mat[:,0].reshape(-1, 1)
    w_a= lwhz_mat[:,1].reshape(-1, 1)
    h_a= lwhz_mat[:,2].reshape(-1, 1)
    z_a= lwhz_mat[:,3].reshape(-1, 1)

    l_b = l_a.reshape(1, -1)
    w_b = w_a.reshape(1, -1)
    h_b = h_a.reshape(1, -1)
    z_b = z_a.reshape(1, -1)

    vol_a = (l_a*w_a*h_a).view(-1, 1) # col: Nposx1
    vol_b = (l_b * w_b * h_b).view(1, -1) # row: 1xK

    if iou_z:
        boxes_a_height_max = (z_a + h_a / 2).reshape(-1, 1) #col
        boxes_a_height_min = (z_a - h_a / 2).reshape(-1, 1) #col
        boxes_b_height_max = boxes_a_height_max.reshape(1, -1) #row
        boxes_b_height_min = boxes_a_height_min.reshape(1, -1) #row

        max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min) #(Npos, Nneg)
        min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max) #(Npos, Nneg)
        overlaps_h = torch.clamp(min_of_max - max_of_min, min=0) # torch.min(h, h_neg)  #(Npos, Nneg) height overlaps between each pos and neg sample

        overlap_vol = torch.min(l_a, l_b) *  torch.min(w_a, w_b) * overlaps_h # NxK
    else:
        overlap_vol = torch.min(l_a, l_b) *  torch.min(w_a, w_b) * torch.min(h_a, h_b) # NxK

    iou3d = overlap_vol / torch.clamp(vol_a + vol_b - overlap_vol, min=1e-6) # NxK

    return iou3d

def compute_shape_desc_dist_mat(desc_mat, method='cosine'):
    if method == 'cosine':
        desc_mat =  nn.functional.normalize(desc_mat, dim=1, p=2) #(Npos, 604)
        shape_desc_cosine_similarity = torch.einsum('nc,ck->nk', [desc_mat, desc_mat.T])
        shape_dist_mat = 1.0 - shape_desc_cosine_similarity
        shape_dist_mat = torch.clamp(shape_dist_mat, min=0)
    else:
        N_clusters= desc_mat.shape[0]
        shape_dist_mat = torch.zeros((N_clusters, N_clusters)) #(Npos, Nneg)
                
        #calculate euclidean distance
        for i in range(N_clusters):
            shape_dist_mat[i] = torch.norm(desc_mat[i] - desc_mat, dim=1) #norm((Nneg, 604)=(1, 604) - (Nneg, 604)) -> Nneg
        # map shape_dist_mat between 0 to 1
        shape_dist_mat = shape_dist_mat / torch.max(shape_dist_mat, dim = 1, keepdim=True)[0] #(Npos, Nneg) / (Npos, 1) = (Npos, Nneg)

    return shape_dist_mat

def get_shape_desc(points, cluster_labels, info,  method):
    shape_descs = [] #nclusters, feat dim
    num_obj_pts = []
    for i, lbl in enumerate(info['cluster_labels_boxes']):
        obj_points = points[cluster_labels == lbl][:,:3] - info['approx_boxes_closeness_to_edge'][i, :3]
        num_obj_pts.append(len(obj_points))
        if len(obj_points) > 5:
            shape_desc = global_descriptors.extract_feats(obj_points, method=method)
            if shape_desc is not None:
                shape_descs.append(shape_desc)
            else:
                shape_descs.append(np.array([np.nan]*shape_dim_dict[method]))
        else:
            shape_descs.append(np.array([np.nan]*shape_dim_dict[method]))

    shape_descs = np.asarray(shape_descs)
    num_obj_pts = np.asarray(num_obj_pts)

    return shape_descs, num_obj_pts

def extract_descs():
    for desc_type in desc_types:
        if desc_type == 'iou_z':
            continue
        sign = desc_type
        
        print(f'Processing {sign}')
        desc_mat = [] #num frames elems -> elem = (n_samples, shape feat dim) or (n_samples, 4)
        class_ids = [] #num frames elems -> elem = (n_samples,)
        num_obj_pts_all = []
        k = 0
        for info_cluster in waymo_infos_cluster:
            # k +=1
            # if k % 2 == 0:
            #     continue
            pc_info = info_cluster['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            frame_id = info['frame_id']

            points = get_lidar(sequence_name, sample_idx)
            pt_cluster_labels = get_cluster_labels(sequence_name, sample_idx)
            pt_seg_labels = get_seglabels(sequence_name, sample_idx, num_points=pt_cluster_labels.shape[0])
            assert pt_cluster_labels.shape[0] == pt_seg_labels.shape[0]

            if desc_type in ['iou', 'iou_z']:
                lwhz_frame = get_lwhz(info_cluster) #n clusters, 4
                desc_mat.append(lwhz_frame)
            elif desc_type in ['vfh', 'esf', 'gasd']:
                shape_desc_frame, num_obj_pts_frame = get_shape_desc(points, pt_cluster_labels, info_cluster, method=desc_type) # n cluster in this frame, shape feat dim
                desc_mat.append(shape_desc_frame)
                num_obj_pts_all.extend(num_obj_pts_frame)

            class_ids_frame = get_class_ids(pt_cluster_labels, pt_seg_labels, info_cluster)
            class_ids.append(class_ids_frame)

        desc_mat = np.concatenate(desc_mat)
        class_ids = np.concatenate(class_ids)
        if desc_type in ['vfh', 'esf', 'gasd']:
            num_obj_pts_all = np.array(num_obj_pts_all)
            pickle.dump(num_obj_pts_all, open(f'num_obj_pts.pkl', 'wb'))

        pickle.dump(desc_mat, open(f'desc_mat_{sign}.pkl', 'wb'))
        pickle.dump(class_ids, open(f'class_ids_{sign}.pkl', 'wb'))

        if desc_type == 'iou':
            pickle.dump(desc_mat, open(f'desc_mat_{sign}_z.pkl', 'wb'))
            pickle.dump(class_ids, open(f'class_ids_{sign}_z.pkl', 'wb'))

        print(f'Computed descs {sign} for {desc_mat.shape[0]} clusters')

def analyse_knn(dist, class_ids, sign):
    n_samples = dist.shape[0]
    acc_samplewise_threshwise = np.zeros((n_samples, len(quantile_thresh))) # rows are samples, cols are thresholds 
    mean_samplewise_acc_threshwise = np.zeros(len(quantile_thresh))
    mean_classwise_acc_threshwise = np.zeros((len(WAYMO_LABELS), len(quantile_thresh))) #includes undefined class 

    for i, thresh in enumerate(quantile_thresh):
        # avg_acc = []
        print(f'Quantile thresh {thresh}')
        row_wise_quantiles = torch.quantile(dist, thresh, dim=1, keepdim=True)
        dist = dist.fill_diagonal_(float('inf'))
        mask_selected = dist < row_wise_quantiles.repeat(1, n_samples)
        class_ids_of_nn = class_ids.view(1, -1).repeat(n_samples, 1) #[mask_selected]
        mask_gt = class_ids_of_nn ==  class_ids.view(-1, 1).repeat(1, n_samples)
        mask_selected_hit = mask_gt & mask_selected
        acc_sample_wise = mask_selected_hit.sum(axis=1)/torch.clamp(mask_selected.sum(axis=1), min=1)

        mean_samplewise_acc_threshwise[i] = acc_sample_wise.mean()
        acc_samplewise_threshwise[:, i] = acc_sample_wise

        for cid in np.unique(class_ids):
            class_mean_acc = acc_sample_wise[class_ids == cid].mean()
            mean_classwise_acc_threshwise[int(cid), i] = class_mean_acc

    print(f'Done Processing {sign}')
    pickle.dump(acc_samplewise_threshwise, open(f'acc_samplewise_threshwise_{sign}.pkl', 'wb'))
    pickle.dump(mean_samplewise_acc_threshwise, open(f'mean_samplewise_acc_threshwise_{sign}.pkl', 'wb'))
    pickle.dump(mean_classwise_acc_threshwise, open(f'mean_classwise_acc_threshwise_{sign}.pkl', 'wb'))
    print(f'Done dumping {sign}')

def load_desc_class_ids(sign, mask=None):
    desc_mat = pickle.load(open(f'desc_mat_{sign}.pkl', 'rb'))
    class_ids = pickle.load(open(f'class_ids_{sign}.pkl', 'rb'))
    # #remove class ids 0
    if mask is None:
        mask = (class_ids > 0) & (~np.isnan(desc_mat.sum(axis=1)))
        desc_mat = desc_mat[mask]
        class_ids = class_ids[mask]
    else:
        desc_mat = desc_mat[mask]
        class_ids = class_ids[mask]

    desc_mat = torch.from_numpy(desc_mat)[:num_samples]
    class_ids = torch.from_numpy(class_ids)[:num_samples]
    return desc_mat, class_ids, mask

################## 1. Extract Descs ###############  

# extract_descs()

################## 2. Analyse KNN ###############
 
for desc_type in desc_types:
    sign = desc_type
    desc_mat, class_ids, mask = load_desc_class_ids(sign)
    num_obj_pts = pickle.load(open(f'num_obj_pts.pkl', 'rb'))
    num_obj_pts = num_obj_pts[mask][:num_samples]
    
    for num_pts in min_num_pts:
        mask = num_obj_pts>num_pts
        desc_mat_min_pts = desc_mat[mask]
        class_ids_min_pts = class_ids[mask]

        n_samples = desc_mat_min_pts.shape[0]
        print(f'{sign}-min pts nsamples {n_samples}')
        for dist_type in shape_dist_methods:
            sign = f'{desc_type}-{num_pts}-{dist_type}'
            print(f'Computing dist {sign}')
            dist = compute_shape_desc_dist_mat(desc_mat_min_pts, method=dist_type)
            print(f'Analysing knn {sign}')
            analyse_knn(dist, class_ids_min_pts, sign)


################## 3. Plot ############### 
signs=[]
for desc_type in desc_types:
    for num_pts in min_num_pts:
        for dist_type in shape_dist_methods:
            sign = f'{desc_type}-{num_pts}-{dist_type}'
            signs += [sign]

    
plt.figure()
for sign in signs:
    mean_samplewise_acc_threshwise = pickle.load(open(f'mean_samplewise_acc_threshwise_{sign}.pkl', 'rb'))
    #plt.plot(quantile_thresh*100, mean_samplewise_acc_threshwise, label=f'{sign}')
    plt.plot(np.round(quantile_thresh*num_samples), mean_samplewise_acc_threshwise, label=f'{sign}')
    print(f'{sign}: {mean_samplewise_acc_threshwise}')

plt.ylabel('Average Precision for K Nearest Neighbors')
# plt.xlabel('Quantiles %')
plt.xlabel('K')
plt.title(f'Average Precision KNN')

plt.grid()
plt.legend()
plt.savefig(f'avg_precknn_shape_minpts.png')
plt.show()
b=1
