import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datasets.features import global_descriptors
from torch import nn


quantile_thresh = np.array([0.01, 0.03, 0.05, 0.1])/100 #0.2, 0.3, 0.4, 0.5, 1,2,3,4,5
shape_dim_dict = {'esf':640, 'vfh': 308, 'gasd': 512}
desc_types = ['iou_z', 'esf']
dist_type = 'cosine'
num_samples = 15000
shape_desc_min_pts = 20


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
    # labels = np.fromfile(label_file, dtype=np.float16)
    # if labels.shape[0] == num_points and labels.max() < 23:
    #     return labels
    # else:
    labels = np.load(label_file)
    # if labels.shape[0] != num_points:
    #     print(label_file, f'lbls: {labels.shape[0]}, numpts: {num_points}')
    # if labels.max() >= 23:
    #     print(label_file)
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

def get_shape_desc(points, cluster_labels, info,  method, min_num_pts):
    shape_descs = [] #nclusters, feat dim
    for i, lbl in enumerate(info['cluster_labels_boxes']):
        obj_points = points[cluster_labels == lbl][:,:3] - info['approx_boxes_closeness_to_edge'][i, :3]
        if len(obj_points) > min_num_pts: #TODO make this 20
            shape_desc = global_descriptors.extract_feats(obj_points, method=method)
                    # visualize_selected_labels(obj_points, data_dict["points"][obj_points_mask, -1], [i])

            if shape_desc is not None:
                shape_descs.append(shape_desc)
            else:
                shape_descs.append(np.array([np.nan]*shape_dim_dict[method]))
        else:
            shape_descs.append(np.array([np.nan]*shape_dim_dict[method]))

    shape_descs = np.asarray(shape_descs)

    return shape_descs


def analyse_knn(dist, class_ids, sign):
    n_samples = dist.shape[0]
    acc_samplewise_threshwise = np.zeros((n_samples, len(quantile_thresh))) # rows are samples, cols are thresholds 
    mean_samplewise_acc_threshwise = np.zeros(len(quantile_thresh))
    mean_classwise_acc_threshwise = np.zeros((len(WAYMO_LABELS), len(quantile_thresh))) #includes undefined class 

    for i, thresh in enumerate(quantile_thresh):
        # avg_acc = []
        print(f'Quantile thresh {thresh}')
        row_wise_quantiles = torch.quantile(dist, thresh, dim=1, keepdim=True)
        mask_selected = dist < row_wise_quantiles.repeat(1, n_samples)
        # class_ids_of_nn = class_ids.view(1, -1).repeat(n_samples, 1) #[mask_selected]
        # mask_gt = class_ids_of_nn ==  class_ids.view(-1, 1).repeat(1, n_samples)
        mask_gt = class_ids.view(-1, 1) == class_ids.view(1, -1)
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
        mask = np.logical_or(class_ids > 0, ~np.isnan(desc_mat[:,0])) 
        desc_mat = desc_mat[mask]
        class_ids = class_ids[mask]
    else:
        desc_mat = desc_mat[mask]
        class_ids = class_ids[mask]

    desc_mat = torch.from_numpy(desc_mat)[:num_samples]
    class_ids = torch.from_numpy(class_ids)[:num_samples]
    return desc_mat, class_ids, mask

################## 1. Extract Descs ###############  

desc_esf, class_ids_esf, mask = load_desc_class_ids('esf-20')
desc_iou, class_ids_iou, _ = load_desc_class_ids('iou_z', mask)
sign='esfANDiou-cosine'
assert torch.equal(class_ids_esf, class_ids_iou)
assert desc_esf.shape[0] == desc_iou.shape[0]
class_ids = class_ids_esf

iou3d = compute_iou_mat(desc_iou, iou_z=True)
iou_dist = 1-iou3d
esf_dist = compute_shape_desc_dist_mat(desc_esf, method='cosine')

n_samples = esf_dist.shape[0]
acc_samplewise_threshwise = np.zeros((n_samples, len(quantile_thresh))) # rows are samples, cols are thresholds 
mean_samplewise_acc_threshwise = np.zeros(len(quantile_thresh))
mean_classwise_acc_threshwise = np.zeros((len(WAYMO_LABELS), len(quantile_thresh))) #includes undefined class 

for i, thresh in enumerate(quantile_thresh):
    # avg_acc = []
    print(f'Quantile thresh {thresh}')
    row_wise_quantiles = torch.quantile(iou_dist, thresh, dim=1, keepdim=True)
    mask_selected_iou = iou_dist < row_wise_quantiles.repeat(1, n_samples)
    row_wise_quantiles = torch.quantile(esf_dist, thresh, dim=1, keepdim=True)
    mask_selected_esf = esf_dist < row_wise_quantiles.repeat(1, n_samples)
    
    mask_selected = mask_selected_iou & mask_selected_esf
    mask_selected_sum = mask_selected.sum(axis=1)
    
    mask_gt = class_ids.view(-1, 1) == class_ids.view(1, -1)
    mask_selected_hit = mask_gt & mask_selected
    acc_sample_wise = mask_selected_hit.sum(axis=1)/torch.clamp(mask_selected.sum(axis=1), min=1)

    mean_samplewise_acc_threshwise[i] = acc_sample_wise.mean()
    acc_samplewise_threshwise[:, i] = acc_sample_wise

    for cid in np.unique(class_ids):
        class_mean_acc = acc_sample_wise[class_ids == cid].mean()
        mean_classwise_acc_threshwise[int(cid), i] = class_mean_acc

print(f'Done Processing')
pickle.dump(acc_samplewise_threshwise, open(f'acc_samplewise_threshwise_{sign}.pkl', 'wb'))
pickle.dump(mean_samplewise_acc_threshwise, open(f'mean_samplewise_acc_threshwise_{sign}.pkl', 'wb'))
pickle.dump(mean_classwise_acc_threshwise, open(f'mean_classwise_acc_threshwise_{sign}.pkl', 'wb'))
print(f'Done dumping {sign}')

################## 3. Plot ############### 
signs = ['esfANDiou-cosine', 'esf-20-cosine', 'iou_z']

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
plt.savefig(f'avg_prec_esfANDiou_esf_iou_knn.png')
plt.show()

b=1
