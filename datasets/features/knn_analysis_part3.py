import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datasets.features import global_descriptors
from torch import nn


quantile_thresh = np.array([0.5, 1, 5])/100 #np.array([0.01, 0.03, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5])/100 
shape_dim_dict = {'esf':640, 'vfh': 308, 'gasd': 512}
num_samples = 15000
shape_desc_min_pts = 20


split_txt_file = '/home/barza/DepthContrast/data/waymo/ImageSets/train_ten.txt'
lidar_data_path = Path('/home/barza/DepthContrast/data/waymo/waymo_processed_data_v_1_2_0')
cluster_root_path = Path('/home/barza/DepthContrast/data/waymo/waymo_processed_data_v_1_2_0_clustered')
seglabels_root_path = Path('/home/barza/DepthContrast/data/waymo/waymo_processed_data_v_1_2_0_labels')
results_save_path = Path('/home/barza/DepthContrast/knn_analysis')

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


def load_desc_class_ids(sign, mask=None):
    desc_mat = pickle.load(open(results_save_path / f'desc_mat_{sign}.pkl', 'rb'))
    class_ids = pickle.load(open(results_save_path / f'class_ids_{sign}.pkl', 'rb'))
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


desc_esf, class_ids_esf, mask = load_desc_class_ids(f'esf')
desc_iou, class_ids_iou, _ = load_desc_class_ids('iou', mask)
sign=f'esf_{shape_desc_min_pts}_ANDiou-cosine'
assert torch.equal(class_ids_esf, class_ids_iou)
assert desc_esf.shape[0] == desc_iou.shape[0]
class_ids = class_ids_esf

num_obj_pts = pickle.load(open(results_save_path / f'num_obj_pts.pkl', 'rb'))
num_obj_pts = num_obj_pts[mask][:num_samples]
mask = num_obj_pts>shape_desc_min_pts
desc_mat_esf_min_pts = desc_esf[mask]
desc_mat_iou_min_pts = desc_iou[mask]
class_ids = class_ids[mask]

iou3d = compute_iou_mat(desc_mat_iou_min_pts, iou_z=True)
iou_dist = 1-iou3d
esf_dist = compute_shape_desc_dist_mat(desc_mat_esf_min_pts, method='cosine')

n_samples = esf_dist.shape[0]
acc_samplewise_threshwise = np.zeros((n_samples, len(quantile_thresh))) # rows are samples, cols are thresholds 
mean_samplewise_acc_threshwise = np.zeros(len(quantile_thresh))
mean_classwise_acc_threshwise = np.zeros((len(WAYMO_LABELS), len(quantile_thresh))) #includes undefined class 

for i, thresh in enumerate(quantile_thresh):
    # avg_acc = []
    print(f'Quantile thresh {thresh}')
    row_wise_quantiles = torch.quantile(iou_dist, thresh, dim=1, keepdim=True)
    iou_dist = iou_dist.fill_diagonal_(float('inf'))
    mask_selected_iou = iou_dist < row_wise_quantiles.repeat(1, n_samples)
    row_wise_quantiles = torch.quantile(esf_dist, thresh, dim=1, keepdim=True)
    esf_dist = esf_dist.fill_diagonal_(float('inf'))
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
pickle.dump(acc_samplewise_threshwise, open(results_save_path / 'pickles' / f'acc_samplewise_threshwise_{sign}.pkl', 'wb'))
pickle.dump(mean_samplewise_acc_threshwise, open(results_save_path / 'pickles' / f'mean_samplewise_acc_threshwise_{sign}.pkl', 'wb'))
pickle.dump(mean_classwise_acc_threshwise, open(results_save_path / 'pickles' / f'mean_classwise_acc_threshwise_{sign}.pkl', 'wb'))
print(f'Done dumping {sign}')

################## 3. Plot ############### 
signs = [sign] # , 'iou', 'esf'

plt.figure()
for sign in signs:
    mean_samplewise_acc_threshwise = pickle.load(open(results_save_path / 'pickles' / f'mean_samplewise_acc_threshwise_{sign}.pkl', 'rb'))
    #plt.plot(quantile_thresh*100, mean_samplewise_acc_threshwise, label=f'{sign}')
    plt.plot(np.round(quantile_thresh*num_samples), mean_samplewise_acc_threshwise, label=f'{sign}')
    print(f'{sign}: {mean_samplewise_acc_threshwise}')

plt.ylabel('Average Precision for K Nearest Neighbors')
# plt.xlabel('Quantiles %')
plt.xlabel('K')
plt.title(f'Average Precision KNN')

plt.grid()
plt.legend()
plt.savefig(results_save_path / 'plots' / f'avg_prec_{sign}_knn.png')
plt.show()

# def plotting(sign, class_ids):
#     mean_classwise_acc_threshwise = pickle.load(open(f'mean_classwise_acc_threshwise_{sign}.pkl', 'rb'))
#     NUM_COLORS = 22
#     cm = plt.get_cmap('gist_rainbow')
#     fig = plt.figure(figsize=(20,8))
#     ax = fig.add_subplot(111)
#     ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
#     LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
#     NUM_STYLES = len(LINE_STYLES)
#     print(f'Plotting classwise precision for {sign}')

#     for i, cid in enumerate(np.unique(class_ids)):  
#         r = np.round(np.random.rand(),1)
#         g = np.round(np.random.rand(),1)
#         b = np.round(np.random.rand(),1)
#         #lines = ax.plot(quantile_thresh*100, mean_classwise_acc_threshwise[int(cid), :], label=WAYMO_LABELS[int(cid)])
#         lines = ax.plot(np.round(quantile_thresh*num_samples), mean_classwise_acc_threshwise[int(cid), :], label=WAYMO_LABELS[int(cid)])
        
#         lines[0].set_color(cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
#         lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
#         print(f'{WAYMO_LABELS[int(cid)]}: {mean_classwise_acc_threshwise[int(cid), :]}')
#         # plt.plot(quantile_thresh, mean_classwise_acc_threshwise[int(cid), :], label=WAYMO_LABELS[int(cid)], color=[r,g,b]) 
#     plt.ylabel('Precision for K Nearest Neighbors')
#     plt.xlabel('K')
#     # plt.xlabel('Quantiles %')

#     plt.title(f'Classwise Average Precision of {sign}-based KNN')
#     plt.legend()
#     plt.grid()

#     plt.savefig(results_save_path / 'plots' /f'classwise_avg_prec_{sign}_knn.png')
#     plt.show()
#     print(f'Done plotting {sign}')

# plotting(sign, class_ids)

