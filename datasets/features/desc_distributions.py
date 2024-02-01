import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datasets.features import global_descriptors
from torch import nn


quantile_thresh = np.array([0.1, 0.5, 1, 2, 3, 4])/100 
shape_dim_dict = {'esf':640, 'vfh': 308, 'gasd': 512}
desc_types = ['iou', 'esf']
num_samples = 15000
random_seed = 80
shape_desc_min_pts = [50] # 



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


def compute_iou_mat(lwhz_mat, iou_z=True):
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

    desc_mat = torch.from_numpy(desc_mat)
    class_ids = torch.from_numpy(class_ids)
    return desc_mat, class_ids, mask

def analyse_knns(dists_list, class_ids, sign):
    n_samples = class_ids.shape[0]
    acc_samplewise_threshwise = np.zeros((n_samples, len(quantile_thresh))) # rows are samples, cols are thresholds 
    mean_samplewise_acc_threshwise = np.zeros(len(quantile_thresh))
    mean_classwise_acc_threshwise = np.zeros((len(WAYMO_LABELS), len(quantile_thresh))) #includes undefined class 

    for i, thresh in enumerate(quantile_thresh):
        print(f'Quantile thresh {thresh}')

        row_wise_quantiles = torch.quantile(dists_list[0], thresh, dim=1, keepdim=True)
        dists_list[0] = dists_list[0].fill_diagonal_(float('inf'))
        mask_selected_0 = dists_list[0] < row_wise_quantiles.repeat(1, n_samples)
        mask_selected = mask_selected_0 
        if len(dists_list) > 1:

            row_wise_quantiles = torch.quantile(dists_list[1], thresh, dim=1, keepdim=True)
            dists_list[1] = dists_list[1].fill_diagonal_(float('inf'))
            mask_selected_1 = dists_list[1] < row_wise_quantiles.repeat(1, n_samples)
            mask_selected = mask_selected_0 & mask_selected_1
        
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


def image_histogram_equalization(flattened_image, number_bins=50):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(flattened_image, number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(flattened_image, bins[:-1], cdf)

    return image_equalized, cdf

################## 1. Extract Descs ###############  


desc_esf, class_ids_esf, notzeroclass_notnanshape_mask = load_desc_class_ids(f'esf')
desc_iou, class_ids_iou, _ = load_desc_class_ids('iou', notzeroclass_notnanshape_mask)
assert torch.equal(class_ids_esf, class_ids_iou)
assert desc_esf.shape[0] == desc_iou.shape[0]
class_ids = class_ids_esf

num_obj_pts = pickle.load(open(results_save_path / f'num_obj_pts.pkl', 'rb'))
num_obj_pts = num_obj_pts[notzeroclass_notnanshape_mask]

# Randomly select 15k pts
indices = np.random.choice(desc_esf.shape[0], size=num_samples, replace=False)
desc_esf = desc_esf[indices]
desc_iou = desc_iou[indices]
class_ids = class_ids[indices]
num_obj_pts = num_obj_pts[indices]
# desc_esf = desc_esf[:num_samples]
# desc_iou = desc_iou[:num_samples]
# class_ids = class_ids[:num_samples]
# num_obj_pts = num_obj_pts[:num_samples]

for min_pts in shape_desc_min_pts:
    #Select clusters with min pts
    min_pts_mask = num_obj_pts>min_pts
    desc_mat_esf_min_pts = desc_esf[min_pts_mask]
    desc_mat_iou_min_pts = desc_iou[min_pts_mask]
    class_ids_min_pts = class_ids[min_pts_mask]
    
    iou3d = compute_iou_mat(desc_mat_iou_min_pts, iou_z=True)
    iou_dist = 1-iou3d
    esf_dist = compute_shape_desc_dist_mat(desc_mat_esf_min_pts, method='cosine')

    mask_gt = class_ids_min_pts.view(-1, 1) == class_ids_min_pts.view(1, -1)
    # # mask_gt = mask_gt.fill_diagonal_(False)
    # iou_dist_of_true_classes = iou_dist.flatten()[torch.logical_not(mask_gt.flatten())]
    # esf_dist_of_true_classes = esf_dist.flatten()[torch.logical_not(mask_gt.flatten())]
    mask_gt = mask_gt.fill_diagonal_(False)
    iou_dist_of_true_classes = iou_dist.flatten()[mask_gt.flatten()]
    esf_dist_of_true_classes = esf_dist.flatten()[mask_gt.flatten()]
    iou_plus_esf_dist = 0.5*iou_dist_of_true_classes[indices] + 0.5*esf_dist_of_true_classes[indices]
    iou_plus_esf_dist = (iou_plus_esf_dist - iou_plus_esf_dist.min()) / (iou_plus_esf_dist.max() - iou_plus_esf_dist.min())
    fig, ax = plt.subplots(1,1)
    # ax = ax.ravel()
    indices = np.random.choice(esf_dist_of_true_classes.shape[0], size=100000, replace=False)
    ax.hist(iou_plus_esf_dist, bins=50)
    ax.set_title('0.4*IoU+0.6*ESF dist Histogram for true classes')
    # ax[0].hist(iou_dist_of_true_classes[indices], bins=50)
    # ax[0].set_title('IoU dist Histogram for true classes')
    # ax[1].hist(esf_dist_of_true_classes[indices], bins=50)
    # ax[1].set_title('ESF dist Histogram for true classes')
    # ax[2].hist(image_histogram_equalization(iou_dist_of_true_classes[indices]), bins=50)
    # ax[2].set_title('IoU dist Equalized Histogram for true classes')
    # ax[3].hist(image_histogram_equalization(esf_dist_of_true_classes[indices]), bins=50)
    # ax[3].set_title('ESF dist Equalized Histogram for true classes')

    save_path = results_save_path / 'plots' / f'iou_plus_esf_desc_distribution.png'
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



