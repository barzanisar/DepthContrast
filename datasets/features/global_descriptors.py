import pclpy
from pclpy import pcl
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
# from umap import UMAP
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
import math
import torch
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from pathlib import Path
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
import open3d as o3d
from sklearn.neighbors import KDTree



def isFinite(point):
    if math.isnan(point.normal_x) or math.isnan(point.normal_y) or math.isnan(point.normal_z or math.isnan(point.curvature)):
        return False
    else:
        return True


def extract_feats(obj_pc, method='vfh', reject_infinite_normal_objs=False):
    cloud = pcl.PointCloud.PointXYZ.from_array(obj_pc)

    #if method not in ['gasd', 'esf']:
    # print(cloud.size())
    ne = pcl.features.NormalEstimation.PointXYZ_Normal()
    ne.setInputCloud(cloud)

    tree = pcl.search.KdTree.PointXYZ()
    ne.setSearchMethod(tree)

    normals = pcl.PointCloud.Normal()
    ne.setRadiusSearch(0.2)
    ne.compute(normals)
    # print(normals.size())

    cloud_normals = pcl.PointCloud.PointNormal().from_array(
        np.hstack((cloud.xyz, normals.normals, normals.curvature.reshape(-1, 1))))
    
    finiteNormalpts = pcl.PointIndices()

    for i in range(cloud_normals.size()):
        if isFinite(cloud_normals.at(i)):
            finiteNormalpts.indices.append(i)
        else:
            # print(f'cloud_normals[{i}] is not finite\n')
            if reject_infinite_normal_objs:
                return None
    
    if len(finiteNormalpts.indices) < 5:
        return None
    #Extract finite normals and their points
    cloud_finite = pcl.PointCloud.PointXYZ()
    normals_finite = pcl.PointCloud.PointXYZ()

    extract = pcl.filters.ExtractIndices.PointXYZ()
    extract.setInputCloud(cloud)
    extract.setIndices(finiteNormalpts)
    extract.setNegative(False)
    extract.filter(cloud_finite)

    extract.setInputCloud(pcl.PointCloud.PointXYZ(normals.normals))
    extract.setIndices(finiteNormalpts)
    extract.setNegative(False)
    extract.filter(normals_finite)
    normals_finite = pcl.PointCloud.Normal(normals_finite.xyz)

    if method == 'vfh':
        return compute_vfh_desc(cloud_finite, normals_finite)
    elif method == 'gasd':
        return compute_gasd_desc(cloud)
    elif method == 'grsd':
        return compute_grsd_desc(cloud_finite, normals_finite)
    elif method == 'esf':
        return compute_esf_desc(cloud)
    else:
        NotImplementedError

def compute_esf_desc(cloud):
    # Create the GASD estimation class, and pass the input dataset to it
    gasd = pcl.features.ESFEstimation.PointXYZ_ESFSignature640()

    gasd.setInputCloud(cloud)

    descriptors = pcl.PointCloud.ESFSignature640()
    
    gasd.compute(descriptors)

    # plt.plot(descriptors.histogram[0])
    # plt.show()

    if len(descriptors.histogram):
        return descriptors.histogram[0]
    else:
        return None


def compute_gasd_desc(cloud):
    # Create the GASD estimation class, and pass the input dataset to it
    gasd = pcl.features.GASDEstimation.PointXYZ_GASDSignature512()

    gasd.setInputCloud(cloud)

    descriptors = pcl.PointCloud.GASDSignature512()
    
    gasd.compute(descriptors)
    # transform = gasd.getTransform()
    # pts_in_canonical_coord = transform[:3,:3] @ cloud.xyz.T + transform[:3,-1].reshape(3,1)
    # pts_in_canonical_coord = pts_in_canonical_coord.T
    # V.draw_scenes(pts_in_canonical_coord[:,:3])

    # plt.plot(descriptors.histogram[0])
    # plt.show()

    if len(descriptors.histogram):
        return descriptors.histogram[0]
    else:
        return None

def compute_grsd_desc(cloud, normals):
    # Create the GASD estimation class, and pass the input dataset to it
    grsd = pcl.features.GRSDEstimation.PointXYZ_Normal_GRSDSignature21()


    grsd.setInputCloud(cloud)
    grsd.setRadiusSearch(0.2)
    tree = pcl.search.KdTree.PointXYZ()
    grsd.setSearchMethod(tree)
    grsd.setInputNormals(normals)

    descriptors = pcl.PointCloud.GRSDSignature21()
    
    grsd.compute(descriptors)

    # plt.plot(descriptors.histogram[0])
    # plt.show()

    if len(descriptors.histogram):
        return descriptors.histogram[0]
    else:
        return None


def compute_vfh_desc(cloud, normals):
    
    vfh = pcl.features.VFHEstimation.PointXYZ_Normal_VFHSignature308()
    vfh.setInputCloud(cloud)
    vfh.setInputNormals(normals)

    tree = pcl.search.KdTree.PointXYZ()
    vfh.setSearchMethod(tree)

    vfhs = pcl.PointCloud.VFHSignature308()

    vfh.compute(vfhs)

    #print(vfhs.size())


    # plt.plot(vfhs.histogram[0])
    # plt.show()
    if len(vfhs.histogram):
        return vfhs.histogram[0]
    else:
        return None
def visualize_tsne(pointwise_gt_cls, tsne, class_names):

    num_classes = len(class_names)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_classes))

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for idx, label in enumerate(class_names):
        # extract the coordinates of the points of this class only
        current_tx = tx[pointwise_gt_cls == idx]
        current_ty = ty[pointwise_gt_cls == idx]

        # if idx == 0:
        #     continue
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label, marker='x', linewidth=0.7)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.grid()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"TSNE on point features")
    plt.show()
    b=1



def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def draw_feats(X,y, class_names):
    colors = ['blue', 'pink', 'green']
    plt.figure()

    for class_id in range(len(class_names)):
        name = class_names[class_id]
        feats = X[y == class_id] #(nsamples for this class, feat dim)
        plt.plot([0,0], color=colors[class_id], label=name)
        for i in range(feats.shape[0]):
            plt.plot(feats[i], color=colors[class_id])
    plt.legend()
    plt.savefig('gasd.png')
    plt.show()

def check_feat_robustness(points, method, reject_infinite_normal_objs):
    desc0 = extract_feats(points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)
    plt.plot(desc0, label=f'original')
    
    #plt.show()

    angles = np.linspace(-90, 90, 2)
    for i in angles:
        angle= np.deg2rad(i)
        print(f'Extracting feats for {i} deg rot')
        rotated_points = rotate_points_along_z(points[np.newaxis, :, :], np.array([angle]))[0]
        desc = extract_feats(rotated_points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)
        print(f"Euclidean dist for rot_{i}_deg: {np.linalg.norm(desc0-desc)}")
        #V.draw_scenes(points=rotated_points[:,:4])
        plt.plot(desc, label=f'rot_{i}_deg')
    
    plt.legend()
    plt.show()
    plt.plot(desc0, label=f'original')

    trans = np.linspace(1, 10, 2)
    for i in trans:
        trans_points = np.copy(points[:,:3])
        trans_points[:,:2] = trans_points[:,:2]+i
        desc = extract_feats(trans_points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)
        print(f"Euclidean dist for trans_{i}_m: {np.linalg.norm(desc0-desc)}")
        #V.draw_scenes(points=trans_points[:,:4])
        plt.plot(desc, label=f'trans_{i}_m')

    plt.legend()
    plt.show()
    plt.plot(desc0, label=f'original')

    scales = np.linspace(0.95, 1.05, 2)
    for i in scales:
        scaled_points = points[:,:3]*i
        desc = extract_feats(scaled_points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)
        print(f"Euclidean dist for scale_{i}: {np.linalg.norm(desc0-desc)}")
        #V.draw_scenes(points=scaled_points[:,:4])
        plt.plot(desc, label=f'scale_{i}_m')


    plt.legend()
    plt.show()
    b=1

def extract_all_feats(dbinfos, method, reject_infinite_normal_objs, class_names, root_path):
    obj_class_idx = []
    desc_mat = []
    valid_info_idx = [] 

    for class_idx, class_name in enumerate(class_names):
        count_valid = 0
        for i, info in tqdm(enumerate(dbinfos[class_name])):
            path = root_path + info['path']
            obj_points = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
            if len(obj_points) > 5:
                # print(f'Computing vfh: {class_names} {i}')
                desc = extract_feats(obj_points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)
                if desc is not None:
                    desc_mat.append(desc)
                    valid_info_idx.append(i)
                    obj_class_idx.append(class_idx)
                    count_valid+=1
        
        print(f'Found {count_valid} Feats for {class_name}#########')

    desc_mat = np.array(desc_mat)
    valid_info_idx = np.array(valid_info_idx)
    obj_class_idx = np.array(obj_class_idx)

    data_dict = {'labels': obj_class_idx,
                 'feats': desc_mat,
                 'valid_info_idx': valid_info_idx}

    pickle.dump(data_dict, open(f"{method}_feats_data_dict.pkl", "wb"))
    print(f'Saved {method}_feats_data_dict.pkl')

    # tsne = TSNE(n_components=2).fit_transform(desc_mat) #N obj, 308 feature vector

    # pickle.dump(tsne, open(f"{method}_tsne.pkl", "wb"))
    # print(f'Saved {method}_tsne.pkl')


if __name__ == "__main__":

    #path = '/home/barza/DepthContrast/data/waymo/waymo_processed_data_10_short_waymo_dbinfos_train_sampled_1.pkl'
    ROOT_PATH = '/home/barza/OpenPCDet/data/waymo/'
    path = ROOT_PATH + 'waymo_processed_data_v_1_2_0_waymo_dbinfos_train_sampled_100.pkl'
    with open(path, 'rb') as f:
        dbinfos = pickle.load(f)
    class_names=['Vehicle', 'Pedestrian', 'Cyclist']
    METHOD='vfh' #vfh 96.5, esf 89.1, gasd 82.3
    reject_infinite_normal_objs=False
    random_seed = 42 
    EXTRACT_ALL_INFOS_FEATS=False
    EVAL_FEATS=False
    EVAL_FEATS_DISTANCES=False
    EVAL_PRECISION_K=False
    EVAL_PRECISION_IOU=False


    CHECK_ROT_TRANS_SCALE_INVARIANCE=False
    COMPARE_FEATS=True
    VISUALIZE_TSNE=False
    VISUALIZE_FEATS=False
    CROSS_VAL=False

    # ########### check feat robustness
    if CHECK_ROT_TRANS_SCALE_INVARIANCE:
        sample=100
        class_name = 'Vehicle'
        path = ROOT_PATH + dbinfos[class_name][sample]['path']
        points = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
        #V.draw_scenes(points=points[:,:4])
        check_feat_robustness(points, METHOD, reject_infinite_normal_objs)

    # ########### compare feats 
    if COMPARE_FEATS:
        sample=40 #100 Veh
        obj_class1 = 'Vehicle'
        path = ROOT_PATH + dbinfos[obj_class1][sample]['path']
        points = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
        obj_points1 = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
        #V.draw_scenes(points=obj_points1[:,:4])
        desc1 = extract_feats(obj_points1[:,:3], method=METHOD, reject_infinite_normal_objs=reject_infinite_normal_objs)


        sample=42 #50 Ped
        obj_class2 = 'Pedestrian'
        path = ROOT_PATH + dbinfos[obj_class2][sample]['path']
        obj_points2 = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
        #V.draw_scenes(points=obj_points2[:,:4])
        desc2 = extract_feats(obj_points2[:,:3], method=METHOD, reject_infinite_normal_objs=reject_infinite_normal_objs)
        
        # plt.plot(desc1, label=obj_class1+'1')
        # plt.plot(desc2, label=obj_class2+'2')
        # plt.title('Euclidean Distance: {:.2f}'.format(np.linalg.norm(desc1-desc2)))
        # plt.legend()
        # plt.show()

        normalized_d1=desc1/np.linalg.norm(desc1)
        normalized_d2=desc2/np.linalg.norm(desc2)
        cosine_dist = 1 - np.dot(normalized_d1, normalized_d2)
        euc_dist = np.linalg.norm(normalized_d1-normalized_d2)


        plt.plot(normalized_d1, label=obj_class1+'1')
        plt.plot(normalized_d2, label=obj_class2+'2')
        plt.title('Normalized vec Euclidean Distance: {:.2f}, cosine_dist: {:.2f}'.format(euc_dist, cosine_dist))
        plt.legend()
        plt.show()
    
    ################## Extract feats
    if EXTRACT_ALL_INFOS_FEATS:
        extract_all_feats(dbinfos, METHOD, reject_infinite_normal_objs, class_names, ROOT_PATH)
    
    
    # ############ load and visualize tsne
    if VISUALIZE_TSNE:
        data_dict = pickle.load(open(f"{METHOD}_feats_data_dict.pkl", "rb"))
        tsne = pickle.load(open("tsne.pkl", "rb"))
        visualize_tsne(data_dict['labels'], tsne, class_names)

    # ############ visualize features
    if VISUALIZE_FEATS:
        data_dict = pickle.load(open(f"{METHOD}_feats_data_dict.pkl", "rb"))
        X = data_dict['feats'] #(nsamples, nfeats)
        y = data_dict['labels'] #(nsamples,)
        draw_feats(X,y, class_names)
    
    if EVAL_PRECISION_IOU:
        n_samples = 0
        for class_idx, class_name in enumerate(class_names):
            n_samples += len(dbinfos[class_name])

        lwhz_mat = np.zeros((n_samples, 4))
        for class_idx, class_name in enumerate(class_names):    
            for i, info in tqdm(enumerate(dbinfos[class_name])):
                lwhz_mat[i, :3] = info['box3d_lidar'][3:6]
                lwhz_mat[i, 3] = info['box3d_lidar'][2]
        

        l_a= lwhz_mat[:,0].reshape(-1, 1)
        w_a= lwhz_mat[:,1].reshape(-1, 1)
        h_a= lwhz_mat[:,2].reshape(-1, 1)
        z_a= lwhz_mat[:,3].reshape(-1, 1)

        l_b = l_a.reshape(1, -1)
        w_b = w_a.reshape(1, -1)
        h_b = h_a.reshape(1, -1)
        z_b = z_a.reshape(1, -1)

        boxes_a_height_max = (z_a + h_a / 2).reshape(-1, 1) #col
        boxes_a_height_min = (z_a - h_a / 2).reshape(-1, 1) #col
        boxes_b_height_max = boxes_a_height_max.reshape(1, -1) #row
        boxes_b_height_min = boxes_a_height_min.reshape(1, -1) #row

        max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min) #(Npos, Nneg)
        min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max) #(Npos, Nneg)
        overlaps_h = torch.clamp(min_of_max - max_of_min, min=0) # torch.min(h, h_neg)  #(Npos, Nneg) height overlaps between each pos and neg sample

        vol_a = (l_a*w_a*h_a).view(-1, 1) # col: Nposx1
        vol_b = (l_b * w_b * h_b).view(1, -1) # row: 1xK
        overlap_vol = torch.min(l_a, l_b) *  torch.min(w_a, w_b) * overlaps_h # NxK
        iou3d = overlap_vol / torch.clamp(vol_a + vol_b - overlap_vol, min=1e-6) # NxK



    if EVAL_PRECISION_K:
        data_dict = pickle.load(open(f"{METHOD}_feats_data_dict.pkl", "rb"))
        X = data_dict['feats'] #(nsamples, nfeats)
        y = data_dict['labels'] #(nsamples,)
        # n_clusters = len(class_names)
        k_neighbors = np.arange(1, 1000)

        for metric in ["euclidean", "cityblock"]:
            print(f'Processing metric: {metric}')
            kd_tree = KDTree(X, metric=metric)
            acc_dict = {}
            for i in range(len(class_names)):
                n_samples_this_class = len(X[y==i])
                acc_dict[i]=np.zeros((n_samples_this_class, len(k_neighbors)))

            for i, class_name in enumerate(class_names):
                X_cls = X[y==i]
                print(f'Processing class: {class_name}')
                for j in tqdm(range(len(X_cls))):
                    dist, ind = kd_tree.query(X_cls[j].reshape(1, -1), k=k_neighbors[-1])
                    count_true=0
                    acc_mask = y[ind] == i
                    acc_dict[i][j, :] = np.cumsum(acc_mask)/k_neighbors
                    
            pickle.dump(acc_dict, open(f"{metric}_knn_desc.pkl", "wb"))
        
        styles = ['-', '--']
        colors = ['r', 'b', 'g']
        for style_idx, metric in enumerate(["euclidean", "cityblock"]):
            acc_dict = pickle.load(open(f"{metric}_knn_desc.pkl", "rb"))
            for i, class_name in enumerate(class_names):
                avg_acc = np.mean(acc_dict[i], axis=0)
                plt.plot(k_neighbors, avg_acc, styles[style_idx], color=colors[i], label=f'{class_name}_{metric}')
            
        plt.legend()
        plt.xlabel('Knn')
        plt.ylabel('Accuracy in Knn')
        plt.show()
        



        
    ##################### Eval feats distances 
    if EVAL_FEATS_DISTANCES:
        data_dict = pickle.load(open(f"{METHOD}_feats_data_dict.pkl", "rb"))
        X = data_dict['feats'] #(nsamples, nfeats)
        y = data_dict['labels'] #(nsamples,)
        n_clusters = len(class_names)
        # Plot the distances
        for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
            avg_dist = np.zeros((n_clusters, n_clusters))
            plt.figure(figsize=(5, 4.5))
            for i in range(n_clusters):
                for j in range(n_clusters):
                    avg_dist[i, j] = pairwise_distances(
                        X[y == i], X[y == j], metric=metric
                    ).mean()
            avg_dist /= avg_dist.max()
            print(f'pair_dist_{METHOD}_{metric}')
            print(avg_dist)
            for i in range(n_clusters):
                for j in range(n_clusters):
                    t = plt.text(
                        i,
                        j,
                        "%5.3f" % avg_dist[i, j],
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                    t.set_path_effects(
                        [PathEffects.withStroke(linewidth=5, foreground="w", alpha=0.5)]
                    )

            plt.imshow(avg_dist, interpolation="nearest", cmap="cividis", vmin=0)
            plt.xticks(range(n_clusters), class_names, rotation=45)
            plt.yticks(range(n_clusters), class_names)
            plt.colorbar()
            plt.suptitle(f"{METHOD}: Interclass {metric} distances", size=18, y=1)
            plt.tight_layout()
            plt.savefig(f'pair_dist_{METHOD}_{metric}.png')

    # ############ classify features
    if EVAL_FEATS:
        data_dict = pickle.load(open(f"{METHOD}_feats_data_dict.pkl", "rb"))
        X = data_dict['feats'] #(nsamples, nfeats)
        y = data_dict['labels'] #(nsamples,)
        classifier = svm.SVC(kernel='poly', class_weight='balanced', random_state=random_seed)
        #classifier = MLPClassifier(hidden_layer_sizes=(128, 128, 3), early_stopping=True, random_state=random_seed)
        clf = make_pipeline(classifier) #(StandardScaler(), classifier)

        if CROSS_VAL:
            cv_scores = cross_val_score(clf, X, y, cv=5)
            mean_score = np.mean(cv_scores)
            print(f'{METHOD} val score: {cv_scores}\t mean score: {mean_score}')
        else:
            # cv_scores = cross_val_score(clf, X[y!=0], y[y!=0], cv=5)
            # mean_score = np.mean(cv_scores)
            # print(f'{METHOD} val score: {cv_scores}\t mean score: {mean_score}')

            X_train, X_test, y_train, y_test = [], [], [], []
            for class_id in range(len(class_names)):
                # if class_id == 0:
                #     continue
                name = class_names[class_id]
                X_cls = X[y == class_id]
                y_cls = y[y == class_id]
                X_tr_cls, X_tst_cls, y_tr_cls, y_tst_cls = train_test_split(X_cls, y_cls, test_size=0.33, random_state=random_seed)
                print(f'{name} has {len(X_tr_cls)} train samples, {len(X_tst_cls)} test samples')
                X_train.append(X_tr_cls)
                X_test.append(X_tst_cls)
                y_train.append(y_tr_cls)
                y_test.append(y_tst_cls)

            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)

            clf.fit(X_train, y_train) #'poly', 'rbf' cross_val_score(clf, X, y, cv=5)
            # s = pickle.dumps(clf)
            # clf2 = pickle.loads(s)

            for class_id in range(len(class_names)):
                # if class_id == 0:
                #     continue

                name = class_names[class_id]
                y_t = y_test[class_id] #-1
                X_t = X_test[class_id] #-1
                score = clf.score(X_t, y_t)
                print(f'{METHOD} val score for {name}:  {score}')


        
