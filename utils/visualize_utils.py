import os
import numpy as np
import pickle

from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# Compute distances between class clusters
def class_feature_distances(features, labels, tag, class_names=['background', 'objects']):
    '''
    features = (b * npoints, 128) = (8*16384, 128)
    '''
    
    mean_rep_for_all_classes = np.zeros((len(class_names), features.shape[-1]))

    for idx, label in enumerate(class_names):
        # find the samples of the current class in the data
        class_mask = labels == idx
        
        this_class_features = features[class_mask]
        mean_rep_for_all_classes[idx, :] = np.mean(this_class_features, axis=0)


    #for metric in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]:
    metric = "cosine"
    D = pairwise_distances(mean_rep_for_all_classes, metric=metric)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)  # for label size
    sns.heatmap(D, annot=True, annot_kws={"size": 12}, xticklabels=class_names,
                yticklabels=class_names)  # font size
    plt.title(f"{metric} distances between {tag}'s tsne class centers")
    plt.savefig(f"{metric}.png")
    #plt.show()

    #print mean, std and min distances between class cluster centers
    mean = np.mean(D[D > 0])
    std = np.std(D[D > 0])

    # Find most similar classes
    C = D
    C[D == 0] = 1000000
    argmin = np.unravel_index(C.argmin(), C.shape)
    min = C[argmin[0], argmin[1]]
    min_class_names = [class_names[argmin[0]], class_names[argmin[1]]]

    # Find second most similar classes
    C[argmin[0], argmin[1]] = 100000
    C[argmin[1], argmin[0]] = 100000
    argmin2 = np.unravel_index(C.argmin(), C.shape)
    min2 = C[argmin2[0], argmin2[1]]
    min2_class_names = [class_names[argmin2[0]], class_names[argmin2[1]]]

    # Find third most similar classes
    C[argmin2[0], argmin2[1]] = 100000
    C[argmin2[1], argmin2[0]] = 100000
    argmin3 = np.unravel_index(C.argmin(), C.shape)
    min3 = C[argmin3[0], argmin3[1]]
    min3_class_names = [class_names[argmin3[0]], class_names[argmin3[1]]]



    tsne_summary = {"mean_cluster_dist": mean,
                    "std_cluster_dist": std,
                    "min_cluster_dist": [min, min2, min3],
                    "most_similar_classes": [min_class_names, min2_class_names, min3_class_names]}

    print(tsne_summary)
    pickle.dump(tsne_summary, open("cosine_summary.pkl", "wb"))


def visualize_tsne(features, labels, class_names=['background', 'objects']):
    features = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(n_components=2, init='random', random_state=0)
    tsne = reducer.fit_transform(features)
    #tsne = TSNE(n_components=2).fit_transform(features)

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
        # find the samples of the current class in the data
        class_mask = labels == idx
        current_tx = tx[class_mask]
        current_ty = ty[class_mask]
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label, marker='x', linewidth=0.5)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.grid()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"TSNE")

    #filename = f"tsne_{tag}.pdf"
    # finally, show the plot
    #plt.savefig(filename)
    plt.show()