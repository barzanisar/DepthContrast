from tools.visual_utils import open3d_vis_utils as V
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

def get_colors(pc, color_feature=None):
    # create colormap
    if color_feature == 0:
        feature = pc[:, 0]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 1:

        feature = pc[:, 1]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 2:
        feature = pc[:, 2]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 3:
        feature = pc[:, 3]
        min_value = np.min(feature)
        max_value = 255

    elif color_feature == 4:
        feature = pc[:, 4]
        min_value = np.min(feature)
        max_value = np.max(feature)

    else:
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
if __name__ == '__main__':
    #pc = np.fromfile('/media/barza/WD_BLACK/datasets/dense/lidar_hdl64_strongest/2018-02-17_09-52-18_00100.bin', dtype=np.float32).reshape((-1,5))
    pc = np.fromfile('/media/barza/WD_BLACK/datasets/dense/snowfall_simulation/gunn/lidar_hdl64_strongest_rainrate_2/2018-02-17_09-52-18_00100.bin', dtype=np.float32).reshape((-1,5))
    V.draw_scenes(points=pc[:, :3], point_colors = get_colors(pc, color_feature=3))