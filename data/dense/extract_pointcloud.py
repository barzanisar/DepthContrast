# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


LIDAR_DIR = 'lidar_hdl64_strongest'
SPLIT = 'train_clear_60'
TRAIN_SPLIT = f'./ImageSets/{SPLIT}.txt'

if __name__ == '__main__':

    sample_id_list = ['_'.join(x.strip().split(',')) for x in open(TRAIN_SPLIT).readlines()]
    for i, sample_id in enumerate(sample_id_list):
        sample_id_list[i] = LIDAR_DIR + '/' + sample_id_list[i] + ".bin"

    np.save(f"./ImageSets/{SPLIT}.npy", sample_id_list) # only store paths as "waymo_processed_data_10/seq_name/frame.npy"
