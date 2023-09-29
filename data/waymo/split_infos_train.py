import pickle
import numpy as np

num_splits = 4
parent_split = 'train_short' #'train_short'
processed_data_tag = 'waymo_processed_data_10_short' #'waymo_processed_data_10_short'

infos_pkl_path = f"/home/barza/DepthContrast/data/waymo/{processed_data_tag}_infos_{parent_split}.pkl"
with open(infos_pkl_path, 'rb') as f:
    infos = pickle.load(f) # loads all infos

infos_dict = {} 
seq_list = []
for info in infos:
    sequence_name = info['point_cloud']['lidar_sequence']
    if sequence_name not in infos_dict:
        infos_dict[sequence_name] = []
        seq_list.append(sequence_name)
        #print(sequence_name)
    infos_dict[sequence_name].append(info)

num_seqs_per_split = len(seq_list) // num_splits

seq_split_list =[]
for i in range(num_splits):
    start_idx = i*num_seqs_per_split
    if i < (num_splits -1):
        seq_split_list.append(seq_list[start_idx:start_idx+num_seqs_per_split])
    else:
        seq_split_list.append(seq_list[start_idx:])

for i, seq_in_this_split in enumerate(seq_split_list):
    split_infos=[]
    for seq in seq_in_this_split:
        split_infos += infos_dict[seq]
    
    path = f"/home/barza/DepthContrast/data/waymo/{processed_data_tag}_infos_{parent_split}_{i}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(split_infos, f) # loads all infos
    
