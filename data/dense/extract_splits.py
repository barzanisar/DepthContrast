
#Read split files and append frame ids to a list
#shuffle indices of len of list
#extract 60% of the list to train, 20% to val, 20% to test

from pathlib import Path
import random

from sklearn.utils import shuffle
root_path = Path(__file__).parent.resolve() / 'ImageSets'
splits = ['snow.txt']
weather = 'snow'
split_percentages = [('train', 60), ('val', 15), ('test', 25)]


frame_id_list = []
for split in splits:
    split_path = root_path / split
    split_ids = [x.strip() for x in open(split_path).readlines()] 
    frame_id_list += split_ids
    print(f'Read {split_path} with {len(split_ids)} samples')
    print(f'Frame id list has {len(frame_id_list)} samples')

num_frames = len(frame_id_list)
shuffled_indices = list(range(num_frames))
random.shuffle(shuffled_indices)

start_idx = 0
for s,p in split_percentages:
    num_idx_select = int(p*num_frames/100)
    assert start_idx + num_idx_select < num_frames
    idx_selected = shuffled_indices[start_idx:start_idx+num_idx_select]
    start_idx = start_idx+num_idx_select
    new_split_path = root_path / f'{s}_{weather}_{p}.txt'
    print(f'Writing in {new_split_path} {len(idx_selected)} samples')
    with open(new_split_path, 'w') as f:
        for i, idx in enumerate(idx_selected):
            f.write(frame_id_list[idx])
            if i != len(idx_selected)-1 :
                f.write('\n')






