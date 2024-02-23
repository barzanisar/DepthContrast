import numpy as np
import MinkowskiEngine as ME
import torch

def array_to_sequence(batch_data):
    return [ row for row in batch_data]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]

# def array_to_sequence(batch_data, batch_size):
#     return [batch_data[batch_data[:,0] == bidx][:, 1:] for bidx in range(batch_size) ]

# def array_to_torch_sequence(batch_data, batch_size):
#     return [ torch.from_numpy(batch_data[batch_data[:,0] == bidx][:, 1:]).float() for bidx in range(batch_size) ]

def list_segments_points(p_coord, p_feats, labels):
    c_coord = []
    c_feats = []

    seg_batch_count = 0

    for batch_num in range(len(labels)):
        for segment_lbl in np.unique(labels[batch_num]):
            if segment_lbl == -1:
                continue

            batch_ind = p_coord[:,0] == batch_num
            segment_ind = labels[batch_num] == segment_lbl

            # we are listing from sparse tensor, the first column is the batch index, which we drop
            segment_coord = p_coord[batch_ind][segment_ind][:,:]
            segment_coord[:,0] = seg_batch_count
            seg_batch_count += 1

            segment_feats = p_feats[batch_ind][segment_ind]

            c_coord.append(segment_coord)
            c_feats.append(segment_feats)

    seg_coord = torch.vstack(c_coord) # (num pts in all segs in all pcs, 4=idx of segment,xyz vox coord)
    seg_feats = torch.vstack(c_feats) # (num pts in all segs in all pcs, 96)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return ME.SparseTensor(
                features=seg_feats,
                coordinates=seg_coord,
                device=device,
            )

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32) # np array (8, 20k, 3=xyz vox coord in float 64) -> torch tensor cpu (8x20k, 4=b_id,xyz vox coord in float 32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:] # np array (8, 20k, 4=xyzi pts) ->  torch tensor cpu (8x20k, 4=xyzi pts in float 32)

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]  #np array (2, 80k) ->  torch tensor cpu (2x80k labels in float 32)
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label.cuda() #sparse gpu tensor and seg_labels in gpu

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ) # sparse tensor in gpu

def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False, frame_id=None):
    p_feats = point_set.copy() #xyzi for each pt
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1) #[xmin, ymin, zmin] -> p_coord = voxel_coords for each pt

    _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    if len(mapping) > num_points:
        if deterministic:
            # for reproducibility we set the seed
            np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord[mapping], p_feats[mapping], labels[mapping] #(20K, 3=xyz voxel coord), (20K, 4=xyzi pt coord), (20K,)

def collate_points_to_sparse_tensor(pi_coord, pi_feats, pj_coord, pj_feats):
    # voxelize on a sparse tensor
    points_i = numpy_to_sparse_tensor(pi_coord, pi_feats) # (bs=8, 20k pts, xyz vox coords), (bs=8, 20k pts, xyzi pt coord and intensity)
    points_j = numpy_to_sparse_tensor(pj_coord, pj_feats)
    # ptsi is a sparse tensor: C=coordinates= (bs=8x20k pts, 4=b_id,xyz voox coords), F=features=(8x20k, xyzi pt coord and intensity)
    return points_i, points_j

def sparse_downstream_collator(batch):
    batch_size = len(batch)
    
    # points = np.asarray([x['points'][:,:-1] for x in batch]) # (bs, 20k, xyzi)
    # voxel_coords = np.asarray([x['voxel_coords'] for x in batch]) # (bs, 20k, xyz vox coord)
    seg_labels = np.concatenate([x["points"][:,-1] for x in batch], axis=0) # (N1, N2, ..., Nbs)
    
    points = [x['points'][:,:-1] for x in batch] # (bs, 20k, xyzi)
    voxel_coords = [x['voxel_coords'] for x in batch] # (bs, 20k, xyz vox coord)

    # sparse_points = numpy_to_sparse_tensor(voxel_coords, points) # sparse gpu tensors -> (C:(8, 20k, 4=b_id, xyz voxcoord), F:(8, 20k, 4=xyzi pts))

    output_batch = {'input': 
                    {
                    #  'sparse_points': sparse_points,
                     'points': points, 
                     'voxel_coords': voxel_coords,
                     'seg_labels': seg_labels,
                     'batch_size': batch_size}
                     }
    
    # # if downstream task is detection:
    # if 'gt_boxes' in batch[0]:
    #     pt_wise_gtbox_idxs = np.concatenate([x["pt_wise_gtbox_idxs"] for x in batch], axis=0) # (N1, N2, ..., Nbs)
        
    #     # make gt boxes in shape (batch size, max gt box len, 8) (xyz, lwh, rz, class label)
    #     max_gt = max([len(x['gt_boxes']) for x in batch])
    #     batch_gt_boxes3d = np.zeros((batch_size, max_gt, batch[0]['gt_boxes'].shape[-1]), dtype=np.float32) # (batch size = 2, max_gt_boxes in a pc in this batch = 67, 8)
    #     for k in range(batch_size):
    #         batch_gt_boxes3d[k, :batch[k]['gt_boxes'].__len__(), :] = batch[k]['gt_boxes']


    #     output_batch['input'].update(
    #                     {'gt_boxes': batch_gt_boxes3d,
    #                     'pt_wise_gtbox_idxs': pt_wise_gtbox_idxs})

    return output_batch

def sparse_moco_collator(batch):
    batch_size = len(batch)
    shape_descs_required = 'shape_desc_cluster_ids' in batch[0]
    
    # #append batchidx, x,y,z,i
    # coords = []
    # coords_moco = []
    # for bidx, x in enumerate(batch):
    #     coor_pad = np.pad(x['points'][:,:4], ((0, 0), (1, 0)), mode='constant', constant_values=bidx) 
    #     coor_pad_moco = np.pad(x['points_moco'][:,:4], ((0, 0), (1, 0)), mode='constant', constant_values=bidx) 
    #     coords.append(coor_pad)
    #     coords_moco.append(coor_pad_moco)
    # points = np.concatenate(coords, axis=0) #(N1 + ... Nbs, 5=bxyzi)
    # points_moco = np.concatenate(coords_moco, axis=0) #(N1 + ... Nbs, 5=bxyzi)

    # #append batchidx, x,y,z vox coord
    # coords = []
    # coords_moco = []
    # for bidx, x in enumerate(batch):
    #     coor_pad = np.pad(x['voxel_coords'][:,:4], ((0, 0), (1, 0)), mode='constant', constant_values=bidx) 
    #     coor_pad_moco = np.pad(x['voxel_coords_moco'][:,:4], ((0, 0), (1, 0)), mode='constant', constant_values=bidx) 
    #     coords.append(coor_pad)
    #     coords_moco.append(coor_pad_moco)
    # voxel_coords = np.concatenate(coords, axis=0) #(N1 + ... Nbs, 5=bxyz vox coord)
    # voxel_coords_moco = np.concatenate(coords_moco, axis=0) #(N1 + ... Nbs, 5=bxyz vox coord)

    points = [x['points'][:,:-1] for x in batch] # (bs, 20k, xyzi)
    points_moco = [x['points_moco'][:,:-1] for x in batch] # (bs, 20k, xyzi)

    voxel_coords = [x['voxel_coords'] for x in batch] # (bs, 20k, xyz vox coord)
    voxel_coords_moco = [x['voxel_coords_moco'] for x in batch] # (bs, 20k, xyz vox coord)

    # points = np.asarray([x['points'][:,:-1] for x in batch]) # (bs, 20k, xyzi)
    # points_moco = np.asarray([x['points_moco'][:,:-1] for x in batch]) # (bs, 20k, xyzi)

    # voxel_coords = np.asarray([x['voxel_coords'] for x in batch]) # (bs, 20k, xyz vox coord)
    # voxel_coords_moco = np.asarray([x['voxel_coords_moco'] for x in batch]) # (bs, 20k, xyz vox coord)

    cluster_ids = [x["points"][:,-1] for x in batch] # ([N1], [N2], ..., [Nbs])
    gt_boxes_cluster_ids = [x["gt_boxes_cluster_ids"] for x in batch]
    
    cluster_ids_moco = [x["points_moco"][:,-1] for x in batch]
    gt_boxes_moco_cluster_ids = [x["gt_boxes_moco_cluster_ids"] for x in batch]
    
    common_cluster_ids = []
    if shape_descs_required:
        shape_desc_cluster_ids = [x["shape_desc_cluster_ids"] for x in batch]
    for i in range(batch_size):
        # get unique labels from pcd_i and pcd_j
        cluster_i = cluster_ids[i]
        cluster_j = cluster_ids_moco[i]
        unique_i = np.unique(cluster_i)
        unique_j = np.unique(cluster_j)

        # get labels present on both pcd (intersection)
        common_ij = np.intersect1d(unique_i, unique_j)[1:]
        if shape_descs_required:
            common_ij_which_has_shape_feats = np.isin(common_ij, shape_desc_cluster_ids[i])
            common_ij = common_ij[common_ij_which_has_shape_feats]

        common_cluster_ids.append(common_ij)
            
        # labels not intersecting both pcd are assigned as -1 (unlabeled)
        cluster_i[np.in1d(cluster_i, common_ij, invert=True)] = -1
        cluster_j[np.in1d(cluster_j, common_ij, invert=True)] = -1

        cluster_ids[i] = cluster_i
        cluster_ids_moco[i] = cluster_j


    common_cluster_gtbox_idx=[]
    common_cluster_gtbox_moco_idx=[]
    unscaled_lwhz = [x["unscaled_lwhz_cluster_id"] for x in batch]
    gt_box_class_ids = [x["gt_boxes"][:,-1] for x in batch]
    common_unscaled_lwhz = []
    common_gt_box_class_ids=[]
    for i in range(batch_size):
        common_box_idx = np.where(np.isin(gt_boxes_cluster_ids[i], common_cluster_ids[i]))[0]
        common_box_idx_moco = np.where(np.isin(gt_boxes_moco_cluster_ids[i], common_cluster_ids[i]))[0]
        assert (gt_boxes_cluster_ids[i][common_box_idx] - gt_boxes_moco_cluster_ids[i][common_box_idx_moco]).sum() == 0
        assert (gt_boxes_cluster_ids[i][common_box_idx] - common_cluster_ids[i]).sum() == 0
        assert (gt_boxes_cluster_ids[i][common_box_idx] - unscaled_lwhz[i][common_box_idx, -1]).sum() == 0
        common_cluster_gtbox_idx.append(common_box_idx)
        common_cluster_gtbox_moco_idx.append(common_box_idx_moco)
        common_unscaled_lwhz.append(unscaled_lwhz[i][common_box_idx, :-1]) #exclude cluster id
        
        assert (batch[i]['gt_boxes'][:,-1][common_box_idx] - batch[i]['gt_boxes_moco'][:,-1][common_box_idx_moco]).sum()== 0
        assert (np.unique(cluster_ids[i])[1:] - gt_boxes_cluster_ids[i][common_box_idx]).sum() == 0
        assert (np.unique(cluster_ids_moco[i])[1:]- gt_boxes_cluster_ids[i][common_box_idx]).sum() ==0
        assert (np.unique(cluster_ids_moco[i])[1:]- gt_boxes_moco_cluster_ids[i][common_box_idx_moco]).sum() ==0
        common_gt_box_class_ids.append(gt_box_class_ids[i][common_box_idx])

    common_unscaled_lwhz = np.concatenate(common_unscaled_lwhz, axis=0)
    common_gt_box_class_ids = np.concatenate(common_gt_box_class_ids, axis=0)

    if shape_descs_required:
        shape_cluster_ids_is_common_mask_batch = []
        for i in range(batch_size):
            shape_cluster_ids_is_common_mask= np.isin(shape_desc_cluster_ids[i], common_cluster_ids[i])
            assert (common_cluster_ids[i]- shape_desc_cluster_ids[i][shape_cluster_ids_is_common_mask]).sum() == 0

            shape_cluster_ids_is_common_mask_batch.append(shape_cluster_ids_is_common_mask)

        shape_cluster_ids_is_common_mask_batch = np.concatenate(shape_cluster_ids_is_common_mask_batch, axis=0).reshape(-1)
        shape_descs = np.concatenate([x["shape_descs"] for x in batch], axis=0)
        shape_descs=shape_descs[shape_cluster_ids_is_common_mask_batch]

    # sparse_points, sparse_points_moco = collate_points_to_sparse_tensor(voxel_coords, points, voxel_coords_moco, points_moco) #xi and xj are sparse tensors for normal and moco pts -> (C:(8, 20k, 4=b_id, xyz voxcoord), F:(8, 20k, 4=xyzi pts))

        # make gt boxes in shape (batch size, max gt box len, 8) (xyz, lwh, rz, class label)
    max_gt = max([len(x['gt_boxes']) for x in batch])
    batch_gt_boxes3d = np.zeros((batch_size, max_gt, batch[0]['gt_boxes'].shape[-1]), dtype=np.float32) # (batch size = 2, max_gt_boxes in a pc in this batch = 67, 8)
    for k in range(batch_size):
        batch_gt_boxes3d[k, :batch[k]['gt_boxes'].__len__(), :] = batch[k]['gt_boxes']

    # make gt boxes in shape (batch size, max gt box len, 8)
    max_gt = max([len(x['gt_boxes_moco']) for x in batch])
    batch_gt_boxes3d_moco = np.zeros((batch_size, max_gt, batch[0]['gt_boxes_moco'].shape[-1]), dtype=np.float32) # (batch size = 2, max_gt_boxes in a pc in this batch = 67, 8)
    for k in range(batch_size):
        batch_gt_boxes3d_moco[k, :batch[k]['gt_boxes_moco'].__len__(), :] = batch[k]['gt_boxes_moco']



    output_batch = {'input': 
                    {'points': points,
                     'voxel_coords': voxel_coords,
                     'cluster_ids': cluster_ids,
                     'gt_boxes': batch_gt_boxes3d,
                     'gt_boxes_cluster_ids': gt_boxes_cluster_ids,
                    'common_unscaled_lwhz': common_unscaled_lwhz,
                    'common_gt_box_class_ids': common_gt_box_class_ids,
                    'common_cluster_gtbox_idx': common_cluster_gtbox_idx,
                     'batch_size': batch_size},
                    
                    'input_moco': 
                    {'points': points_moco,
                     'voxel_coords': voxel_coords_moco,
                     'cluster_ids': cluster_ids_moco,
                     'gt_boxes': batch_gt_boxes3d_moco,
                     'gt_boxes_cluster_ids': gt_boxes_moco_cluster_ids,
                     'common_cluster_gtbox_idx': common_cluster_gtbox_moco_idx,
                     'batch_size': batch_size}
                    
                    }

        
    if shape_descs_required:
        output_batch['input'].update({
                     'shape_descs': shape_descs,
                    #  'shape_cluster_ids_is_common_mask_batch': shape_cluster_ids_is_common_mask_batch,
                    })
        


    return output_batch