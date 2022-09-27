#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np

def rotx(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def pc2obj(pc, filepath='test.obj'):
    pc = pc.T
    nverts = pc.shape[1]
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in range(nverts):
            f.write("v %.4f %.4f %.4f\n" % (pc[0,v],pc[1,v],pc[2,v]))

def write_ply_color(points, colors, out_filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    ### Write header here
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex %d\n" % N)
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    fout.write("end_header\n")
    for i in range(N):
        #c = pyplot.cm.hsv(labels[i])
        c = colors[i,:]
        c = [int(x*255) for x in c]
        fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()
            
def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()
            
def check_aspect(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2])/np.max(crop_range[:2])
    xz_aspect = np.min(crop_range[[0,2]])/np.max(crop_range[[0,2]])
    yz_aspect = np.min(crop_range[1:])/np.max(crop_range[1:])
    return (xy_aspect >= aspect_min) or (xz_aspect >= aspect_min) or (yz_aspect >= aspect_min)

def check_aspect2D(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2])/np.max(crop_range[:2])
    return (xy_aspect >= aspect_min)

def get_transform3d(data, input_transforms_list, vox=False):
    output_transforms = []
    ptdata = data['data']
    outdata = []
    counter = 0
    centers = []
    aug_trans_matrix_list=[]
    #np.random.seed(1024)


    DEBUG_REVERSE_TRANS = False
    # Compute augmentation transformation matrix
    aug_trans_matrix = np.eye(3, 3, dtype=np.float32)
    for point_cloud in ptdata:
        #Randomly select 50000 points
        if len(point_cloud) > 50000:
            newidx = np.random.choice(len(point_cloud), 50000, replace=False)
            point_cloud = point_cloud[newidx,:]
        
        if DEBUG_REVERSE_TRANS:
            #debug reverse transformation
            old_points = np.copy(point_cloud[:,0:3])
        for transform_config in input_transforms_list:
            if transform_config['name'] == 'subcenter':
                continue
                xyz_center = np.expand_dims(np.mean(point_cloud[:,:3], axis=0), 0)
                point_cloud[:,:3] = point_cloud[:,:3] - xyz_center
            if transform_config['name'] == 'RandomFlipLidar':
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    #compute aug matrix
                    flip_trans_x = np.eye(3, 3, dtype=np.float32)
                    flip_trans_x[1, 1] = -1
                    aug_trans_matrix = aug_trans_matrix @ flip_trans_x
            if transform_config['name'] == 'RandomRotateLidar':
                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/2) - np.pi/4 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                #compute aug matrix
                aug_trans_matrix = aug_trans_matrix @ np.transpose(rot_mat)

            if transform_config['name'] == 'RandomScaleLidar':
                noise_scale = np.random.uniform(0.95, 1.05)
                point_cloud[:,0:3] = point_cloud[:,0:3] * noise_scale
                #compute aug matrix
                aug_trans_matrix = aug_trans_matrix * noise_scale

                
            if transform_config['name'] == 'randomcuboidLidar':
                # not compatible with voxel wise contrastive loss!
                range_xyz = np.max(point_cloud[:,0:2], axis=0) - np.min(point_cloud[:,0:2], axis=0)
                if ('randcrop' in transform_config):
                    crop_range = float(transform_config['crop']) + np.random.rand(2) * (float(transform_config['randcrop']) - float(transform_config['crop']))
                    if ('aspect' in transform_config):
                        loop_count = 0
                        while not check_aspect2D(crop_range, float(transform_config['aspect'])):
                            loop_count += 1
                            crop_range = float(transform_config['crop']) + np.random.rand(2) * (float(transform_config['randcrop']) - float(transform_config['crop']))
                            if loop_count > 100:
                                break
                else:
                    crop_range = float(transform_config['crop'])

                loop_count = 0
                while True:
                    loop_count += 1
                    
                    sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]

                    new_range = range_xyz * crop_range / 2.0

                    max_xyz = sample_center[0:2] + new_range
                    min_xyz = sample_center[0:2] - new_range

                    upper_idx = np.sum((point_cloud[:,0:2] <= max_xyz).astype(np.int32), 1) == 2
                    lower_idx = np.sum((point_cloud[:,0:2] >= min_xyz).astype(np.int32), 1) == 2

                    new_pointidx = (upper_idx) & (lower_idx)
                
                    if (loop_count > 100) or (np.sum(new_pointidx) > float(transform_config['npoints'])):
                        break
                
                point_cloud = point_cloud[new_pointidx,:]


            if transform_config['name'] == 'randomdrop':
                # not compatible with voxel wise contrastive loss!
                range_xyz = np.max(point_cloud[:,0:3], axis=0) - np.min(point_cloud[:,0:3], axis=0)

                crop_range = float(transform_config['crop'])
                new_range = range_xyz * crop_range / 2.0

                if "dist_sample" in transform_config:
                    numb,numv = np.histogram(point_cloud[:,2])
                    max_idx = np.argmax(numb)
                    minidx = max(0,max_idx-2)
                    maxidx = min(len(numv)-1,max_idx+2)
                    range_v = [numv[minidx], numv[maxidx]]
                loop_count = 0
                #write_ply_color(point_cloud[:,:3], point_cloud[:,3:], "before.ply")
                while True:
                    sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]
                    loop_count += 1
                    if "dist_sample" in transform_config:
                        if (loop_count <= 100):
                            if (sample_center[-1] > range_v[1]) or (sample_center[-1] < range_v[0]):
                                continue
                    break
                max_xyz = sample_center + new_range
                min_xyz = sample_center - new_range

                upper_idx = np.sum((point_cloud[:,0:3] < max_xyz).astype(np.int32), 1) == 3
                lower_idx = np.sum((point_cloud[:,0:3] > min_xyz).astype(np.int32), 1) == 3

                new_pointidx = ~((upper_idx) & (lower_idx))
                point_cloud = point_cloud[new_pointidx,:]


            if transform_config['name'] == 'ToTensorLidar':
                lpt = len(point_cloud)
                if (vox == False):
                    num_points = 16384
                else:
                    num_points = lpt

                points = point_cloud
                if num_points < len(points):
                    pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
                    pts_near_flag = pts_depth < 40.0
                    far_idxs_choice = np.where(pts_near_flag == 0)[0]
                    near_idxs = np.where(pts_near_flag == 1)[0]
                    near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                    choice = []
                    if num_points > len(far_idxs_choice):
                        near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) if len(far_idxs_choice) > 0 else near_idxs_choice
                    else: 
                        choice = np.arange(0, len(points), dtype=np.int32)
                        choice = np.random.choice(choice, num_points, replace=False)
                    np.random.shuffle(choice)
                else:
                    choice = np.arange(0, len(points), dtype=np.int32)
                    if num_points > len(points):
                        if (num_points - len(points)) <= len(choice):
                            extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                        else:
                            extra_choice = np.random.choice(choice, num_points - len(points), replace=True)
                        choice = np.concatenate((choice, extra_choice), axis=0)
                    np.random.shuffle(choice)                    

                point_cloud = point_cloud[choice,:]

                if DEBUG_REVERSE_TRANS:
                    #debug reverse transformation
                    old_points = old_points[choice,:]
                    new_points = np.array(point_cloud[:,:3])
                    inverse_trans_new_points = new_points @ np.linalg.inv(aug_trans_matrix)
                    max_diff = np.abs(old_points - inverse_trans_new_points).max()
                    #print('max diff is: {}'.format(max_diff))
                    # print(f'Old max: {np.max(old_points, axis=0)}, min: {np.min(old_points, axis=0)}')
                    # print(f'New max: {np.max(new_points, axis=0)}, min: {np.min(new_points, axis=0)}')
                    # print(f'Inverse New max: {np.max(inverse_trans_new_points, axis=0)}, min: {np.min(inverse_trans_new_points, axis=0)}')

                    assert max_diff < 0.001
              
                aug_trans_matrix = torch.tensor(aug_trans_matrix).float()

                if (vox == False):
                    point_cloud = torch.tensor(point_cloud).float()
                    # if DEBUG_REVERSE_TRANS:
                    #     old_points = torch.tensor(old_points).float()

            
        outdata.append(point_cloud)
        # if DEBUG_REVERSE_TRANS:
        #     outdata.append(old_points)
        aug_trans_matrix_list.append(aug_trans_matrix)
    data['data'] = outdata
    data['aug_trans_matrix'] = aug_trans_matrix
    return data
