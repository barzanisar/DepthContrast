# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import shutil
import torch
import numpy as np
import torch.distributed as dist
import datetime
import collections.abc as container_abcs
from utils.logger import Logger
import subprocess

from datasets import build_dataset, get_loader
import torch.multiprocessing as mp

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) #utils
ROOT_DIR = os.path.dirname(ROOT_DIR) #DepthContrast
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'OpenPCDet', "pcdet"))

from pcdet.utils import box_utils

def confusion_matrix(preds, labels, num_classes):
    hist = (
        torch.bincount(
            num_classes * labels + preds,
            minlength=num_classes ** 2,
        )
        .reshape(num_classes, num_classes)
        .float()
    )
    return hist


def compute_IoU_from_cmatrix(hist, ignore_index=None):
    """Computes the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        m_IoU, fw_IoU, and matrix IoU
    """
    if ignore_index is not None:
        hist[ignore_index] = 0.0
    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
    IoU = intersection.float() / union.float()
    IoU[union == 0] = 1.0
    if ignore_index is not None:
        IoU = torch.cat((IoU[:ignore_index], IoU[ignore_index+1:]))
    m_IoU = torch.mean(IoU).item()
    fw_IoU = (
        torch.sum(intersection) / (2 * torch.sum(hist) - torch.sum(intersection))
    ).item()
    return m_IoU, fw_IoU, IoU


def compute_IoU(preds, labels, num_classes, ignore_index=None):
    """Computes the Intersection over Union (IoU)."""
    hist = confusion_matrix(preds, labels, num_classes)
    return compute_IoU_from_cmatrix(hist, ignore_index)

def initialize_distributed_backend(args, ngpus_per_node):
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        if args.launcher == 'fair':
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            # now they change arg.rank definition from rank of node to global rank of gpu
            args.rank = args.rank * ngpus_per_node + args.local_rank
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        elif args.launcher == 'slurm':
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            proc_id = int(os.environ['SLURM_PROCID']) # global rank
            ntasks = int(os.environ['SLURM_NNODES']) * ngpus_per_node
            node_list = os.environ['SLURM_NODELIST']
            assert ngpus_per_node == torch.cuda.device_count(), torch.cuda.device_count()
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput('scontrol show hostname {} | head -n 1'.format(node_list))
            print(f"node_list: {node_list}")
            print(f"addr: {addr}")
            print(f"ntasks: {ntasks}")
            print(f"proc_id: {proc_id}")
            addr = 'gra' + os.environ['SLURM_NODELIST'][4:8] if int(os.environ['SLURM_NNODES']) > 1 else os.environ['SLURM_NODELIST']
            os.environ['MASTER_PORT'] = str(args.tcp_port) #29500
            os.environ['MASTER_ADDR'] = '127.0.0.1' #addr #'gra' + os.environ['SLURM_NODELIST'][4:8] #'127.0.0.1'#addr
            addr = os.environ['MASTER_ADDR']
            os.environ['WORLD_SIZE'] = str(ntasks)
            os.environ['RANK'] = str(proc_id)
            env_dict = {
                key: os.environ[key]
                for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
            }
            print(env_dict)
            print("LAUNCHING!")
            dist.init_process_group(backend=args.dist_backend, rank=proc_id, world_size=int(os.environ['WORLD_SIZE']),
                                    init_method=f'tcp://{addr}:{args.tcp_port}')
            print("LAUNCHed!")
            args.world_size = dist.get_world_size()
            args.local_rank = str(proc_id % num_gpus) #local rank
            args.rank = dist.get_rank() # global rank

        elif args.launcher == 'pytorch':
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            env_dict = {
                key: os.environ[key]
                for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
            }
            print("LAUNCHING!")
            print(env_dict)
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(args.local_rank % num_gpus)
            dist.init_process_group(
                backend=args.dist_backend,
                init_method='tcp://127.0.0.1:%d' % args.tcp_port,
                rank=args.local_rank,
                world_size=num_gpus
            )
            args.world_size = dist.get_world_size()
            args.rank = dist.get_rank()  # dist.get_rank() returns global rank

    if args.rank == -1:
        args.rank = 0
    return args

### For testing only
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

### Recurisive copy to GPU
def recursive_copy_to_gpu(value, non_blocking=True, max_depth=3, curr_depth=0):
    """
    Recursively searches lists, tuples, dicts and copies to GPU if possible.
    Note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the GPU.
    """
    if curr_depth >= max_depth:
        raise ValueError("Depth of value object is too deep")

    try:
        try:
            return value.cuda(non_blocking=non_blocking)
        except:
            return value.to(torch.device('cuda'))
    except AttributeError:
        if isinstance(value, container_abcs.Sequence):
            gpu_val = []
            for val in value:
                gpu_val.append(
                    recursive_copy_to_gpu(
                        val,
                        non_blocking=non_blocking,
                        max_depth=max_depth,
                        curr_depth=curr_depth + 1,
                    )
                )

            return gpu_val if isinstance(value, list) else tuple(gpu_val)
        elif isinstance(value, container_abcs.Mapping):
            gpu_val = {}
            for key, val in value.items():
                gpu_val[key] = recursive_copy_to_gpu(
                    val,
                    non_blocking=non_blocking,
                    max_depth=max_depth,
                    curr_depth=curr_depth + 1,
                )

            return gpu_val

        raise AttributeError("Value must have .cuda attr or be a Seq / Map iterable")

def prep_environment(args, cfg):
    from torch.utils.tensorboard import SummaryWriter
    
    phase = 'linear_probe' if cfg.get('linear_probe', False) else 'train'
    # Prepare loggers (must be configured after initialize_distributed_backend())
    model_dir = '{}/{}/{}'.format(cfg['model']['model_dir'], cfg['model']['name'], phase)
    
    
    if args.rank == 0:
        if cfg['linear_probe']:
            assert os.path.isdir(model_dir.split('/linear_probe')[0])
        prep_output_folder(model_dir)

    log_fn = '{}/{}_{}.log'.format(model_dir, phase, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = Logger(quiet=args.quiet, log_fn=log_fn, rank=args.rank)

    logger.add_line(str(datetime.datetime.now()))
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))

    logger.add_line("=" * 30 + "   Config   " + "=" * 30)
    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  '+ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))
    print_dict(cfg)

    logger.add_line("=" * 30 + "   Args   " + "=" * 30)
    for k in args.__dict__:
        logger.add_line('{:30} {}'.format(k, args.__dict__[k]))

    tb_writter = None
    if cfg['log2tb'] and args.rank == 0:
        tb_dir = '{}/tensorboard'.format(model_dir)
        os.system('mkdir -p {}'.format(tb_dir))
        tb_writter = SummaryWriter(tb_dir)

    return logger, tb_writter, model_dir


def build_model(cfg, logger=None, linear_probe=False):
    import models
    return models.build_model(cfg, logger, linear_probe)


def distribute_model_to_cuda(models, args):

    squeeze = False
    if not isinstance(models, list):
        models = [models]
        squeeze = True

    for i in range(len(models)):
        if args.multiprocessing_distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.local_rank is not None:
                torch.cuda.set_device(args.local_rank)
                models[i].cuda(args.local_rank)
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.local_rank % torch.cuda.device_count()]) #device_ids should always be local_rank!
            else:
                models[i].cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i])
        elif args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            models[i] = models[i].cuda(args.local_rank)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            # Careful!!! DataParallel does not work for vox
            models[i] = torch.nn.DataParallel(models[i]).cuda()

    if squeeze:
        models = models[0]

    return models, args


def build_dataloaders_train_val(cfg, num_workers, distributed, logger, linear_probe=False):
    train_loader = build_dataloader(cfg, num_workers, distributed, linear_probe=linear_probe, mode = 'train', logger=logger)
    logger.add_line("\n"+"="*30+"   Train data   "+"="*30)
    logger.add_line(str(train_loader.dataset))

    val_loader = build_dataloader(cfg, num_workers, distributed, linear_probe=linear_probe, mode = 'val', logger=logger)
    logger.add_line("\n"+"="*30+"   Val data   "+"="*30)
    logger.add_line(str(val_loader.dataset))
    return train_loader, val_loader

def build_dataloaders(cfg, num_workers, distributed, logger, linear_probe=False):
    train_loader = build_dataloader(cfg, num_workers, distributed, linear_probe=linear_probe, mode = 'train', logger=logger)
    logger.add_line("\n"+"="*30+"   Train data   "+"="*30)
    logger.add_line(str(train_loader.dataset))
    return train_loader


def build_dataloader(config, num_workers, distributed, linear_probe=False, mode='train', logger=None):
    import torch.utils.data as data
    import torch.utils.data.distributed
    import datasets

    datasets, data_and_label_keys = {}, {}
    datasets = build_dataset(config, linear_probe, mode, logger)

    loader = get_loader(
        dataset=datasets,
        dataset_config=config,
        num_dataloader_workers=num_workers,
        pin_memory=False,### Questionable
    )
    return loader

def build_criterion(cfg, logger=None):
    import criterions
    criterion = criterions.__dict__[cfg['name']](cfg['args'])
    if logger is not None:
        logger.add_line(str(criterion))

    return criterion


def build_optimizer(params, cfg, logger=None):
    if cfg['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            params=params,
            lr=cfg['lr']['base_lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov=cfg['nesterov']
        )

    elif cfg['name'] == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg['lr']['base_lr'],
            weight_decay=cfg['weight_decay'],
            betas=cfg['betas'] if 'betas' in cfg else [0.9, 0.999]
        )

    else:
        raise ValueError('Unknown optimizer.')

    if cfg['lr']['name'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr']['milestones'], gamma=cfg['lr']['gamma'])
    else:
        ### By default we use a cosine param scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'], eta_min=cfg['lr']['final_lr'])
    return optimizer, scheduler

def get_labels(batch_gt_boxes_lidar, coords, label_type = 'class_names'):
    '''
    coords: numpy array (batch size, n points, 3)

    batch_labels: list of len "batch_size". Each element is a numpy array of len npoints
    '''
    batch_size = len(batch_gt_boxes_lidar)
    batch_labels = []
    for pc_idx, gt_boxes_lidar in enumerate(batch_gt_boxes_lidar):
        points = coords[pc_idx]
        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar[:,:7]) #nboxes, 8 corners, 3coords
        num_objects = gt_boxes_lidar.shape[0]
        labels = np.zeros(points.shape[0])

        for k in range(num_objects):
            flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k]) #flag=points dim labels=np.zeros(points.shape[0]), labels[flag] = class_names[obj_name].get_index
            labels[flag] = gt_boxes_lidar[k, 7] if label_type=='class_names' else 1
        batch_labels.append(labels) # background = 0, car = 1, so on OR background = 0, object = 1
    batch_labels = torch.stack([torch.tensor(batch_labels[i]).long() for i in range(batch_size)])
    return batch_labels



class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0, dist=False):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.best_metric = 0.
        self.dist = dist #distributed

    def save(self, epoch, filename=None, eval_metric=0., **kwargs):
        if self.rank != 0:
            return

        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        state = {'epoch': epoch}
        for k in kwargs:
            state[k] = kwargs[k].state_dict()

        if filename is None:
            save_checkpoint(state=state, is_best=is_best, model_dir=self.checkpoint_dir)
        else:
            save_checkpoint(state=state, is_best=False, filename='{}/{}'.format(self.checkpoint_dir, filename))

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, fn=None, restore_last=False, restore_best=False, **kwargs):
        checkpoint_fn = f'{self.checkpoint_dir}/{fn}' if fn is not None else self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        for k in kwargs:
            if (k == 'model'):
                newparam = {}
                for tempk in ckp[k]:
                    if tempk[:7] == 'module.':
                        newparam[tempk[7:]] = ckp[k][tempk]
                    else:
                        newparam[tempk] = ckp[k][tempk]
                ### Fix the module issue
                kwargs[k].load_state_dict(newparam)
            else:
                kwargs[k].load_state_dict(ckp[k])
        return start_epoch


def save_checkpoint(state, is_best, model_dir='.', filename=None):
    if filename is None:
        filename = '{}/checkpoint.pth.tar'.format(model_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(model_dir))


def prep_output_folder(model_dir): 
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)



