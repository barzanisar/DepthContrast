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
import glob
from pathlib import Path
from functools import partial


from datasets import build_dataset, get_loader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) #utils
ROOT_DIR = os.path.dirname(ROOT_DIR) #DepthContrast
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'OpenPCDet', "pcdet"))


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

def initialize_distributed_backend(args):
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        if args.launcher == 'slurm':
            proc_id = int(os.environ['SLURM_PROCID']) # global rank of GPUS: 0,1,2,3,4,5,6,7,8
            rank = proc_id
            # world_size = int(os.environ['SLURM_NTASKS']) # set #SBATCH --ntasks-per-node=M where M is the same value set in #SBATCH --gres=gpu:M
            local_rank = int(os.environ['SLURM_LOCALID'])
            #ntasks = int(os.environ['SLURM_NNODES']) * ngpus_per_node # total num gpus i.e. world size
            #assert ngpus_per_node == torch.cuda.device_count(), torch.cuda.device_count()
            ngpus_per_node = torch.cuda.device_count()
            torch.cuda.set_device(proc_id % ngpus_per_node) # set local rank: if procid id 5 then set_device takes local gpu rank i.e. 1
           
            print(f"LAUNCHING! rank:{rank}, ws:{args.world_size}, local_rank:{local_rank}, dist_url: {args.dist_url}")

            dist.init_process_group(backend=args.dist_backend, rank=rank, world_size=args.world_size,
                                    init_method=args.dist_url)
            args.world_size = dist.get_world_size() # total num gpus across all nodes
            args.local_rank = str(proc_id % ngpus_per_node) #local rank
            args.rank = dist.get_rank() # global rank
            print(f"LAUNCHED! rank:{rank} {dist.get_rank()}, ws:{dist.get_world_size()}, local_rank:{local_rank}")

        elif args.launcher == 'pytorch':
            env_dict = {
                key: os.environ[key]
                for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK")
            }
            print("LAUNCHING!")
            print(env_dict)

            ngpus_per_node = torch.cuda.device_count()

            # """ This next line is the key to getting DistributedDataParallel working on SLURM:
            #     SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
            #     current process inside a node and is also 0 or 1 in this example."""

            local_rank = args.local_rank
            rank = int(os.environ.get("SLURM_NODEID", 0))*ngpus_per_node + local_rank

            current_device = local_rank
            torch.cuda.set_device(current_device)
            """ this block initializes a process group and initiate communications
		        between all processes running on all nodes """

            print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
            dist.init_process_group(backend='nccl', init_method="env://")
            print('From Rank: {}, ==> Process Group Ready!...'.format(rank))
            args.rank = rank
            print(env_dict, f"args.rank: {args.rank}", f"args.world_size: {args.world_size}", f"dist.get_world_size(): {dist.get_world_size()}")

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

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key in ['cluster_ids', 'shape_cluster_ids_is_common_mask_batch', 'sparse_points']:
            continue

        if isinstance(val, np.ndarray):
            try:
                batch_dict[key] = torch.from_numpy(val).float().cuda(non_blocking=True)
            except:
                print(f"ERROR: {key}, {val.shape}, {val.dtype}")
                raise ValueError
        elif torch.is_tensor(val):
            val.cuda(non_blocking=True)

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

def get_ckpts_to_eval(cfg, logger, pretrain_model_dir, eval_list_dir):
    checkpoints_to_eval = []
    assert os.path.isdir(pretrain_model_dir) 
    assert os.path.isdir(eval_list_dir), f'Eval list dir: {eval_list_dir} does not exist!'
    
    # evaluated ckpt record
    ckpt_record_file = Path(eval_list_dir) / 'eval_list.txt'
    with open(ckpt_record_file, 'a'):
        pass
    if cfg['load_pretrained_checkpoint'] == 'all':
        ckpt_list = glob.glob(os.path.join(pretrain_model_dir, 'checkpoint*.pth.tar'))
        ckpt_list.sort(key=os.path.getmtime)
        evaluated_ckpt_list = [x.strip() for x in open(ckpt_record_file, 'r').readlines()]
        if len(evaluated_ckpt_list) == 0:
            checkpoints_to_eval = [x.split('/')[-1] for x in ckpt_list]
        else:
            for cur_ckpt in ckpt_list:
                cur_ckpt_name = cur_ckpt.split('/')[-1]
                if cur_ckpt_name not in evaluated_ckpt_list:
                    checkpoints_to_eval.append(cur_ckpt_name)
    else:
        checkpoints_to_eval.append(cfg['load_pretrained_checkpoint'])
    
    return checkpoints_to_eval, ckpt_record_file

def prep_environment(args, cfg, pretraining=True):
    from torch.utils.tensorboard import SummaryWriter
    DATASET_NAMES = { 'WaymoDataset': 'waymo', 'SemanticKittiDataset': 'semantickitti', 'NuscenesDataset': 'nuscenes'}
    dataset_name = DATASET_NAMES[cfg['dataset']['DATASET_NAMES'][0]]
    if pretraining:
        phase_name = f'pretrain_waymo'
    else:
        downstream_task = 'segmentation' #if 'SEGMENTATION_HEAD' in cfg['model'] else 'detection'
        if 'downstream_model_dir' in cfg['model']:
            phase_name = cfg['model']['downstream_model_dir']
        else:
            if cfg['model']['linear_probe']:
                phase_name = f'linearprobe_{downstream_task}_{dataset_name}'
            else:
                phase_name = f'finetune_{downstream_task}_{dataset_name}'

    # Prepare loggers (must be configured after initialize_distributed_backend())
    model_dir = '{}/{}/{}'.format(cfg['model']['model_dir'], cfg['model']['name'], phase_name)
    pretrain_model_dir = '{}/{}/pretrain_waymo'.format(cfg['model']['model_dir'], cfg['model']['name'])
    
    
    if args.rank == 0:
        if not pretraining:
            assert os.path.isdir(model_dir.split(f'/{phase_name}')[0])
        prep_output_folder(model_dir)

    log_fn = '{}/{}.log'.format(model_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
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

    return logger, tb_writter, model_dir, pretrain_model_dir, phase_name


def build_model(cfg, pretraining, dataset, logger=None):
    import models
    return models.build_model(cfg, pretraining, dataset, logger)


def distribute_model_to_cuda(models, args, find_unused_params=False):

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
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=find_unused_params) #device_ids should always be local_rank!
            else:
                models[i].cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], find_unused_parameters=find_unused_params)
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


def build_dataloader(config, num_workers, pretraining=True, mode='train', logger=None):
    datasets = build_dataset(config, pretraining, mode, logger)
    logger.add_line("\n"+"="*30+f"   {mode} data   "+"="*30)
    # logger.add_line(str(datasets.dataset))
    
    loader = get_loader(
        dataset=datasets,
        dataset_config=config,
        num_dataloader_workers=num_workers,
        pin_memory=True ### Questionable
    )
    return loader

def build_criterion(cfg, logger=None):
    import criterions
    criterion = criterions.__dict__['NCELossMoco'](cfg)
    
    if logger is not None:
        logger.add_line(str(criterion))

    return criterion


def build_optimizer(params, cfg, total_iters_each_epoch=None, logger=None):
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
    elif cfg['lr']['name'] == 'cosine':
        ### By default we use a cosine param scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters_each_epoch*cfg['num_epochs'], eta_min=cfg['lr']['base_lr']/1000)
    else:
        def linear_warmup_with_cosdecay(cur_step, total_steps, warmup_steps=0, min_scale=1e-5):
            if cur_step < warmup_steps:
                return (1 - min_scale) * cur_step / warmup_steps + min_scale
            else:
                ratio = (cur_step - warmup_steps) / total_steps
                return (1 - min_scale) * 0.5 * (1 + np.cos(np.pi * ratio)) + min_scale

    
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                linear_warmup_with_cosdecay,
                total_steps=total_iters_each_epoch*cfg['num_epochs'],
                warmup_steps=total_iters_each_epoch*cfg['warmup_epochs']
            )
        )
        
    return optimizer, scheduler


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, logger, rank=0, dist=False):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.best_metric = 0.
        self.dist = dist #distributed
        self.logger = logger

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
        
    def restore(self, fn=None, restore_last=False, restore_best=False, skip_model_layers=None, **kwargs):
        checkpoint_fn = f'{self.checkpoint_dir}/{fn}' if fn is not None else self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        self.logger.add_line(f"Loading chkpoint from: {checkpoint_fn}")
        start_epoch = ckp['epoch']
        for k in kwargs:
            if (k == 'model'):
                ckpt_state_dict = ckp[k]
                new_model_state_dict = kwargs[k].state_dict()
                model_init_layers = {param_name: False for param_name in new_model_state_dict}
                # new_state_dict = {} # make ckpt state_stict same as model_state_dict
                not_found, not_init = [], []
                for param_name in model_init_layers:
                    if skip_model_layers is not None and len(skip_model_layers) > 0:
                        for sl in skip_model_layers:
                            if param_name.find(sl) >=0:
                                # self.logger.add_line(f"Ignored layer:\t{param_name}")
                                not_init.append(param_name)
                                break
                        if len(not_init) and not_init[-1] == param_name:        
                            continue

                    if "module."+param_name in ckpt_state_dict:
                        new_model_state_dict[param_name].copy_(ckpt_state_dict["module."+param_name])
                        model_init_layers[param_name] = True
                        # self.logger.add_line(f"Init layer:\t{param_name}")

                    elif param_name in ckpt_state_dict:
                        new_model_state_dict[param_name].copy_(ckpt_state_dict[param_name]) 
                        model_init_layers[param_name] = True
                        # self.logger.add_line(f"Init layer:\t{param_name}")
                    else:
                        not_found.append(param_name)
                        # self.logger.add_line(f"Not found:\t{param_name}")
                
                kwargs[k].load_state_dict(new_model_state_dict)

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

def undersample_bk_class(output, y_gt):
        background_points_mask = y_gt == 0
        obj_points_mask = y_gt > 0


        bk_y_gt = y_gt[background_points_mask]
        bk_output = output[background_points_mask]

        perm = torch.randperm(bk_y_gt.size(0))
        idx = perm[:obj_points_mask.sum().item()]
        downsampled_bk_y_gt = bk_y_gt[idx]
        downsampled_bk_output = bk_output[idx]

        new_y_gt = torch.cat((downsampled_bk_y_gt, y_gt[obj_points_mask]))
        new_output = torch.cat((downsampled_bk_output, output[obj_points_mask]))

        return new_output, new_y_gt


