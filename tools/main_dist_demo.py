# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#      

import argparse
import os
import random
import time
import warnings
import yaml

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
  
import utils.logger
from utils import main_utils, wandb_utils

parser = argparse.ArgumentParser(description='PyTorch Self Supervised Training in 3D')

parser.add_argument('--cfg', type=str, default=None, help='specify the config for training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--local_rank', default=0, type=int,
                    help='local process id i.e. GPU id to use.')
parser.add_argument('--ngpus', default=4, type=int,
                    help='number of GPUs to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def initialize_distributed_backend(args):
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
    
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank % num_gpus)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method='tcp://127.0.0.1:%d' % args.tcp_port,
            rank=args.local_rank,
            world_size=num_gpus
        )
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()

        # synchronizes all the threads to reach this point before moving on
        dist.barrier()

    if args.rank == -1:
        args.rank = 0
    return args

def distribute_model_to_cuda(model, args, find_unused_params=False):
    torch.cuda.set_device(args.local_rank)
    model = model.cuda(args.local_rank)
    if args.multiprocessing_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=find_unused_params) #device_ids should always be local_rank!

    return model, args 

def main():
    args = parser.parse_args()

    # Read config file
    cfg = yaml.safe_load(open(args.cfg))

    # Fix seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    
    args.ngpus = torch.cuda.device_count()  
    main_worker(args, cfg)


def main_worker(args, cfg):
    
    # Initialize the distributed learning process
    args = initialize_distributed_backend(args)

    # Define model
    model = main_utils.build_model(cfg['model'], cfg['cluster'])
    if args.multiprocessing_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model, args = distribute_model_to_cuda(model, args, find_unused_params=cfg['cluster'])

    # Define dataloaders. Uses DistributedSampler as the dataloader sampler if multiprocessing
    train_loader = main_utils.build_dataloaders(cfg['dataset'], cfg['num_workers'], cfg['cluster'], args.multiprocessing_distributed)       

    # Define criterion    
    train_criterion = main_utils.build_criterion(cfg['loss'], cfg['cluster'], cfg['linear_probe'])
    train_criterion = train_criterion.cuda()
            
    # Define optimizer
    optimizer, scheduler = main_utils.build_optimizer(
        params=list(model.parameters())+list(train_criterion.parameters()),
        cfg=cfg['optimizer'])
    
    # # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']

    cudnn.benchmark = True

    ############################ TRAIN #########################################
    for epoch in range(start_epoch, end_epoch):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_one_epoch('train', train_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, tb_writter, lr=scheduler.get_lr())
        scheduler.step(epoch)

def train_one_epoch(phase, loader, model, optimizer, criterion, epoch, args, cfg, logger, tb_writter, lr):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    batch_time = metrics_utils.AverageMeter(f'{phase}-Avg Batch Process Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter(f'{phase}-Avg Batch Load Time', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter(f'{phase}-Loss', ':.3e')
    progress = utils.logger.ProgressMeter(len(loader), [batch_time, data_time, loss_meter], phase=phase, epoch=epoch, logger=logger, tb_writter=tb_writter)

    # switch to train mode
    model.train(phase == 'train')
    # ['points', 'points_moco']
    end = time.time()
    device = args.local_rank if args.local_rank is not None else 0
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end) # Time to load one batch

        if phase == 'train':
            output_dict = model(sample) #list: query encoder's = embedding[0] size = (8, 128) -> we want (B, num voxel, 128), key encoder = emb[1] = (8,128)
        else:
            with torch.no_grad():
                output_dict = model(sample)
        
        loss = criterion(output_dict)
        loss_meter.update(loss.item(), cfg['dataset']['BATCHSIZE_PER_REPLICA'])


        # compute gradient and do SGD step during training
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end) # This is printed as Time # Time to train one batch
        end = time.time()

        # print to terminal and tensorboard
        if (i+1) % cfg['print_freq'] == 0 or i == 0 or i+1 == len(loader):
            progress.display(i+1)
            

    # Sync metrics across all GPUs and print final averages
    if args.multiprocessing_distributed:
        progress.synchronize_meters(args.local_rank)
        progress.display(len(loader)*args.world_size)

    metrics_dict = {'epoch': epoch, 'lr': lr}
    if tb_writter is not None:
        for meter in progress.meters:
            tb_writter.add_scalar('{}-epoch/{}'.format(phase, meter.name), meter.avg, epoch)
            metrics_dict[meter.name +'-epoch'] = meter.avg
    
    wandb_utils.log(cfg, args, metrics_dict, epoch)

if __name__ == '__main__':
    main()
