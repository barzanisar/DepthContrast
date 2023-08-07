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
import torch.distributed as dist

from torch.nn.utils import clip_grad_norm_

import torch.backends.cudnn as cudnn
import torch.optim
from third_party.OpenPCDet.pcdet.config import cfg, cfg_from_yaml_file
  
import utils.logger
from utils import main_utils, wandb_utils


parser = argparse.ArgumentParser(description='PyTorch Self Supervised Training in 3D')

parser.add_argument('--cfg', type=str, default=None, help='specify the config for training')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training') #remove, ws is total no. of gpus, NOT nodes!
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training') #remove
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'fair'], default='none')
# parser.add_argument('--tcp_port', type=int, default=18878, help='tcp port for distrbuted training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                    help='url used to set up distributed training') #tc port
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--local_rank', default=0, type=int,
                    help='local process id i.e. GPU id to use.') #local_rank = 0
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def main():
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg, cfg)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.multiprocessing_distributed:
        assert args.launcher in ['pytorch', 'slurm']
    # Simply call main_worker function
    main_worker(args, cfg)

def main_worker(args, cfg):
    # Run on every GPU with args.local_rank
    # Setup environment
    args = main_utils.initialize_distributed_backend(args) ### Use other method instead
    logger, tb_writter, model_dir = main_utils.prep_environment(args, cfg)
    wandb_utils.init(cfg, args, job_type='train')
    # print("=" * 30 + "   DDP   " + "=" * 30)
    # print(f"world_size: {args.world_size}")
    # print(f"local_rank: {args.local_rank}")
    # print(f"rank: {args.rank}")

    
    CLUSTER = False
    if 'PRETEXT_HEAD'in cfg['model']['MODEL_BASE']:
        CLUSTER = cfg['model']['MODEL_BASE']['PRETEXT_HEAD']['NAME'] == 'SegHead'
    
    # Define dataloaders
    train_loader = main_utils.build_dataloaders(cfg['dataset'], cfg['num_workers'], CLUSTER, args.multiprocessing_distributed, logger)  

    # Define model
    model = main_utils.build_model(cfg['model'], CLUSTER, train_loader.dataset, logger)
    if args.multiprocessing_distributed:
        logger.add_line('='*30 + 'Sync Batch Normalization' + '='*30)
        
        # Only sync bn for detection head layers and not backbone 3d for shuffle bn to work
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        # TODO: This is a dirty way of doing it
        if 'MODEL_HEAD' in cfg['model']:
            model.head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.head)

    model, args = main_utils.distribute_model_to_cuda(model, args) #, find_unused_params=CLUSTER


    # Define criterion    
    train_criterion = main_utils.build_criterion(cfg['loss'], CLUSTER, cfg['linear_probe'],logger=logger)
    train_criterion = train_criterion.cuda()
            
    # Define optimizer
    optimizer, scheduler = main_utils.build_optimizer(
        params=list(model.parameters())+list(train_criterion.parameters()),
        cfg=cfg['optimizer'],
        logger=logger)
    ckp_manager = main_utils.CheckpointManager(model_dir, logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)
    
    # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume']:
        if ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(restore_last=True, model=model, optimizer=optimizer, train_criterion=train_criterion)
            scheduler.step(start_epoch)
            logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))
        else:
            logger.add_line("No checkpoint found at '{}'".format(ckp_manager.last_checkpoint_fn()))

    cudnn.benchmark = True

    ############################ TRAIN #########################################
    test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
    for epoch in range(start_epoch, end_epoch):
        if (epoch >= cfg['save_ckpt_after_epochs'] and epoch % cfg['ckpt_save_interval']== 0):
            ckp_manager.save(epoch, model=model, filename='checkpoint-ep{}.pth.tar'.format(epoch))
            logger.add_line(f'Saved checkpoint checkpoint-ep{epoch}.pth.tar before beginning epoch {epoch}')

        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
        logger.add_line('LR: {}'.format(scheduler.get_last_lr()))
        run_phase('train', train_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, tb_writter, lr=scheduler.get_last_lr())
        scheduler.step(epoch)

        if ((epoch % test_freq) == 0) or (epoch == end_epoch - 1):
            ckp_manager.save(epoch+1, model=model, optimizer=optimizer, train_criterion=train_criterion)
            logger.add_line(f'Saved checkpoint for testing {ckp_manager.last_checkpoint_fn()} after ending epoch {epoch}, {epoch+1} is recorded for this chkp')

    if args.multiprocessing_distributed:
        dist.destroy_process_group()


def run_phase(phase, loader, model, optimizer, criterion, epoch, args, cfg, logger, tb_writter, lr):
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

        
        loss = criterion(output_dict['output_base'], output_dict['output_moco']) # contrastive loss
        if 'MODEL_HEAD' in cfg['model']:
            loss += output_dict['loss_head']  # detection loss
        loss_meter.update(loss.item(), cfg['dataset']['BATCHSIZE_PER_REPLICA'])


        # compute gradient and do SGD step during training
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is not None:
            #             print(name, f' {param.grad.shape}')
            #         else:
            #             print(name, f' is None')

        is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
        assert not is_nan

        # measure elapsed time
        batch_time.update(time.time() - end) # This is printed as Time # Time to train one batch
        end = time.time()

        # print to terminal and tensorboard
        step = epoch * len(loader) + i #sample is a batch of 8 transformed point clouds, len(loader) is the total number of batches
        if (i+1) % cfg['print_freq'] == 0 or i == 0 or i+1 == len(loader):
            progress.display(i+1)
            
            # # Log to wb
            # metrics_dict = {'epoch': epoch, 'step': step}
            # for meter in progress.meters:
            #     metrics_dict[meter.name + '-batch'] = meter.avg
            # wandb_utils.log(cfg, args, metrics_dict, step)

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
