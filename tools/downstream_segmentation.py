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
from pathlib import Path

import numpy as np
from utils.ioueval import iouEval, EvalMetrics
from utils.data_map import WAYMO_LABELS, NUSCENES_LABELS, SEMANTIC_KITTI_LABELS
from models.base_ssl3d_model import parameter_description
import MinkowskiEngine as ME

parser = argparse.ArgumentParser(description='PyTorch Self Supervised Training in 3D')

parser.add_argument('--cfg', type=str, default=None, help='specify the config for training')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--world-size', default=-1, type=int,
                    help='total number of gpus for distributed training') #remove, ws is total no. of gpus, NOT nodes!
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training') #remove
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'fair'], default='none')
# parser.add_argument('--tcp_port', type=int, default=18878, help='tcp port for distrbuted training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                    help='url used to set up distributed training') #tc port
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--local_rank', default=0, type=int,
                    help='local process id i.e. GPU id to use.') #local_rank = 0
parser.add_argument('--batchsize_per_gpu', default=-1, type=int,
                    help='batchsize_per_gpu')
parser.add_argument('--epochs', default=-1, type=int,
                    help='num epochs')
parser.add_argument('--frame_sampling_div', default=-1, type=int,
                    help='frame_sampling_interval')
parser.add_argument('--data_skip_ratio', default=-1, type=int,
                    help='train data scene skip ratio for nuscenes')
parser.add_argument('--val_interval', default=-1, type=int,
                    help='val interval nuscenes')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--pretrained_ckpt', type=str, default='default', help='load single pretrained ckpt for linear probing or finetuning')
parser.add_argument('--linear_probe_last_n_ckpts', type=int, default=-1, help='last num ckpts to linear probe')
parser.add_argument('--model_name', type=str, default='default', help='pretrained model name')
parser.add_argument('--job_type', type=str, default='default', help='model job_type')
parser.add_argument('--pretrain_extra_tag', type=str, default='default', help='model pretrain_extra_tag')
parser.add_argument('--extra_tag', type=str, default='default', help='model extra_tag')
parser.add_argument('--workers', default=-1, type=int, help='workers per gpu')
parser.add_argument('--val_after_epochs', default=0, type=int, help='workers per gpu')
parser.add_argument('--wandb_dont_resume', action='store_true', default=False, help='for compute canada offline wandb, dont resume')


def main():
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg, cfg)
    if args.batchsize_per_gpu > 0:
        cfg['dataset']['BATCHSIZE_PER_REPLICA']=args.batchsize_per_gpu
    if args.frame_sampling_div > 0:
        if 'FRAME_SAMPLING_INTERVAL' in cfg['dataset']:
            cfg['dataset']['FRAME_SAMPLING_INTERVAL']['train'] /= args.frame_sampling_div
            cfg['dataset']['FRAME_SAMPLING_INTERVAL']['train'] = int(cfg['dataset']['FRAME_SAMPLING_INTERVAL']['train'])
    if args.data_skip_ratio > 0:
        if 'DATA_SKIP_RATIO' in cfg['dataset']:
            cfg['dataset']['DATA_SKIP_RATIO']['train'] = args.data_skip_ratio
    if args.val_interval > 0:
        cfg['val_interval'] = args.val_interval
    
    cfg['val_after_epochs'] = args.val_after_epochs

    if args.epochs > 0:
        cfg['optimizer']['num_epochs']=args.epochs
    if args.model_name != 'default':
        cfg['model']['name'] = args.model_name
    if args.job_type != 'default':
        cfg['model']['job_type'] = args.job_type
    if args.pretrain_extra_tag != 'default':
        cfg['model']['pretrain_extra_tag'] = args.pretrain_extra_tag
    if args.extra_tag != 'default':
        cfg['model']['extra_tag'] = args.extra_tag
    if args.workers > 0:
        cfg['num_workers'] = args.workers

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
    args = main_utils.initialize_distributed_backend(args)
    cfg['model']['INPUT'] = cfg['dataset']['INPUT']
    linear_probe = cfg['model']['linear_probe']
    logger, _, downstream_dir, pretrain_model_dir = main_utils.prep_environment(args, cfg, pretraining=False)
    # pretrain_model_dir = f'{pretrain_model_dir}/ckpt'

    if args.multiprocessing_distributed:
        torch.distributed.barrier()
    if args.pretrained_ckpt != 'default':
        cfg['load_pretrained_checkpoint'] = args.pretrained_ckpt
    if not linear_probe:
        assert cfg['load_pretrained_checkpoint'] != 'all'
    # checkpoints_to_eval, ckpt_record_file = main_utils.get_ckpts_to_eval(cfg, logger, 
    #                                                                      pretrain_model_dir=pretrain_model_dir, 
    #                                                                      eval_list_dir=downstream_dir)
    checkpoints_to_eval = [cfg['load_pretrained_checkpoint']]
    logger.add_line(f'Before ckpts to eval: {checkpoints_to_eval}')
    if linear_probe:
        checkpoints_to_eval = [x for x in checkpoints_to_eval if x != 'checkpoint.pth.tar']
        checkpoints_to_eval = sorted(checkpoints_to_eval, key=lambda x: int(x.split('-ep')[1].split('.')[0]))

        if args.linear_probe_last_n_ckpts > 0:
            last_ckpt_num = int(checkpoints_to_eval[-1].split('-ep')[1].split('.')[0]) 
            eval_ckpts_after_ckpt_n = last_ckpt_num - args.linear_probe_last_n_ckpts
            checkpoints_to_eval = [x for x in checkpoints_to_eval if int(x.split('-ep')[1].split('.')[0]) > eval_ckpts_after_ckpt_n]

    cfg['checkpoints_to_eval'] = checkpoints_to_eval
    logger.add_line('\n'+'='*30 + '     Checkpoints to Eval     '+ '='*30)
    logger.add_line(f'After ckpts to eval: {checkpoints_to_eval}')

    if len(checkpoints_to_eval):
        # Define dataloaders once
        train_loader = main_utils.build_dataloader(cfg['dataset'], cfg['num_workers'],  pretraining=False, mode='train', logger=logger)  
        val_loader = main_utils.build_dataloader(cfg['dataset'], cfg['num_workers'],  pretraining=False, mode='val', logger=logger) 

        # Define model bcz we dont want to carry forward num batches tracked and classifier params from previous checkpt training
        model = main_utils.build_model(cfg['model'], pretraining=False, dataset=train_loader.dataset, logger=logger) 

        if args.multiprocessing_distributed:
            if cfg['model']['MODEL_BASE']['BACKBONE_3D']['NAME'] == 'MinkUNet':
                model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Save initial model for lp so that all checkpts lp_training/val starts from 
        # the same initial params of the classifier
        if linear_probe:
            init_model_fn = f'{downstream_dir}/initial_linear_probe_model.pth.tar'
            if args.rank == 0 and not os.path.isfile(init_model_fn):
                torch.save(model.state_dict(), init_model_fn)
            
            if args.multiprocessing_distributed:
                torch.distributed.barrier()
        for ckpt in checkpoints_to_eval:
            logger.add_line('\n'+'='*30 + f'Evaluating ckpt: {ckpt}' +'='*30)
            cfg['pretrain_checkpoint'] = ckpt
            if linear_probe:
                cfg['pretrain_ckpt_epoch'] = int(ckpt.split('-ep')[1].split('.')[0])
                # ckpt_name = ckpt.split('.')[0]
                # downstream_dir = Path(downstream_dir) / ckpt_name
                # downstream_dir.mkdir(parents=True, exist_ok=True)
                init_model = torch.load(init_model_fn, map_location='cpu')
                model.load_state_dict(init_model)
            
            wandb_utils.init(cfg, args, Path(f'{downstream_dir}/wandb_run_id.txt'), pretraining=False)
            eval_one_ckpt(args, cfg, logger, 
                                    downstream_dir = f'{downstream_dir}/ckpt', 
                                    pretrain_model_dir=f'{pretrain_model_dir}/ckpt', 
                                    train_loader=train_loader, val_loader=val_loader, model=model,
                                    linear_probe=linear_probe)

            # # # if finetune more than one ckpt
            # if not linear_probe and run is not None:
            #     run.finish()
            # # record this epoch which has been evaluated
            # if args.rank == 0:
            #     with open(ckpt_record_file, 'a') as f:
            #         print('%s' % ckpt, file=f)
            logger.add_line('\n'+'='*30 + f'Ckpt {ckpt} has been evaluated'+ '='*30)

    if args.multiprocessing_distributed:
        dist.destroy_process_group()

def eval_one_ckpt(args, cfg, logger, 
                  downstream_dir, pretrain_model_dir, 
                  train_loader, val_loader, model,
                  linear_probe, 
                  tb_writter=None):
    ckp_manager_pretrained_model = main_utils.CheckpointManager(pretrain_model_dir, logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)
    ckp_manager_downstream = main_utils.CheckpointManager(downstream_dir, logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)

    # Load model state dict i.e. load checkpoint
    base_model_epoch = ckp_manager_pretrained_model.restore(fn=cfg['pretrain_checkpoint'], skip_model_layers=['global_step', 'num_batches_tracked'], model=model)


    if linear_probe:
        # Freeze Backbone
        for param in model.trunk[0].parameters():
            param.requires_grad = False

    # model To cuda
    model, args = main_utils.distribute_model_to_cuda(model, args)

    if not linear_probe:
        if 'head_lr' in cfg['optimizer']['lr']:
            if args.multiprocessing_distributed:
                params_to_optimize = [{"params": list(model.module.trunk[0].parameters()), "lr": cfg['optimizer']['lr']['base_lr']},
                                      {"params": list(model.module.segmentation_head.parameters()), "lr": cfg['optimizer']['lr']['head_lr']}]
            else:
                params_to_optimize = [{"params": list(model.trunk[0].parameters()), "lr": cfg['optimizer']['lr']['base_lr']},
                                      {"params": list(model.segmentation_head.parameters()), "lr": cfg['optimizer']['lr']['head_lr']}]
        else:
            params_to_optimize = list(model.parameters())
    else:
        params_to_optimize = [param for key, param in model.named_parameters() if param.requires_grad]
        if args.multiprocessing_distributed:
            model.module.trunk[0].eval()
        else:
            model.trunk[0].eval()     

    #logger.add_line(parameter_description(model))

    optimizer, scheduler = main_utils.build_optimizer(
            params=params_to_optimize,
            cfg=cfg['optimizer'],
            total_iters_each_epoch=len(train_loader),
            logger=logger)
        

    if 'SEGMENTATION_HEAD' in cfg['model']:
        #num_classes = cfg['model']['SEGMENTATION_HEAD']['CLS_FC'][-1]
        class_name_dict = {'SemanticKittiDataset': SEMANTIC_KITTI_LABELS,
                           'NuscenesDataset': NUSCENES_LABELS,
                           'WaymoDataset': WAYMO_LABELS
                           }
        class_names = class_name_dict[cfg['dataset']['DATASET_NAMES'][0]]
    else:
        raise NotImplementedError
    # else:
    #     # num_classes = 3
    #     class_names = cfg['dataset']['CLASS_NAMES']

    cfg['num_classes'] = len(class_names)
    cfg['class_names'] = class_names
    if 'val_interval' not in cfg:
        cfg['val_interval'] = 1
    
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']

    # # Optionally resume from the last downstream checkpoint
    if not linear_probe and cfg['resume']:
        if ckp_manager_downstream.checkpoint_exists(last=True):
            start_epoch = ckp_manager_downstream.restore(restore_last=True, model=model, optimizer=optimizer)
            scheduler.step(start_epoch*len(train_loader))
            logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(ckp_manager_downstream.last_checkpoint_fn(), start_epoch))
        else:
            logger.add_line("No checkpoint found at '{}'".format(ckp_manager_downstream.last_checkpoint_fn()))

    # cudnn.benchmark = True
    cudnn.enabled = False

    ############################ TRAIN Linear classifier head #########################################    
    evaluator = iouEval(n_classes=cfg['num_classes'], ignore=0)

    if linear_probe:
        eval_dict_ckpt = EvalMetrics(class_names, cfg['pretrain_ckpt_epoch'], save_dir=downstream_dir)
    for epoch in range(start_epoch, end_epoch):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
        logger.add_line('LR: {}'.format(scheduler.get_lr()))
        train_eval_metrics_dict_single_downstream_epoch = run_phase('train', train_loader, model, optimizer, scheduler, epoch, args, cfg, logger, tb_writter, evaluator)
        
        # Validate one epoch 
        if epoch >= cfg ['val_after_epochs']:
            if epoch % cfg['val_interval'] == 0 or epoch == (end_epoch-1):
                val_eval_metrics_dict_single_downstream_epoch = run_phase('val', val_loader, model, optimizer, scheduler, epoch, args, cfg, logger, tb_writter, evaluator)

        #Save linear probe ckpt
        ckp_manager_downstream.save(epoch+1, model=model, optimizer=optimizer)
        logger.add_line(f'Saved downstream checkpoint {ckp_manager_downstream.last_checkpoint_fn()} after ending epoch {epoch}, {epoch+1} is recorded for this chkp')
        
        if linear_probe:
            eval_dict_ckpt.push_back_single_epoch('train', epoch, train_eval_metrics_dict_single_downstream_epoch)
            eval_dict_ckpt.push_back_single_epoch('val', epoch, val_eval_metrics_dict_single_downstream_epoch)
        else:
            #finetune
            results_dict = train_eval_metrics_dict_single_downstream_epoch
            if epoch >= cfg ['val_after_epochs']:
                if epoch % cfg['val_interval'] == 0 or epoch == (end_epoch-1):
                    results_dict.update(val_eval_metrics_dict_single_downstream_epoch)
            wandb_utils.log(cfg, args, results_dict,  step=epoch)

    if linear_probe:
        ckpt_best_downstream_results = eval_dict_ckpt.get_best_dict()
        wandb_utils.log(cfg, args, ckpt_best_downstream_results,  step=cfg['pretrain_ckpt_epoch'])
        eval_dict_ckpt.save(args.rank)

    
def run_phase(phase, loader, model, optimizer, scheduler, epoch, args, cfg, logger, tb_writter, evaluator=None):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    batch_time = metrics_utils.AverageMeter(f'Avg Batch Process Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter(f'Avg Batch Load Time', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter(f'Total Loss', ':.3e') # total loss
    seg_celoss_meter = metrics_utils.AverageMeter(f'seg_celoss', ':.3e')
    seg_lovloss_meter = metrics_utils.AverageMeter(f'seg_lovloss', ':.3e')

   
    list_of_meters = [batch_time, data_time, loss_meter]
    
    if 'SEGMENTATION_HEAD' in cfg['model']:
        list_of_meters += [seg_celoss_meter, seg_lovloss_meter]


    list_of_meters_to_display = [batch_time.name, data_time.name, loss_meter.name]
    progress = utils.logger.ProgressMeter(len(loader), list_of_meters, phase=phase, meters_to_display=list_of_meters_to_display, epoch=epoch, logger=logger, tb_writter=tb_writter)

    #Evaluator metrics:
    eval_metrics_dict={}

    all_y_gt=[]
    all_preds=[]

    class_names = cfg['class_names']

    # switch to train mode
    model.train(phase == 'train')
    end = time.time()
    lr = scheduler.get_last_lr()
    for i, sample in enumerate(loader):
        torch.cuda.empty_cache()

        # measure data loading time
        data_time.update(time.time() - end) # Time to load one batch

        if phase == 'train':
            output_dict = model(sample)
        else:
            with torch.no_grad():
                output_dict = model(sample)

        # Segmentation loss
        loss = 0
        if 'SEGMENTATION_HEAD' in cfg['model']:
            loss += output_dict['loss_seg']
            seg_celoss_meter.update(output_dict['loss_seg_CELoss'])
            seg_lovloss_meter.update(output_dict['loss_seg_LovLoss'])

        loss_meter.update(loss.item())
       
        # compute gradient and do SGD step during training
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            scheduler.step() # fordownstream: reduce LR step every iterations


        if evaluator is not None:
            # convert output probabilities to predicted class 

            #Update progress
            all_y_gt.append(output_dict['output']['seg_labels'].view(-1))
            all_preds.append(output_dict['output']['pred_labels'].view(-1))
            evaluator.addBatch(output_dict['output']['pred_labels'].long().cpu().numpy(), output_dict['output']['seg_labels'].long().cpu().numpy())
            evaluator.addLoss(loss.item())

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

    if tb_writter is not None:
        for meter in progress.meters:
            tb_writter.add_scalar('{}-epoch/{}'.format(phase, meter.name), meter.avg, epoch)
    
    prefix= '{}/{}'.format(cfg['model']['job_type'], phase)
    eval_metrics_dict = {f'{prefix}/epoch': epoch}

    if phase == 'train':
        if len(optimizer.param_groups) > 1:
            eval_metrics_dict[f'{prefix}/lr_backbone'] = lr[0] #optimizer.param_groups[0]['lr']
            eval_metrics_dict[f'{prefix}/lr_head'] = lr[1] #optimizer.param_groups[1]['lr']
        else:
            eval_metrics_dict[f'{prefix}/lr']=lr[0]
    for meter in progress.meters:
        eval_metrics_dict[prefix + '/' + meter.name +'-epoch'] = meter.avg

    # Evaluator:
    accuracy=0 
    mean_iou=0
    if evaluator is not None:
        mean_iou, class_iou = evaluator.getIoU()
        accuracy = 100. * evaluator.getacc()
        mean_iou = mean_iou.item()
        accuracy = accuracy.item()
        eval_metrics_dict[f'{prefix}/acc'] = accuracy
        eval_metrics_dict[f'{prefix}/mIoU'] = mean_iou
        eval_metrics_dict[f'{prefix}/loss'] = evaluator.getloss()

        for class_num in range(class_iou.shape[0]):
            eval_metrics_dict[f'{prefix}/per_class_iou/{class_names[class_num]}'] = class_iou[class_num].item()

        evaluator.reset()


        logger.add_line(f'\n{phase} Accuracy: {accuracy}')
        logger.add_line(f'{phase} mIoU: {mean_iou}')


    # wandb_utils.log(cfg, args, eval_metrics_dict, epoch)

    return eval_metrics_dict

if __name__ == '__main__':
    main()
