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
from utils.ioueval import iouEval
from utils.data_map import WAYMO_LABELS
from models.base_ssl3d_model import parameter_description


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
    args = main_utils.initialize_distributed_backend(args)
    logger, _, downstream_dir, pretrain_model_dir, phase_name = main_utils.prep_environment(args, cfg, pretraining=False)

    checkpoints_to_eval, ckpt_record_file = main_utils.get_ckpts_to_eval(cfg, logger, 
                                                                         pretrain_model_dir=pretrain_model_dir, 
                                                                         eval_list_dir=downstream_dir)

    # Define dataloaders once
    train_loader = main_utils.build_dataloader(cfg['dataset'], cfg['num_workers'],  pretraining=False, mode='train', logger=logger)  
    val_loader = main_utils.build_dataloader(cfg['dataset'], cfg['num_workers'],  pretraining=False, mode='val', logger=logger) 

    # Define model
    model = main_utils.build_model(cfg['model'], pretraining=False, dataset=train_loader.dataset, logger=logger) 

    # configuration
    linear_probe = cfg['model']['linear_probe']
    downstream_task = 'segmentation' if 'SEGMENTATION_HEAD' in cfg['model'] else 'detection'

    for ckpt in checkpoints_to_eval:
        logger.add_line('\n'+'='*30 + f'Evaluating ckpt: {ckpt}' +'='*30)
        cfg['checkpoint'] = ckpt #TODO
        ckpt_name = ckpt.split('.')[0]
        _, run = wandb_utils.reinit(cfg, args, job_type=phase_name) #TODO

        downstream_ckpt_dir = Path(downstream_dir) / ckpt_name
        downstream_ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_dict = eval_one_ckpt(args, cfg, logger, 
                                  downstream_dir = str(downstream_ckpt_dir), 
                                  pretrain_model_dir=pretrain_model_dir, 
                                  train_loader=train_loader, val_loader=val_loader,
                                  model=model,
                                  linear_probe=linear_probe,
                                  downstream_task=downstream_task)
        if eval_dict is not None:

            highest_metric = -1
            highest_metric = wandb_utils.summary(cfg, args, eval_dict, step=None, highest_metric=highest_metric)

            if run is not None:
                run.finish()
            # record this epoch which has been evaluated
            with open(ckpt_record_file, 'a') as f:
                print('%s' % ckpt, file=f)
            logger.add_line('\n'+'='*30 + f'Ckpt {ckpt} has been evaluated'+ '='*30)

    if args.multiprocessing_distributed:
        dist.destroy_process_group()

def eval_one_ckpt(args, cfg, logger, 
                  downstream_dir, pretrain_model_dir, 
                  train_loader, val_loader, model,
                  linear_probe, downstream_task,
                  tb_writter=None):
    ckp_manager_pretrained_model = main_utils.CheckpointManager(pretrain_model_dir, logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)
    ckp_manager_linear = main_utils.CheckpointManager(downstream_dir, logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)

    # Load model state dict i.e. load checkpoint
    base_model_epoch = ckp_manager_pretrained_model.restore(fn=cfg['checkpoint'], skip_model_layers=['global_step', 'num_batches_tracked'], model=model)
        
    logger.add_line(f"Downstream Epoch {base_model_epoch-1}")


    if linear_probe:
        # Freeze Backbone
        for param in model.trunk[0].parameters():
            param.requires_grad = False


    if args.multiprocessing_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # model To cuda
    model, args = main_utils.distribute_model_to_cuda(model, args)

    # Define segmentation criterion TODO
    # train_criterion = None
    # if downstream_task == 'segmentation':
    #     train_criterion =  main_utils.build_criterion(cfg['loss'], cfg['cluster'], logger=logger) 
    #     train_criterion = train_criterion.cuda()
    
    if not linear_probe:
        params_to_optimize = [model.parameters()]
    else:
        params_to_optimize = [param for key, param in model.named_parameters() if param.requires_grad]
        if args.multiprocessing_distributed:
            model.module.trunk[0].eval()
        else:
            model.trunk[0].eval()     

    # logger.add_line(parameter_description(model))

    optimizer, scheduler = main_utils.build_optimizer(
            params=params_to_optimize,
            cfg=cfg['optimizer'],
            total_iters_each_epoch=len(train_loader),
            logger=logger)
        

    if 'SEGMENTATION_HEAD' in cfg['model']:
        num_classes = cfg['model']['SEGMENTATION_HEAD']['CLS_FC'][-1]
    else:
        num_classes = 3

    cfg['num_classes'] = num_classes
    
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']

    # Optionally resume from a checkpoint
    if cfg['resume']:
        if ckp_manager_linear.checkpoint_exists(last=True):
            start_epoch = ckp_manager_linear.restore(restore_last=True, model=model, optimizer=optimizer)
            scheduler.step(start_epoch)
            logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(ckp_manager_linear.last_checkpoint_fn(), start_epoch))
        else:
            logger.add_line("No checkpoint found at '{}'".format(ckp_manager_linear.last_checkpoint_fn()))

    cudnn.benchmark = True

    ############################ TRAIN Linear classifier head #########################################
    test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
    val_accuracies = []
    val_mIoUs = []
    epochs = []
    evaluator = iouEval(n_classes=num_classes, ignore=0)
    for epoch in range(start_epoch, end_epoch):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
        logger.add_line('LR: {}'.format(scheduler.get_lr()))
        run_phase('train', train_loader, model, optimizer, scheduler, epoch, args, cfg, logger, tb_writter, evaluator)
        
        # Validate one epoch
        accuracy, m_iou= run_phase('val', val_loader, model, optimizer, scheduler, epoch, args, cfg, logger, tb_writter, evaluator)

        #Save linear probe ckpt
        if ((epoch % test_freq) == 0) or (epoch == end_epoch - 1):
            ckp_manager_linear.save(epoch+1, model=model, optimizer=optimizer)
            logger.add_line(f'Saved linear_probe checkpoint {ckp_manager_linear.last_checkpoint_fn()} after ending epoch {epoch}, {epoch+1} is recorded for this chkp')
        
        val_accuracies.append(accuracy)
        val_mIoUs.append(m_iou)
        epochs.append(epoch)

    if len(val_accuracies):
        max_acc_epoch = epochs[np.array(val_accuracies).argmax()]
        eval_dict = {
                    'base_model_epoch': base_model_epoch,
                    'lp_max_acc_epoch': max_acc_epoch,
                    'max_acc': np.array(val_accuracies).max(),
                    'std_acc' : np.array(val_accuracies).std(),
                    'mean_acc' : np.array(val_accuracies).mean(),
                    'min_acc' : np.array(val_accuracies).min(),
                    'max_mIou' : np.array(val_mIoUs).max(),
                    'std_mIou' : np.array(val_mIoUs).std(),
                    'mean_mIou' : np.array(val_mIoUs).mean(),
                    'min_mIou' : np.array(val_mIoUs).min()
                    }
        return eval_dict
    else:
        return None
    
def run_phase(phase, loader, model, optimizer, scheduler, epoch, args, cfg, logger, tb_writter, evaluator=None):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    batch_time = metrics_utils.AverageMeter(f'Avg Batch Process Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter(f'Avg Batch Load Time', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter(f'Total Loss', ':.3e') # total loss
    det_cls_loss_meter = metrics_utils.AverageMeter(f'det_cls_loss', ':.3e')
    det_reg_loss_meter = metrics_utils.AverageMeter(f'det_reg_loss', ':.3e')
    det_cls_rcnn_loss_meter = metrics_utils.AverageMeter(f'det_cls_rcnn_loss', ':.3e')
    det_reg_rcnn_loss_meter = metrics_utils.AverageMeter(f'det_reg_rcnn_loss', ':.3e')
    seg_celoss_meter = metrics_utils.AverageMeter(f'seg_celoss', ':.3e')
    seg_lovloss_meter = metrics_utils.AverageMeter(f'seg_lovloss', ':.3e')

   
    list_of_meters = [batch_time, data_time, loss_meter]
    
    if 'MODEL_DET_HEAD' in cfg.model:
        list_of_meters += [det_cls_loss_meter, det_reg_loss_meter]
        if 'ROI_HEAD' in cfg.model.MODEL_DET_HEAD:
            list_of_meters += [det_cls_rcnn_loss_meter, det_reg_rcnn_loss_meter]
    
    if 'SEGMENTATION_HEAD' in cfg['model']:
        list_of_meters += [seg_celoss_meter, seg_lovloss_meter]


    list_of_meters_to_display = [batch_time.name, data_time.name, loss_meter.name]
    progress = utils.logger.ProgressMeter(len(loader), list_of_meters, phase=phase, meters_to_display=list_of_meters_to_display, epoch=epoch, logger=logger, tb_writter=tb_writter)

    #Evaluator metrics:
    eval_metrics_dict={}

    all_y_gt=[]
    all_preds=[]
    if 'SEGMENTATION_HEAD' in cfg['model']:
        class_names = WAYMO_LABELS
    else:
        class_names = cfg['dataset']['CLASS_NAMES']

    # switch to train mode
    model.train(phase == 'train')
    end = time.time()
    lr = scheduler.get_last_lr()[0]
    for i, sample in enumerate(loader):
        torch.cuda.empty_cache()

        # measure data loading time
        data_time.update(time.time() - end) # Time to load one batch

        if phase == 'train':
            output_dict = model(sample)
        else:
            with torch.no_grad():
                output_dict = model(sample)

         # contrastive loss
        #loss = criterion(output_dict['output'], output_dict['output_moco'])
        # Segmentation loss
        loss = 0
        if 'SEGMENTATION_HEAD' in cfg['model']:
            loss += output_dict['loss_seg']
            seg_celoss_meter.update(output_dict['loss_seg_CELoss'])
            seg_lovloss_meter.update(output_dict['loss_seg_LovLoss'])

        # detection loss
        if 'MODEL_DET_HEAD' in cfg['model']:
            loss += output_dict['loss_det_head']
            det_cls_loss_meter.update(output_dict['loss_det_cls'])
            det_reg_loss_meter.update(output_dict['loss_det_reg'])
            if 'ROI_HEAD' in cfg.model:
                det_cls_rcnn_loss_meter.update(output_dict['loss_det_cls_rcnn'])
                det_reg_rcnn_loss_meter.update(output_dict['loss_det_reg_rcnn'])

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
    
    eval_metrics_dict = {f'{phase}/epoch': epoch}

    if phase == 'train':
        eval_metrics_dict[f'{phase}/lr'] =lr
    for meter in progress.meters:
        eval_metrics_dict[phase + '/' + meter.name +'-epoch'] = meter.avg

    # Evaluator:
    accuracy=0 
    mean_iou=0
    if evaluator is not None:
        mean_iou, class_iou = evaluator.getIoU()
        accuracy = 100. * evaluator.getacc()
        mean_iou = mean_iou.item()
        accuracy = accuracy.item()
        eval_metrics_dict[f'{phase}/acc'] = accuracy
        eval_metrics_dict[f'{phase}/mIoU'] = mean_iou
        eval_metrics_dict[f'{phase}/loss'] = evaluator.getloss()

        if cfg['num_classes'] > 2:
            # per class iou
            for class_num in range(class_iou.shape[0]):
                eval_metrics_dict[f'{phase}/per_class_iou/{class_names[class_num]}'] = class_iou[class_num].item()
        else:
            eval_metrics_dict[f'{phase}/per_class_iou/background'] = class_iou[0].item()
            eval_metrics_dict[f'{phase}/per_class_iou/foreground'] = class_iou[1].item()



        evaluator.reset()


        logger.add_line(f'\n {phase} Accuracy: {accuracy}')
        logger.add_line(f'{phase} mIoU: {mean_iou}, IoU per class: {class_iou}')


    wandb_utils.log(cfg, args, eval_metrics_dict, epoch)

    return accuracy, mean_iou

if __name__ == '__main__':
    main()