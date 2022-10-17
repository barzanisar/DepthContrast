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

import torch.multiprocessing as mp
import numpy as np
  
import utils.logger
from utils import main_utils, wandb_utils
from models.trunks.mlp import MLP
from utils.ioueval import iouEval
from utils.data_map import LABELS
import glob
from pathlib import Path



parser = argparse.ArgumentParser(description='PyTorch Self Supervised Training in 3D')

parser.add_argument('--cfg', type=str, default=None, help='specify the config for training')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training') #remove, ws is total no. of gpus, NOT nodes!
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training') #remove
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'fair'], default='none')
parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                    help='url used to set up distributed training') #tc port
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1024, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--local_rank', default=0, type=int,
                    help='local process id i.e. GPU id to use.') #local_rank = 0
parser.add_argument('--ngpus', default=4, type=int,
                    help='number of GPUs to use.') #not needed
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    assert cfg.get('linear_probe', False)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    if args.multiprocessing_distributed:
        assert args.launcher  == 'pytorch'
        args.ngpus = torch.cuda.device_count() 
        main_worker(args.local_rank, args.ngpus, args, cfg)
    else:
        args.ngpus = torch.cuda.device_count()
        # Simply call main_worker function
        main_worker(args.local_rank, args.ngpus, args, cfg)

def eval_one_ckpt(args, cfg, logger, tb_writter, linear_probe_dir, linear_probe_dataset=None):
    #checkpoint_name = cfg['checkpoint'].split('.')[0]

    ckp_manager_base_model = main_utils.CheckpointManager(linear_probe_dir.split(f'/linear_probe_{linear_probe_dataset}')[0] + '/train', logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)
    ckp_manager_linear = main_utils.CheckpointManager(linear_probe_dir, logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)

    # Define model
    model = main_utils.build_model(cfg['model'], cfg['cluster'], logger=logger, linear_probe=True)

    # Load model state dict i.e. load checkpoint
    if cfg.get('checkpoint', None) is not None:
        base_model_epoch = ckp_manager_base_model.restore(fn=cfg['checkpoint'], model=model)
    else:
        base_model_epoch = ckp_manager_base_model.restore(restore_last=True, model=model)
        

    logger.add_line(f"Linear Probing Epoch {base_model_epoch-1}")
    # Delete moco encoder
    del model.trunk[1]
    #print(model)

    # Freeze training for feature layers
    for param in model.parameters():
        param.requires_grad = False

    # linear classifier
    model.trunk[0].head = MLP(cfg["model"]["linear_probe_dim"], use_dropout=len(cfg["model"]["linear_probe_dim"])>2)

    # model To cuda #TODO remove unused params and set to False
    model, args = main_utils.distribute_model_to_cuda(model, args)

    # Define criterion 
    train_criterion =  main_utils.build_criterion(cfg['loss'], cfg['cluster'], linear_probe=True, logger=logger) 
    train_criterion = train_criterion.cuda()
    
    # Define dataloaders
    train_loader, val_loader = main_utils.build_dataloaders_train_val(cfg['dataset'], cfg['num_workers'], cfg['cluster'], args.multiprocessing_distributed, logger, linear_probe=True)       

    if args.multiprocessing_distributed:     
        # Define optimizer
        optimizer, scheduler = main_utils.build_optimizer(
            params=model.module.trunk[0].head.parameters(),
            cfg=cfg['optimizer'],
            logger=logger)
    else:
        # Define optimizer
        optimizer, scheduler = main_utils.build_optimizer(
            params=model.trunk[0].head.parameters(),
            cfg=cfg['optimizer'],
            logger=logger)

    
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
    evaluator = iouEval(n_classes=cfg['loss']['args']['num_classes'], ignore=cfg['loss']['args']['ignore_index'])
    for epoch in range(start_epoch, end_epoch):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
        logger.add_line('LR: {}'.format(scheduler.get_lr()))
        run_phase('train', train_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, tb_writter, evaluator)
        scheduler.step(epoch)

        # Validate one epoch
        accuracy, m_iou= run_phase('val', val_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, tb_writter, evaluator)

        #Save linear probe ckpt
        if ((epoch % test_freq) == 0) or (epoch == end_epoch - 1):
            ckp_manager_linear.save(epoch+1, model=model, optimizer=optimizer, train_criterion=train_criterion)
            logger.add_line(f'Saved linear_probe checkpoint for testing {ckp_manager_linear.last_checkpoint_fn()} after ending epoch {epoch}, {epoch+1} is recorded for this chkp')
        
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
    

def main_worker(gpu, ngpus, args, cfg):
    args.local_rank = gpu
    ngpus_per_node = ngpus
    cfg['cluster'] = False
    cfg['linear_probe'] = True
    linear_probe_dataset = 'semKitti' if 'SEMANTIC_KITTI' in cfg['dataset'] else 'dense'

    # Setup environment
    args = main_utils.initialize_distributed_backend(args, ngpus_per_node) ### Use other method instead
    logger, tb_writter, linear_probe_dir = main_utils.prep_environment(args, cfg, linear_probe_dataset=linear_probe_dataset)
    
    
    checkpoints_to_eval = []
    model_dir = Path(cfg['model']['model_dir']) / cfg['model']['name']
    ckpt_dir = model_dir / 'train'
    assert os.path.isdir(ckpt_dir) 
    assert os.path.isdir(linear_probe_dir) 
    # evaluated ckpt record
    ckpt_record_file = Path(linear_probe_dir) / 'eval_list.txt'
    with open(ckpt_record_file, 'a'):
        pass
    if args.eval_all or cfg['checkpoint'] == 'all':
        ckpt_list = glob.glob(os.path.join(ckpt_dir, 'checkpoint*.pth.tar'))
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
        checkpoints_to_eval.append(cfg['checkpoint'])
    
    max_acc_best = (-1, 'None')
    mean_acc_best = (-1, 'None')
    max_miou_best = (-1, 'None')
    mean_miou_best = (-1, 'None')

    for ckpt in checkpoints_to_eval:
        logger.add_line(f' Evaluating ckpt: {ckpt}')
        cfg['checkpoint'] = ckpt
        ckpt_name = ckpt.split('.')[0]
        _, run = wandb_utils.reinit(cfg, args, job_type='linear_probe')

        linear_probe_ckpt_dir = Path(linear_probe_dir) / ckpt_name
        linear_probe_ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_dict = eval_one_ckpt(args, cfg, logger, tb_writter, linear_probe_dir = str(linear_probe_ckpt_dir), linear_probe_dataset=linear_probe_dataset)
        if eval_dict is not None:
            if eval_dict['max_acc'] > max_acc_best[0]:
                max_acc_best = (eval_dict['max_acc'], ckpt)
            if eval_dict['mean_acc'] > mean_acc_best[0]:
                mean_acc_best = (eval_dict['mean_acc'], ckpt)
            if eval_dict['max_mIou'] > max_miou_best[0]:
                max_miou_best = (eval_dict['max_mIou'], ckpt)
            if eval_dict['mean_mIou'] > mean_miou_best[0]:
                mean_miou_best = (eval_dict['mean_mIou'], ckpt)

            highest_metric = -1
            highest_metric = wandb_utils.summary(cfg, args, eval_dict, step=None, highest_metric=highest_metric) #step=int(eval_dict['base_model_epoch'])

            run.finish()
            # record this epoch which has been evaluated
            with open(ckpt_record_file, 'a') as f:
                print('%s' % ckpt, file=f)
            logger.add_line(f'Ckpt {ckpt} has been evaluated')
    
    logger.add_line(f'max_acc_best: {max_acc_best}')
    logger.add_line(f'mean_acc_best: {mean_acc_best}')
    logger.add_line(f'max_miou_best: {max_miou_best}')
    logger.add_line(f'mean_miou_best: {mean_miou_best}')



def class_stats(y_gt):
    num_points = y_gt.view(-1).shape[0]
    background_points_mask = y_gt == 0
    car_mask = y_gt == 1
    ped_mask = y_gt == 2
    rv_mask = y_gt == 3
    lv_mask = y_gt == 4
    background_percent = background_points_mask.sum()/num_points
    car_percent = car_mask.sum()/num_points
    ped_percent = ped_mask.sum()/num_points
    rv_percent = rv_mask.sum()/num_points
    lv_percent = lv_mask.sum()/num_points

    print(f'bk: {background_percent}, car: {car_percent}, ped: {ped_percent}, rv:{rv_percent}, lv:{lv_percent}')

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

def ignore_cyclist(output, y_gt):
    noncyclist_mask = y_gt != 3
    return output[noncyclist_mask], y_gt[noncyclist_mask]

def run_phase(phase, loader, model, optimizer, criterion, epoch, args, cfg, logger, tb_writter, evaluator=None):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    batch_time = metrics_utils.AverageMeter('Batch Process Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Batch Load Time', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.3e')
    # accuracy_meter = metrics_utils.AverageMeter('Accuracy', ':.2f')
    
    # #IoU 
    # mIoU_meter = metrics_utils.AverageMeter('mIoU', ':.3f')    

    progress = utils.logger.ProgressMeter(len(loader), [loss_meter], phase=phase, epoch=epoch, logger=logger, tb_writter=tb_writter)

    #Evaluator metrics:
    eval_metrics_dict={}

    # switch to train mode
    model.train(phase == 'train')

    #Accuracy
    correct = 0
    total = 0

    all_y_gt=[]
    all_preds=[]
    num_classes = cfg['loss']['args']['num_classes']
    
    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end) # Time to load one batch


        if phase == 'train':
            output_feat = model(sample) #output=(batch_size, Npoints, num classes), coords=(8pcs, Npoints, 3)
        else:
            with torch.no_grad():
                output_feat = model(sample)
        
        if num_classes == 2:
            # For background/foreground classes, undersample background class 
            output, y_gt = undersample_bk_class(output_feat[0], sample['labels'])
        else:
            output = output_feat[0].view(-1, 20)
            y_gt = sample['labels'].view(-1)


        # Load y_gt to gpu:
        y_gt = main_utils.recursive_copy_to_gpu(y_gt)
        loss = criterion(output, y_gt)
        
        loss_meter.update(loss.item(), output.size(0))

        # compute gradient and do SGD step during training
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # convert output probabilities to predicted class 
        pred = output.max(dim=1)[1] #pred_y 

        #Update progress
        all_y_gt.append(y_gt.view(-1))
        all_preds.append(pred.view(-1))

        # correct = pred.eq(y_gt).sum().item()
        # correct /= y_gt.size(0)
        # batch_acc = (correct * 100.)

        if evaluator is not None:
            evaluator.addBatch(pred.long().cpu().numpy(), y_gt.long().cpu().numpy())
            evaluator.addLoss(loss.item())

    
        # compare predictions to true label
        correct += np.sum(pred.eq(y_gt).cpu().numpy())
        total += y_gt.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end) # This is printed as Time # Time to train one batch
        end = time.time()

        # print to terminal and tensorboard
        if (i+1) % cfg['print_freq'] == 0 or i == 0 or i+1 == len(loader):
            progress.display(i+1)
    

    # Evaluator:
    mean_iou, class_iou = evaluator.getIoU()
    accuracy = 100. * evaluator.getacc()
    mean_iou = mean_iou.item()
    accuracy = accuracy.item()
    eval_metrics_dict={f'{phase}/loss': evaluator.getloss(),
                        f'{phase}/acc': accuracy,
                        f'{phase}/mIoU': mean_iou}


    if num_classes > 2:
        # per class iou
        for class_num in range(class_iou.shape[0]):
            eval_metrics_dict[f'{phase}/per_class_iou/{LABELS[class_num]}'] = class_iou[class_num].item()
    else:
        eval_metrics_dict[f'{phase}/per_class_iou/background'] = class_iou[0].item()
        eval_metrics_dict[f'{phase}/per_class_iou/foreground'] = class_iou[1].item()



    evaluator.reset()


    logger.add_line(f'\n {phase} Accuracy: {accuracy}')
    logger.add_line(f'{phase} mIoU: {mean_iou}, IoU per class: {class_iou}')



    # Sync metrics across all GPUs and print final averages
    if args.multiprocessing_distributed:
        progress.synchronize_meters(args.local_rank)
        progress.display(len(loader)*args.world_size)


    wandb_utils.log(cfg, args, eval_metrics_dict, epoch)

    return accuracy, mean_iou


if __name__ == '__main__':
    main()