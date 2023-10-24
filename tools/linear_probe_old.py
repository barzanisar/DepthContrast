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

#from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     print('ERROR: cant spawn')
#     pass

import torch.multiprocessing as mp
import numpy as np
  
import utils.logger
from utils import main_utils, wandb_utils
from models.trunks.mlp import MLP


parser = argparse.ArgumentParser(description='PyTorch Self Supervised Training in 3D')

parser.add_argument('--cfg', type=str, default=None, help='specify the config for training')
parser.add_argument('--quiet', action='store_true')

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
        assert args.launcher in ['pytorch', 'slurm', 'fair']
        if False: #args.launcher == 'fair'
            num_nodes = int(os.environ['SLURM_NNODES'])
            args.rank = int(os.environ['SLURM_NODEID'])
            node0 = 'gra' + os.environ['SLURM_NODELIST'][4:8]
            # print("=" * 30 + "   DDP   " + "=" * 30)
            # print(f"node0 : {node0}")
            # print(f"num_nodes : {num_nodes}")
            args.dist_url = f"tcp://{node0}:1234" #'tcp://127.0.0.1:29500' "tcp://gra1154:29500"
            ngpus_per_node = args.ngpus
            args.world_size = ngpus_per_node * num_nodes #total number of gpus
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
        else:
            args.ngpus = torch.cuda.device_count()  # remove for fair
            # print("=" * 30 + "   DDP   " + "=" * 30)
            # print(f"args.ngpus : {args.ngpus}")
            # print(f"args.local_rank : {args.local_rank}")
            # print(f"args.launcher : {args.launcher}")
            main_worker(args.local_rank, args.ngpus, args, cfg)
    else:
        args.ngpus = torch.cuda.device_count()  # remove for fair
        # Simply call main_worker function
        main_worker(args.local_rank, args.ngpus, args, cfg)
    
def main_worker(gpu, ngpus, args, cfg):
    args.local_rank = gpu
    ngpus_per_node = ngpus
    cfg['cluster'] = False
    cfg['linear_probe'] = True

    # Setup environment
    args = main_utils.initialize_distributed_backend(args, ngpus_per_node) ### Use other method instead
    logger, tb_writter, model_dir = main_utils.prep_environment(args, cfg)
    wandb_utils.init(cfg, args, job_type='linear_probe')
    ckp_manager_base_model = main_utils.CheckpointManager(model_dir.split('/linear_probe')[0] + '/train', logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)
    ckp_manager_linear = main_utils.CheckpointManager(model_dir, logger=logger, rank=args.rank, dist=args.multiprocessing_distributed)

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
    val_epoch_accuracy = -1
    val_epoch_obj_accuracy = -1
    for epoch in range(start_epoch, end_epoch):
        # if (epoch % cfg['ckpt_save_interval']) == 0:
        #     #ckp_manager_linear.save(epoch, model=model, train_criterion=train_criterion, optimizer=optimizer, filename='checkpoint-ep{}.pth.tar'.format(epoch))
        #     logger.add_line(f'Saved linear_probe checkpoint checkpoint-ep{epoch}.pth.tar before beginning epoch {epoch}')

        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
        logger.add_line('LR: {}'.format(scheduler.get_lr()))
        run_phase('train', train_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, tb_writter)
        scheduler.step(epoch)

        # Validate one epoch
        accuracy, obj_accuracy = run_phase('val', val_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, tb_writter)

        #TODO
        if ((epoch % test_freq) == 0) or (epoch == end_epoch - 1):
            ckp_manager_linear.save(epoch+1, model=model, optimizer=optimizer, train_criterion=train_criterion)
            logger.add_line(f'Saved linear_probe checkpoint for testing {ckp_manager_linear.last_checkpoint_fn()} after ending epoch {epoch}, {epoch+1} is recorded for this chkp')
        if accuracy > val_epoch_accuracy:
            logger.add_line('='*30 + 'VALIDATION ACCURACY INCREASED' + '='*30)
            #ckp_manager_linear.save(epoch+1, model=model, optimizer=optimizer, train_criterion=train_criterion, filename='checkpoint-best-acc-ep{}.pth.tar'.format(epoch+1))
            #logger.add_line(f'Saved Best linear_probe checkpoint checkpoint-best-acc-ep{epoch+1}.pth.tar after ending epoch {epoch}, {epoch+1} is recorded for this chkp')
            logger.add_line(f'Epoch {epoch}: Val Accuracy increased from {val_epoch_accuracy} to {accuracy}')
            val_epoch_accuracy = accuracy
        if obj_accuracy > val_epoch_obj_accuracy:
            logger.add_line('='*30 + 'VALIDATION OBJECT ACCURACY INCREASED' + '='*30)
            logger.add_line(f'Epoch {epoch}: Val Obj Accuracy increased from {val_epoch_obj_accuracy} to {obj_accuracy}')
            val_epoch_obj_accuracy = obj_accuracy


def class_stats(labels):
    num_points = labels.view(-1).shape[0]
    background_points_mask = labels == 0
    car_mask = labels == 1
    ped_mask = labels == 2
    rv_mask = labels == 3
    lv_mask = labels == 4
    background_percent = background_points_mask.sum()/num_points
    car_percent = car_mask.sum()/num_points
    ped_percent = ped_mask.sum()/num_points
    rv_percent = rv_mask.sum()/num_points
    lv_percent = lv_mask.sum()/num_points

    print(f'bk: {background_percent}, car: {car_percent}, ped: {ped_percent}, rv:{rv_percent}, lv:{lv_percent}')

def undersample_bk_class(output, labels):
        background_points_mask = labels == 0
        obj_points_mask = labels > 0


        bk_labels = labels[background_points_mask]
        bk_output = output[background_points_mask]

        perm = torch.randperm(bk_labels.size(0))
        idx = perm[:obj_points_mask.sum().item()]
        downsampled_bk_labels = bk_labels[idx]
        downsampled_bk_output = bk_output[idx]

        new_labels = torch.cat((downsampled_bk_labels, labels[obj_points_mask]))
        new_output = torch.cat((downsampled_bk_output, output[obj_points_mask]))

        return new_output, new_labels

def ignore_cyclist(output, labels):
    noncyclist_mask = labels != 3
    return output[noncyclist_mask], labels[noncyclist_mask]

def run_phase(phase, loader, model, optimizer, criterion, epoch, args, cfg, logger, tb_writter):
    from utils import metrics_utils
    #class_names = ['PassengerCar', 'Pedestrian', 'RidableVehicle']
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    batch_time = metrics_utils.AverageMeter('Batch Process Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Batch Load Time', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.3e')
    accuracy_meter = metrics_utils.AverageMeter('Accuracy', ':.2f')
    obj_recall_meter = metrics_utils.AverageMeter('Obj Accuracy', ':.2f')
    obj_precision_meter = metrics_utils.AverageMeter('Obj Precision', ':.2f')
    
    #IoU 
    mIoU_meter = metrics_utils.AverageMeter('mIoU', ':.3f')    

    progress = utils.logger.ProgressMeter(len(loader), [loss_meter, accuracy_meter, obj_recall_meter, obj_precision_meter, mIoU_meter], phase=phase, epoch=epoch, logger=logger, tb_writter=tb_writter)

    # switch to train mode
    model.train(phase == 'train')

    #Accuracy
    correct = 0
    total = 0

    # Recall
    correct_obj_points= 0
    total_obj_points = 0

    # Precision
    correct_obj_points_predictions = 0
    total_obj_points_predictions = 0

    all_labels=[]
    all_preds=[]
    
    end = time.time()
    device = args.local_rank if args.local_rank is not None else 0
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end) # Time to load one batch


        if phase == 'train':
            output_feat = model(sample) #output=(batch_size, Npoints, num classes), coords=(8pcs, Npoints, 3)
        else:
            with torch.no_grad():
                output_feat = model(sample)
        

        output = output_feat[0]
        # # compute loss
        # labels = main_utils.get_labels(sample['gt_boxes_lidar'], output_dict[0]['linear_probe_xyz'].detach().cpu().numpy(), cfg['dataset']['LABEL_TYPE']) #(8, 16384)
        
        # # Ignore ridable vehicle
        # output, labels = ignore_cyclist(output, labels)

        # Undersampling bk class from labels and output to have 1:1 ratio of bk:objects class
        output, labels = undersample_bk_class(output, sample['labels'])

        #Print class stats
        #class_stats(labels)

        # Load labels to gpu:
        labels = main_utils.recursive_copy_to_gpu(labels)
        loss = criterion(output, labels)
        #loss = criterion(output.transpose(1,2), labels)
        loss_meter.update(loss.item(), output.size(0))

        # compute gradient and do SGD step during training
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1] #pred_labels
        pred = torch.squeeze(pred)

        #Update progress
        all_labels.append(labels.view(-1))
        all_preds.append(pred.view(-1))
    
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(labels)).cpu().numpy())
        total += labels.size(0) #* labels.size(1)

        #Obj accuracy or recall = num correctly pred gt obj points/ num total gt obj points
        gt_obj_flag = labels>0
        total_obj_points += gt_obj_flag.sum().item()
        pred_obj = pred[gt_obj_flag]
        labels_obj = labels[gt_obj_flag]
        correct_obj_points += np.sum(np.squeeze(pred_obj.eq(labels_obj)).cpu().numpy())

        # accuracy_meter.update(100. * correct/total)
        # obj_recall_meter.update(100. * correct_obj_points / total_obj_points)

        #Precision
        pred_obj_flag = pred>0
        total_obj_points_predictions += pred_obj_flag.sum().item()
        obj_predictions = pred[pred_obj_flag]
        labels_for_obj_predictions = labels[pred_obj_flag]
        correct_obj_points_predictions += np.sum(np.squeeze(obj_predictions.eq(labels_for_obj_predictions)).cpu().numpy())


        # measure elapsed time
        batch_time.update(time.time() - end) # This is printed as Time # Time to train one batch
        end = time.time()

        # print to terminal and tensorboard
        step = epoch * len(loader) + i #sample is a batch of 8 transformed point clouds, len(loader) is the total number of batches
        if (i+1) % cfg['print_freq'] == 0 or i == 0 or i+1 == len(loader):
            progress.display(i+1)
    
    ignore_index = cfg['loss']['args']['ignore_index']
    if ignore_index < 0:
        ignore_index=None

    num_classes = cfg['loss']['args']['num_classes']
    m_IoU, fw_IoU, IoU = main_utils.compute_IoU(torch.cat(all_preds).view(-1), torch.cat(all_labels).view(-1), num_classes = num_classes, ignore_index=ignore_index)

    accuracy = 100. * correct / total
    obj_accuracy = 100. * correct_obj_points / total_obj_points
    precision = 100. * correct_obj_points_predictions / total_obj_points_predictions

    mIoU_meter.update(m_IoU)
    accuracy_meter.update(accuracy)
    obj_recall_meter.update(obj_accuracy)
    obj_precision_meter.update(precision)
    logger.add_line('\n %s Accuracy: %2d%% (%2d/%2d)' % (phase, accuracy, correct, total))
    logger.add_line('\n %s Obj Recall: %2d%% (%2d/%2d)' % (phase, obj_accuracy, correct_obj_points, total_obj_points))
    logger.add_line('\n %s Obj Precision: %2d%% (%2d/%2d)' % (phase, precision, correct_obj_points_predictions, total_obj_points_predictions))
    logger.add_line(f'{phase} mIoU: {m_IoU}, IoU per class: {IoU}')


    # Sync metrics across all GPUs and print final averages
    if args.multiprocessing_distributed:
        progress.synchronize_meters(args.local_rank)
        progress.display(len(loader)*args.world_size)

    metrics_dict = {'epoch': epoch, 'phase': phase}
    if tb_writter is not None:
        for meter in progress.meters:
            tb_writter.add_scalar('{}-epoch/{}'.format(phase, meter.name), meter.avg, epoch)
            metrics_dict[meter.name + f'--{phase}'] = meter.avg

    wandb_utils.log(cfg, args, metrics_dict, epoch) ### TODO: summary? how to log for train and val separatly?

    return accuracy, obj_accuracy


if __name__ == '__main__':
    main()