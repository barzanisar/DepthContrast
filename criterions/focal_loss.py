from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from torch.autograd import Variable

'''
Ref: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
'''

# def one_hot_embedding(labels, num_classes):
#     '''Embedding labels to one-hot form.
#     Args:
#       labels: (LongTensor) class labels, sized [N,].
#       num_classes: (int) number of classes.
#     Returns:
#       (tensor) encoded labels, sized [N,#classes].
#     '''
#     y = torch.eye(num_classes)  # [D,D]
#     return y[labels]            # [N,D]

# class WeightedCrossEntropyLoss(nn.Module):
#     def __init__(self, config):
#         super(WeightedCrossEntropyLoss, self).__init__()
#         self.num_classes = config['NUM_CLASSES']
    
#     def forward(self, cls_pred, cls_targets):
#         batch_size = cls_targets.shape[0]
#         num_points_per_batch = cls_targets.shape[1]
#         N = batch_size * num_points_per_batch
#         num_samples_per_class = torch.zeros(self.num_classes+1)
#         for i in range(self.num_classes+1):
#             mask = cls_targets == i
#             num_samples_per_class[i] = mask.sum()
                
#         class_freq = num_samples_per_class / N
#         power = 1
#         eps = 1e-4
#         weights = 1/(class_freq + eps)**power
#         #weights = weights / torch.sum(weights)

#         loss = torch.nn.CrossEntropyLoss(weight = weights)
#         loss.cuda()
#         return loss(cls_pred.transpose(1,2), cls_targets)

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self, config):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if config['reduction'] not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.num_classes = config['num_classes'] + 1
        self.alpha = None if config['alpha'] == 'None' else config['alpha']
        self.gamma = config['gamma']
        self.ignore_index = config['ignore_index']
        self.reduction = config['reduction']
        if self.alpha is not None:
            self.alpha = torch.tensor(self.alpha).float().cuda()
            self.nll_loss = nn.NLLLoss(
                weight=self.alpha, reduction='none', ignore_index=self.ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        if self.alpha is None:
            N = y.shape[0]
            num_samples_per_class = torch.zeros(self.num_classes)
            for i in range(self.num_classes):
                mask = y == i
                num_samples_per_class[i] = mask.sum()
                    
            class_freq = num_samples_per_class / N
            # Method 1:
            power = 1
            #eps = 1e-4
            alpha = torch.zeros(self.num_classes)
            for i in range(self.num_classes):
                if num_samples_per_class[i] > 0:
                    alpha[i] = 1/(class_freq[i])**power
                    #alpha[i] = 1/(num_samples_per_class[i])**power
            
            # alpha = alpha/alpha.sum() * self.num_classes
            # Method 2:
            # alpha = 1 - class_freq #stable increase but only 63% max val acc
            # Method 3: 
            # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
            # https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
            # beta = 0.9999 # 0.99, 0.999, 0.9999
            # effective_num = 1.0 - np.power(beta, num_samples_per_class.numpy())
            # weights = (1.0 - beta) / np.array(effective_num)
            # weights = weights / np.sum(weights) * self.num_classes
            # alpha = torch.tensor(weights)

            #print('Calc alphas: ', alpha)
            self.nll_loss = nn.NLLLoss(
                weight=alpha.cuda(), reduction='none', ignore_index=self.ignore_index)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss