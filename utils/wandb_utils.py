import os

import wandb


def is_wandb_enabled(cfg, args):
    wandb_enabled = False
    if cfg['log2wb'] and args.rank == 0:
        wandb_enabled = True
        assert os.environ.get('WANDB_API_KEY') is not None
    # if os.environ.get('WANDB_API_KEY') is None:
    #     wandb_enabled = False
    return wandb_enabled


def init(cfg, args, job_type='train'):
    """
        Initialize wandb by passing in config
    """
    if not is_wandb_enabled(cfg, args):
        return False

    wandb.init(name=cfg['tag'],
               config=cfg,
               project=cfg['WANDB']['PROJECT'],
               entity=cfg['WANDB']['ENTITY'],
               job_type=job_type,
               dir='/DepthContrast/wandb')
    return True


def log(cfg, args, log_dict, step):
    if not is_wandb_enabled(cfg, args):
        return False

    assert isinstance(log_dict, dict)
    assert isinstance(step, int)

    wandb.log(log_dict, step)
    return True


# def summary(cfg, args, log_dict, step, highest_metric=-1):
#     """
#     Wandb summary information
#     Args:
#         cfg
#     """

#     if not is_wandb_enabled(cfg, args):
#         return

#     assert isinstance(log_dict, dict)
#     assert isinstance(step, int)

#     metric = log_dict.get(cfg.WANDB.get('SUMMARY_HIGHEST_METRIC'))
#     if metric is not None and metric > highest_metric:
#         # wandb overwrites summary with last epoch run. Append '_best' to keep highest metric
#         for key, value in log_dict.items():
#             wandb.run.summary[key + '_best'] = value
#         wandb.run.summary['epoch'] = step
#         highest_metric = metric

#     return highest_metric


# def log_and_summary(cfg, args, log_dict, step, highest_metric=-1):
#     log(cfg, log_dict, step)
#     return summary(cfg, log_dict, step, highest_metric)