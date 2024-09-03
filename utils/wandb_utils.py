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


def init(cfg, args, run_file, pretraining=True):
    """
        Initialize wandb by passing in config
    """
    if not is_wandb_enabled(cfg, args):
        return False

    dir = cfg['WANDB'].get('dir', None)

    run_name = cfg['model']['name']
    if pretraining:
        # pretraining
        job_type = 'pretrain'
        group_name = cfg['model']['extra_tag']
    elif 'scratch' in run_name:
        # scratch training
        job_type = cfg['model']['job_type']
        group_name = cfg['model']['extra_tag']
    else:
        #finetuning
        job_type = cfg['model']['job_type']
        group_name = 'pretrain_{}_finetine_{}'.format(cfg['model']['pretrain_extra_tag'], cfg['model']['extra_tag'])

    if run_file.exists() and not args.wandb_dont_resume:
        resume_id = run_file.read_text()
        wandb.init(name=run_name, config=cfg,
                            project=cfg['WANDB']['PROJECT'],
                            entity=cfg['WANDB']['ENTITY'],
                            group=group_name,
                            job_type=job_type,
                            id=resume_id, resume='must', dir=dir)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        run = wandb.init(name=run_name, config=cfg,
                            project=cfg['WANDB']['PROJECT'],
                            entity=cfg['WANDB']['ENTITY'],
                            group=group_name,
                            job_type=job_type,
                            dir=dir)
        run_file.write_text(str(run.id))
    return True

def reinit(cfg, args, job_type='train'):
    """
        Initialize wandb by passing in config
    """
    if not is_wandb_enabled(cfg, args):
        return False, None

    dir = cfg['WANDB'].get('dir', None)
    
    run_name = cfg['model']['name']
    if job_type != 'train':
        ckpt_name = cfg['pretrain_checkpoint'].split('.')[0]
        run_name += '-' + job_type + '-'+ckpt_name

    run = wandb.init(name=run_name,
               config=cfg,
               tags=cfg['wb_extra_tags'],
               project=cfg['WANDB']['PROJECT'],
               entity=cfg['WANDB']['ENTITY'],
               job_type=job_type,
               dir= dir,
               reinit=True)
    return True, run



def log(cfg, args, log_dict, step):
    if not is_wandb_enabled(cfg, args):
        return False

    assert isinstance(log_dict, dict)
    assert isinstance(step, int)
    wandb.log(log_dict, step)
    return True


def summary(cfg, args, log_dict, step, highest_metric=-1):
    """
    Wandb summary information
    Args:
        cfg
    """

    if not is_wandb_enabled(cfg, args):
        return

    assert isinstance(log_dict, dict)
    #assert isinstance(step, int)

    metric = log_dict.get(cfg['WANDB'].get('SUMMARY_HIGHEST_METRIC'))
    if metric is not None and metric > highest_metric:
        # wandb overwrites summary with last epoch run. Append '_best' to keep highest metric
        for key, value in log_dict.items():
            wandb.run.summary[key + '_best'] = value
        #wandb.run.summary['epoch'] = step
        highest_metric = metric

    return highest_metric


def log_and_summary(cfg, args, log_dict, step, highest_metric=-1):
    log(cfg, args, log_dict, step)
    return summary(cfg, args, log_dict, step, highest_metric)