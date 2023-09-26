import torch
import utils.lr_policy as lr_policy

def get_epoch_lr(cur_epoch, cfg):
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr