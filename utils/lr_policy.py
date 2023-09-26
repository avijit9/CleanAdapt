
"""Learning rate policy."""

import math


def get_lr_at_epoch(args, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func('cosine')(args, cur_epoch)
    # Perform warm up.
    if cur_epoch < args.warmup_epochs:
        lr_start = args.warmup_start_lr
        lr_end = get_lr_func('cosine')(
            args, args.warmup_epochs
        )
        alpha = (lr_end - lr_start) / args.warmup_epochs
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(args, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return (
        args.base_lr
        * (math.cos(math.pi * cur_epoch / args.num_epochs) + 1.0)
        * 0.5
    )



def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]