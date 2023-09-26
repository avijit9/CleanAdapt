import numpy as np
import datetime
import os
import sys
import random
import pickle
import builtins
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import dataset.transforms as T

import warnings
warnings.filterwarnings("ignore")

from parse_args import create_parser
from models.model import SourceOnlyModel
from dataset.get_datasets import get_data, get_weak_transforms, get_strong_transforms, get_dataloader
from utils.utils import set_seed, adjust_learning_rate, save_checkpoint
from trainers import source_only_trainer, validation

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed != -1:
        set_seed(args.seed)

    best_target_acc = 0.
    best_source_acc = 0. 

    print("\n############################################################################\n")
    print("Experimental Configs: ", args)
    print("\n############################################################################\n")

    print("==> Using Domain Adaptation Mode: {} [{}]".format(args.adaptation_mode, args.modality))

    # Save and log directory creation
    result_dir = os.path.join(args.save_dir, '_'.join(
        (args.source_dataset, args.target_dataset, args.adaptation_mode, args.modality)))
    log_dir = os.path.join(result_dir, 'logs')
    save_dir = os.path.join(result_dir, 'checkpoints')

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(save_dir,   exist_ok=True)

    run_name = "-".join(
        [args.adaptation_mode, args.source_dataset, args.target_dataset]
    )
    wandb.login()
    run = wandb.init(
        project = "video-domain-adaptation",
        config = args,
        dir = log_dir,
        entity = 'avijit9',
        name = run_name
    )

    # Dataloader creation
    weak_transform_train = get_weak_transforms(args, 'train')
    strong_transform_train = get_strong_transforms(args, 'train')
    transform_val = get_weak_transforms(args, 'val')
    
    print("==> Constructing the source dataloaders..")
    source_train_dataset = get_data([weak_transform_train, strong_transform_train], args, 'train', args.source_dataset)
    source_val_dataset = get_data(transform_val, args, 'val', args.source_dataset)
    source_train_loader = get_dataloader(args, 'train', source_train_dataset)
    source_val_loader = get_dataloader(args, 'val', source_val_dataset)
    
    print("==> Constructing the target dataloaders..")
    target_val_dataset = get_data(transform_val, args, 'val', args.target_dataset)
    target_val_loader = get_dataloader(args, 'val', target_val_dataset)  

    # Create the model
    print("==> Loading the I3D backbone")
    model = SourceOnlyModel(args)            
    model = torch.nn.parallel.DataParallel(model, device_ids = list(range(args.gpus))).to(device)
    

    # define the loss function and optimizers here.
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    # define the optimizer
    optimizer = optim.SGD(model.parameters(), args.lr,
    weight_decay = args.weight_decay, momentum = args.momentum)

    # start training
    for epoch in range(0, args.num_epochs):
        
        adjust_learning_rate(optimizer, epoch, args)
        
        train_epoch_acc, train_epoch_loss = source_only_trainer.train_one_epoch(source_train_loader, \
            model, criterion, optimizer, epoch, args, device)
        
        if run is not None:
            run.log({"source/loss_supervised": train_epoch_loss, "epoch": epoch})
            run.log({"source/accuracy": train_epoch_acc, "epoch": epoch})

        source_val_epoch_acc, source_val_epoch_loss = validation.validate(source_val_loader, model, \
            epoch, args, device)
        
        target_val_epoch_acc, target_val_epoch_loss = validation.validate(target_val_loader, model, \
            epoch, args, device)

        if run is not None:
            run.log({"source/val_accuracy": source_val_epoch_acc, "epoch": epoch})
            run.log({"source/val_loss": source_val_epoch_loss, "epoch": epoch})
            run.log({"target/val_accuracy": target_val_epoch_acc, "epoch": epoch})
            run.log({"target/val_loss": target_val_epoch_loss, "epoch": epoch})        

        if source_val_epoch_acc > best_source_acc:
            best_source_acc = source_val_epoch_acc
            best_target_acc = target_val_epoch_acc
            print("Found source best acc {} at epoch {}.".format(source_val_epoch_acc, epoch))
            print("Found target acc {} at epoch {}.".format(target_val_epoch_acc, epoch))
            
            is_source_best = True
        else:
            is_source_best = False

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'i3d',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_target_val_acc': best_target_acc,
        }, is_source_best, checkpoint_dir = save_dir, epoch = epoch + 1)



    print("==> Training done!")
    print("==> [Source] Best accuracy {}".format(best_source_acc))
    print("==> [Target] Best accuracy {}".format(best_target_acc))
    

if __name__ == "__main__":

    parser = create_parser()
    
    args = parser.parse_args()
    main(args)
