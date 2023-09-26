import argparse
import yaml

def create_parser():

    
    parser = argparse.ArgumentParser()

    # Configs related to the dataset and adaptation mode (most important ones)
    parser.add_argument('--source_dataset',    type=str, help='Name of the source dataset')
    parser.add_argument('--target_dataset',    type=str, help='Name of the target dataset')
    parser.add_argument('--adaptation_mode',   type=str, help='Name of the adaptation types',
    choices = ['source_only', 'SLT'])
    parser.add_argument('--modality',   type=str, help='Name of the adaptation types',
    choices = ['RGB', 'Flow', 'Joint'])

    # Configs related to the model training
    parser.add_argument('--batch_size',        type=int, default = 1, help='Numbers of videos in a mini-batch')
    parser.add_argument('--num_workers',       type=int, default = 4, help='Number of subprocesses for dataloading')
    parser.add_argument('--num_classes',       type=int, default = 12, help='Number of total classes in the dataset')
    parser.add_argument('--opt',               type=str, default = 'sgd',  help='Name of optimizer')
    parser.add_argument('--lr',                type=float, default = 0.01, help='Learning rate')
    parser.add_argument('--momentum',          type=float, default = 0.9, help='Momentum value in optimizer')
    parser.add_argument('--weight_decay',      type=float, default = 1e-4, help='Weight decay')
    parser.add_argument('--num_epochs',             type=int,   default = 1, help='Total number of epochs')
    parser.add_argument('--seed',              type=int, default = 9, help='seed for deterministic results')
    parser.add_argument('--print_freq',              type=int, default = 10, help='print frequency')
    parser.add_argument('--pretrained',        type=str, default = None, help='pretrained-weights path')
    parser.add_argument('--ckpt_path',        type=str, default = None, help='checkpoint path')
    parser.add_argument('--milestones',   type=int,   nargs='+',  help='Epoch values to change learning rate')
    parser.add_argument('--dropout',                type=float, default = 0.5, help='Learning rate')
    parser.add_argument('--clip_length',  type=int, default = 16, help='Number of frames within a clip')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--gamma',   type=float, default = 0.1, help='gamma factor for decreasing the learning rate') 
    parser.add_argument('--sampling_rate',  type=int, default = 1, help='frame sampling rate')
    # Data and log directories
    parser.add_argument('--data_path',    type=str, help='Path to train and test split files')
    parser.add_argument('--split_path',   type=str, help='Path to train and test split files')
    parser.add_argument('--save_dir',     type=str, help='Path to results directory')
    parser.add_argument('--comment',     type=str, default = "", help='Path to results directory')
    parser.add_argument('--gpus',   type=int, default = 1, help='number of gpus to be used')
    

    # Pseudo-labeling related arguments
    parser.add_argument('--pretrained_weight_path', type=str, default = None, help='pretrained path for the model')
    parser.add_argument('--source_pretrained_weight_path', type=str, default = None, help='pretrained path for the model')
    parser.add_argument('--adapted_pretrained_weight_path', type=str, default = None, help='pretrained path for the model')
    parser.add_argument('--pseudo_label_path', type = str, default = None, help = "path to the pseudo-labels generated")
    parser.add_argument('--r', type=float, default = 1., help='split ratio to select the noisy and clean samples')
    parser.add_argument('--tau', type=float, default = 0.9, help='confidence threshold for co-training')
    parser.add_argument('--use-ema', action='store_true', default=False, help='use EMA model')
    parser.add_argument('--ema-decay', type=float, default = 0.99, help='decay variable for EMA')

    # adding argument specific to the EPic-kitchen
    parser.add_argument('--base_lr', type=float, default = 0.1, help='Learning rate')
    parser.add_argument('--warmup_start_lr', type=float, default = 0.01, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int,   default = 34, help='Total number of epochs')

    return parser