"""
This step generates pseudo-labels based on the cleaning method
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import datetime
import os
import sys
import random
import json
from tqdm import tqdm
import pickle as pickle

import torch
import torch.nn.functional as F
from torchvision import transforms

from parse_args import create_parser
from models.model import SourceOnlyModel
from dataset.get_datasets import get_data, get_weak_transforms, get_dataloader, get_strong_transforms
from utils.utils import set_seed, AverageMeter, accuracy
import dataset.transforms as T


def main(args):

    # Check if GPU is available (CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if args.seed != -1:
        set_seed(args.seed)

    # Save and log directory creation
    result_dir = os.path.join(args.save_dir, '_'.join(
        (args.source_dataset, args.target_dataset, args.modality)))

    os.makedirs(result_dir, exist_ok=True)

    # get the target training dataset (we use the test time transforms for PL generation)
    transform_train = get_weak_transforms(args, 'val')
    print("==> Constructing the target dataloaders..")
    target_train_dataset = get_data(transform_train, args, 'generate-pseudo-label', args.target_dataset)
    target_train_loader = get_dataloader(args, 'generate-pseudo-label', target_train_dataset)

    print("==> Loading the I3D backbone")
    
    model = SourceOnlyModel(args).to(device)
    
    if args.pretrained_weight_path is not None:
        print("==> Loading pretrained weights from {}".format(args.pretrained_weight_path))
        checkpoint = torch.load(args.pretrained_weight_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        print("==> Please give a valid checkpoint..")
        sys.exit(0)

    # print("AdaBN phase..")
    # model.train()

    # # with torch.no_grad():
    # for batch_idx, batch in enumerate(target_train_loader):
    #     _, _, _, _ = model.no_clean(batch, transform_normalize, args)

    # print("AdaBN phase done!")

    print("Label generation phase begins..")

    model.eval()

    if args.target_dataset in ["D1", "D2", "D3"]:
        full_split_file_path = os.path.join(args.split_path, "{}_{}.pkl".format(args.target_dataset, 'train'))
        
        with open(full_split_file_path, 'rb') as f:
            pseudo_label_dict = pickle.load(f)

        pseudo_label_dict['rgb_pseudo_labels'] = 0
        pseudo_label_dict['flow_pseudo_labels'] = 0
    else:
        pseudo_label_dict = {}

    top1_acc = AverageMeter("Top1")
    

    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(target_train_loader)):
            
            seq, targets, idx = batch
            if args.modality == 'Joint':
                seq, targets = [seq[0].to(device), seq[1].to(device)], targets.to(device)
            else:
                seq, targets = seq.to(device), targets.to(device)
            
            logits, _ = model(seq, args)

            if args.modality == 'Joint':
                pred_logits = (logits[0] + logits[1]) / 2
                # rgb_pred_logits = logits[0]
                # flow_pred_logits = logits[1]
            else:
                pred_logits = logits[0]
            
            prediction = torch.argmax(pred_logits, dim = 1)
            # flow_prediction = torch.argmax(flow_pred_logits, dim = 1)
            
            if args.target_dataset in ["D1", "D2", "D3"]:
                # import pdb; pdb.set_trace()
                pseudo_label_dict.loc[int(idx[0]), 'pseudo_labels'] = prediction.item()
            else:
                pseudo_label_dict[batch[-1][0]] = int(prediction.item())
            
            top1_acc.update((prediction == targets).float().mean())
            # flow_top1_acc.update((flow_prediction == targets).float().mean())

    # from sklearn.metrics import confusion_matrix
    # matrix = confusion_matrix(pseudo_label_dict['verb_class'].tolist(), pseudo_label_dict['rgb_pseudo_labels'].tolist())
    # print("=> RGB")
    # print(matrix.diagonal()/matrix.sum(axis=1)*100)  
    # matrix = confusion_matrix(pseudo_label_dict['verb_class'].tolist(), pseudo_label_dict['flow_pseudo_labels'].tolist())
    # print("=> Flow")
    # print(matrix.diagonal()/matrix.sum(axis=1)*100)  

    # print((pseudo_label_dict['pseudo_labels'] == pseudo_label_dict['verb_class']).sum() / len(pseudo_label_dict['verb_class']))
    if args.target_dataset in ["D1", "D2", "D3"]:
        pl_file_name = os.path.join(result_dir, 'pseudo_annotations.pkl')
        pseudo_label_dict.to_pickle(pl_file_name)
    else:
        pl_file_name = os.path.join(result_dir, 'pseudo_annotations.json')
        json.dump(pseudo_label_dict, open(pl_file_name, 'w'))
    print("==> pseudo-label annotations are saved to: {} w/ accuracy {} w/ {} samples"
    .format(pl_file_name, top1_acc.avg * 100, top1_acc.count))
    # print("==> [RGB] pseudo-label annotations are saved to: {} w/ accuracy {} w/ {} samples"
    # .format(pl_file_name, rgb_top1_acc.avg * 100, rgb_top1_acc.count))
    # print("==> [Flow] pseudo-label annotations are saved to: {} w/ accuracy {} w/ {} samples"
    # .format(pl_file_name, flow_top1_acc.avg * 100, flow_top1_acc.count))

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    main(args)
