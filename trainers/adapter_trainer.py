import time

import torch
import torch.nn.functional as F

import numpy as np

from utils.utils import AverageMeter, ProgressMeter, accuracy, update_ema_variables
# from utils.losses import loss_jocor
from sklearn.mixture import GaussianMixture

@torch.no_grad()
def sample_selection_step_teacher_student(dataloader, model, args, device, r = 0.8):
    loss_dict = {}
    pseudo_dict = {}
    orig_target_dict = {}
    selected_sample_dict = {} # stores the flag for the selected samples 
    updated_pseudo_dict = {}


    for batch_idx, batch in enumerate(dataloader):

        weak_seq, strong_seq, pseudo_target, orig_target, _ = batch
        
        if args.modality == 'Joint':
            weak_seq, strong_seq, pseudo_target, orig_target = [weak_seq[0].to(device), weak_seq[1].to(device)], \
                [strong_seq[0].to(device), strong_seq[1].to(device)], \
                pseudo_target.to(device), orig_target.to(device)
                
        else:
            weak_seq, pseudo_target, orig_target = weak_seq.to(device), \
            pseudo_target.to(device), orig_target.to(device)

        logits, _ = model.ema(weak_seq, args)

        if args.modality == 'RGB' or args.modality == 'Flow':
            total_loss = F.cross_entropy(logits[0], pseudo_target, reduction = 'none')
            predictions = torch.argmax(logits[0], dim = 1)
        else:
            total_loss = F.cross_entropy(logits[0], pseudo_target, reduction = 'none') + \
                F.cross_entropy(logits[1], pseudo_target, reduction = 'none')
            predictions = torch.argmax(0.5 * (logits[0] + logits[1]), dim = 1)

        for i in range(total_loss.size(0)):
            loss_dict[batch[-1][i]] = total_loss[i].item()
            updated_pseudo_dict[batch[-1][i]] = predictions[i].item()
            pseudo_dict[batch[-1][i]] = pseudo_target[i].item()
            orig_target_dict[batch[-1][i]] = orig_target[i].item()

    for cls_idx in range(args.num_classes):

        # select the videos with a particulare cls_idx values
        cls_videos = [key for key, value in updated_pseudo_dict.items() if value == cls_idx]

        # get the loss values of the videos of a particular class
        loss_values = [loss_dict[video] for video in cls_videos]

        # define r% percentile samples having low loss values as clean samples
        # try:
        if len(loss_values) > 0:
            # import pdb; pdb.set_trace()
            # loss_values = (loss_values - np.min(loss_values)) / (np.max(loss_values) - np.min(loss_values))
            clean_samples = np.where(loss_values <= np.percentile(loss_values, r * 100))[0]
            for video in cls_videos:
                if cls_videos.index(video) in clean_samples:
                    selected_sample_dict[video] = 1
                else:
                    selected_sample_dict[video] = 0
            
        # except:
            # import pdb; pdb.set_trace()
        # clean_samples = np.where(loss_values > np.percentile(loss_values, r * 100))[0] # rebuttal experiment
        # pseudo_laebls = [pseudo_dict[video] for video in cls_videos]
        # orig_labels = [orig_target_dict[video] for video in cls_videos]
        # clean_samples = [idx for idx, (a, b) in enumerate(zip(pseudo_laebls, orig_labels)) if a == b]

        # for clean samples, update the dict
        

    
    # ask dataloader to load only clean samples, ignoring the others
    dataloader.dataset.selected_sample_dict = selected_sample_dict
    dataloader.dataset._update_video_list(select_all = False)
    dataloader.dataset._update_pseudo_labels(updated_pseudo_dict)
    
    # this is just to evaluate quality of the selected clean samples
    pseudo_target_list = []
    orig_target_list = []
    updated_pseudo_target_list = []
    for video in list(pseudo_dict.keys()):
        if selected_sample_dict[video] == 1:
            pseudo_target_list.append(pseudo_dict[video])
            updated_pseudo_target_list.append(updated_pseudo_dict[video])
            orig_target_list.append(orig_target_dict[video])    

    accuracy = sum(1 for x,y in zip(pseudo_target_list, orig_target_list) if x == y) / float(len(pseudo_target_list))
    updated_accuracy = sum(1 for x,y in zip(updated_pseudo_target_list, orig_target_list) if x == y) / float(len(updated_pseudo_target_list))
    if args.adaptation_mode == 'SLT':
        # print("[r = {}] SLT selected {} samples with {:.4f}% accuracy".format(r, len(pseudo_target_list), accuracy * 100))
        print("[r = {}] SLT selected {} samples with {:.4f}% accuracy new PLs".format(r, len(updated_pseudo_target_list), updated_accuracy * 100))    
    
@torch.no_grad()
def sample_selection_step(dataloader, model, args, device, r = 0.8):
    model.eval()
    loss_dict = {}
    pseudo_dict = {}
    orig_target_dict = {}
    selected_sample_dict = {} # stores the flag for the selected samples 
    
    for batch_idx, batch in enumerate(dataloader):

        weak_seq, strong_seq, pseudo_target, orig_target, _ = batch
        
        if args.modality == 'Joint':
            weak_seq, strong_seq, pseudo_target, orig_target = [weak_seq[0].to(device), weak_seq[1].to(device)], \
                [strong_seq[0].to(device), strong_seq[1].to(device)], \
                pseudo_target.to(device), orig_target.to(device)
                
        else:
            weak_seq, pseudo_target, orig_target = weak_seq.to(device), \
            pseudo_target.to(device), orig_target.to(device)

        logits, _ = model(weak_seq, args)

        if args.modality == 'RGB' or args.modality == 'Flow':
            total_loss = F.cross_entropy(logits[0], pseudo_target, reduction = 'none')
        else:
            total_loss = F.cross_entropy(logits[0], pseudo_target, reduction = 'none') + \
                F.cross_entropy(logits[1], pseudo_target, reduction = 'none')
            

        for i in range(total_loss.size(0)):
            loss_dict[batch[-1][i]] = total_loss[i].item()
            pseudo_dict[batch[-1][i]] = pseudo_target[i].item()
            orig_target_dict[batch[-1][i]] = orig_target[i].item()

    for cls_idx in range(args.num_classes):

        # select the videos with a particulare cls_idx values
        cls_videos = [key for key, value in pseudo_dict.items() if value == cls_idx]

        # get the loss values of the videos of a particular class
        loss_values = [loss_dict[video] for video in cls_videos]
       
        # define r% percentile samples having low loss values as clean samples
        clean_samples = np.where(loss_values <= np.percentile(loss_values, r * 100))[0]
        # clean_samples = np.where(loss_values > np.percentile(loss_values, r * 100))[0] # rebuttal experiment
        # pseudo_laebls = [pseudo_dict[video] for video in cls_videos]
        # orig_labels = [orig_target_dict[video] for video in cls_videos]
        # clean_samples = [idx for idx, (a, b) in enumerate(zip(pseudo_laebls, orig_labels)) if a == b]

        # for clean samples, update the dict
        for video in cls_videos:
            if cls_videos.index(video) in clean_samples:
                selected_sample_dict[video] = 1
            else:
                selected_sample_dict[video] = 0

    
    # ask dataloader to load only clean samples, ignoring the others
    dataloader.dataset.selected_sample_dict = selected_sample_dict
    dataloader.dataset._update_video_list(select_all = False)
    
    
    # this is just to evaluate quality of the selected clean samples
    pseudo_target_list = []
    orig_target_list = []
    updated_pseudo_target_list = []
    for video in list(pseudo_dict.keys()):
        if selected_sample_dict[video] == 1:
            pseudo_target_list.append(pseudo_dict[video])
            orig_target_list.append(orig_target_dict[video])    

    accuracy = sum(1 for x,y in zip(pseudo_target_list, orig_target_list) if x == y) / float(len(pseudo_target_list))
    
    if args.adaptation_mode == 'SLT':
        print("[r = {}] SLT selected {} samples with {:.4f}% accuracy".format(r, len(pseudo_target_list), accuracy * 100))
    
    


def train_one_epoch(data_loader, model, ema_model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    train_1_acc = AverageMeter('Acc-RGB@cls', ':1.2f')
    train_2_acc = AverageMeter('Acc-Flow@cls', ':1.2f') 
    train_1_loss = AverageMeter('Loss-RGB@cls', ':1.2f')
    train_2_loss = AverageMeter('Loss-Flow@cls', ':1.2f')
    
    progress = ProgressMeter(len(data_loader), [
        batch_time, data_time, train_1_acc, train_2_acc, train_1_loss, train_2_loss
    ], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    
    for batch_idx, batch in enumerate(data_loader):
        data_time.update(time.time() - end)

        weak_seq, strong_seq, pseudo_target, orig_target, _ = batch

        if args.modality == 'Joint':
            weak_seq, strong_seq, pseudo_target, orig_target = [weak_seq[0].to(device), weak_seq[1].to(device)], \
                [strong_seq[0].float().to(device), strong_seq[1].float().to(device)], \
                pseudo_target.to(device), orig_target.to(device)
        else:
            weak_seq, pseudo_target, orig_target = weak_seq.to(device), \
            pseudo_target.to(device), orig_target.to(device)

        if args.use_ema:
            logits, _ = model(strong_seq, args)
        else:
            logits, _ = model(weak_seq, args)

        if args.modality == 'RGB' or args.modality == 'Flow':
            prec_1 = accuracy(logits[0], pseudo_target)[0]
            train_1_acc.update(prec_1.item())
            loss = criterion(logits[0], pseudo_target).mean()
            train_1_loss.update(loss.mean().item())
        else:
            prec_1 = accuracy(logits[0], pseudo_target)[0]
            prec_2 = accuracy(logits[1], pseudo_target)[0]

            train_1_acc.update(prec_1.item())
            train_2_acc.update(prec_2.item())

            loss_1 = criterion(logits[0], pseudo_target)
            loss_2 = criterion(logits[1], pseudo_target)
            loss = loss_1.mean() + loss_2.mean()

            train_1_loss.update(loss_1.mean().item())
            train_2_loss.update(loss_2.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.use_ema:
            ema_model.update(model)
        # global_step = (epoch + 1) * (batch_idx + 1)
        # update_ema_variables(model, ema_model, args.ema_decay, global_step = global_step)

        # =======================
        # Booking part
        # ======================

        # clean_idx = (pseudo_target == orig_target)
        # noise_idx = (pseudo_target != orig_target)

        
        # clean_loss = loss_1[clean_idx].sum() / (clean_idx.sum() + 1e-9)
        # ce_clean_rgb.update(clean_loss.item())
        # clean_loss = loss_2[clean_idx].sum() / (clean_idx.sum() + 1e-9)
        # ce_clean_flow.update(clean_loss.item())
        
        # noisy_loss = loss_1[noise_idx].sum() / (noise_idx.sum() + 1e-9)
        # ce_noisy_rgb.update(noisy_loss.item())
        # noisy_loss = loss_2[noise_idx].sum() / (noise_idx.sum() + 1e-9)
        # ce_noisy_flow.update(noisy_loss.item())
        
        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

    if args.modality == 'RGB' or args.modality == 'Flow':
        return [train_1_acc.avg], [train_1_loss.avg]
    else:
        return [train_1_acc.avg, train_2_acc.avg], [train_1_loss.avg, train_2_loss.avg]

