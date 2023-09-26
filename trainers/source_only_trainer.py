import time

import torch

from utils.utils import AverageMeter, ProgressMeter,\
    accuracy

from utils import optimizer as optim

def train_one_epoch(data_loader, model, criterion, optimizer, epoch, args, device):

    '''
    executes one epoch of training on the train data
    '''

    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':1.2f')
    ce_loss = AverageMeter('CE', ':1.2f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, acc_cls, ce_loss],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()

    end = time.time()

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(data_loader):
    
        
        data_time.update(time.time() - end)

        seq, targets, _ = batch
        if args.modality == 'Joint':
            seq, targets = [seq[0].to(device), seq[1].to(device)], targets.to(device)
        else:
            seq, targets = seq.to(device), targets.to(device)
            

        logits, _ = model(seq, args)
        if args.modality == 'Joint':
            pred_logits = (logits[0] + logits[1]) / 2
        else:
            pred_logits = logits[0]

        loss = criterion(pred_logits, targets)

        cls_out = torch.argmax(pred_logits, dim = 1)

        acc = accuracy(pred_logits, targets)[0]
        acc_cls.update(acc[0], targets.size(0))

        ce_loss.update(loss, targets.size(0))


        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

    return acc_cls.avg, ce_loss.avg,









