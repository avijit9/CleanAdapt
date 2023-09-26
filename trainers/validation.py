
import torch
import torch.nn.functional as F


from utils.utils import AverageMeter, accuracy

def validate(loader, model, epoch, args, device):
    
    model.eval()
    

    correct = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            
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
                

            outputs = F.softmax(pred_logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (pred.cpu() == targets.cpu()).sum()
            loss += F.cross_entropy(pred_logits, targets) * targets.size(0)
        acc1 = 100 * float(correct) / float(total)
        loss = loss / float(total)        
    return acc1, loss

