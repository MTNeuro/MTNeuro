import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from MTNeuro.self_supervised.utils import MetricLogger

def compute_class_accuracy(preds, labels, cls):
    mask = torch.zeros(preds.shape).cuda()
    mask[labels == cls] = 1
    n_mask = mask.sum().item()
    cor = preds==labels
    cor = cor * mask
    corrects = cor.sum().item()
    return corrects, n_mask

def compute_class_accuracy2(preds, labels, cls):
    p = torch.ones(preds.shape)*4
    l = torch.ones(preds.shape)*4
    p[preds.cpu()==cls] = cls
    l[labels.cpu()==cls] = cls
    corrects = p.eq(l.view_as(p)).sum().item()
    return corrects


def compute_avg_accuracy(encoder, classifier, loader, transform=None, device='cpu', num_classes=2):
    r"""Evaluates the classification accuracy when a :obj:`torch.data.DataLoader` is given.
​
    Args:
        encoder (torch.nn.Module): Frozen encoder.
        classifier (torch.nn.Module): Linear layer.
        loader (torch.data.DataLoader): Dataloader.
        transform (Callable, Optional): Transformation to use. Added for the purposes of
            normalization. (default: :obj:`None`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)
​
    Returns:
        float: Accuracy.
    """
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    correct_class = torch.zeros(num_classes)
    confusion = torch.zeros(num_classes,num_classes)
    labels = []
    outputs = []

    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            if transform is not None:
                x, label = transform(x, label = label)
            # print(label)
            bs = x.size(0)//4
            label = label.view(bs, 4)
            # print(label)
            label = label[:,0]
            # print('after',label)
            representation = encoder(x.float())
            pred_logits = classifier(representation)
            pred_logits = pred_logits.view(bs, 4, num_classes).sum(1)
            # print(label.size(), pred_logits.size())
            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            for i in range(num_classes):
                correct_class[i] += compute_class_accuracy2(pred_class, label, i)
            correct += pred_class.eq(label.view_as(pred_class)).sum().item()
            total += len(label)
            confusion += confusion_matrix(label.cpu(), pred_class.cpu(), labels=list(range(num_classes)))
            labels.append(label.cpu())
            outputs.append(pred_class.cpu())
        outputs = np.concatenate(outputs)
        labels = np.concatenate(labels)
        f1 = f1_score(labels, outputs,  labels=list(range(num_classes)), average=None)
    return correct/total, correct_class/total, confusion, f1


def compute_accuracy(encoder, classifier, loader, transform=None, device='cpu', num_classes=2):
    r"""Evaluates the classification accuracy when a :obj:`torch.data.DataLoader` is given.

    Args:
        encoder (torch.nn.Module): Frozen encoder.
        classifier (torch.nn.Module): Linear layer.
        loader (torch.data.DataLoader): Dataloader.
        transform (Callable, Optional): Transformation to use. Added for the purposes of
            normalization. (default: :obj:`None`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        float: Accuracy.
    """
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0
    n_class  = torch.zeros(num_classes)
    correct_class = torch.zeros(num_classes)
    confusion = torch.zeros(num_classes,num_classes)
    labels = []
    outputs = []
    
    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            label = torch.squeeze(label, 1).long()
            if transform is not None:
                x, label = transform(x, label = label)

            representation = encoder(x.float())
            pred_logits = classifier(representation)

            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            for i in range(num_classes):
                corrects, n_mask =  compute_class_accuracy(pred_class, label, i)
                correct_class[i] += corrects
                n_class[i]+=n_mask

            correct += pred_class.eq(label.view_as(pred_class)).sum().item()
            total += len(label)
            
            confusion += confusion_matrix(label.cpu(), pred_class.cpu(), labels=list(range(num_classes)))
            labels.append(label.cpu())
            outputs.append(pred_class.cpu())
            
        outputs = np.concatenate(outputs)
        labels = np.concatenate(labels)
        f1 = f1_score(labels, outputs,  labels=list(range(num_classes)), average=None)
        
    return correct/total, correct_class/n_class, confusion, f1

def downsample_excitatory(pred_logits, label, p=0.5):
    mask0 = (label == 0)
    mask1 = (label != 0)
    label0 = label[mask0]
    label1 = label[mask1]
    pred_logits0 = pred_logits[mask0]
    pred_logits1 = pred_logits[mask1]
    perm = torch.randperm(label1.size(0))
    idx = perm[:int(p*label1.size(0))]
    pred_logits = torch.cat([pred_logits0, pred_logits1[idx]])
    label = torch.cat([label0, label1[idx]])
    return pred_logits, label
    
def train_classifier(encoder, classifier, train_loader, test_loader, optimizer, num_epochs, scheduler=None,
                     train_transform=None, test_transform=None, device='cpu', writer=None, tag='',
                     tqdm_progress=False, smoothing_factor=0.5, plot_path = None, criterion = None, num_classes = 2):
    # todo figure out tensor dataloader
    r"""Trains linear layer to predict angle.

    Args:
        encoder (torch.nn.Module): Frozen encoder.
        classifier (torch.nn.Module): Trainable linear layer.
        train_loader (torch.data.DataLoader or list of torch.nn.Tensor): Inputs and target class.
        test_loader (torch.data.DataLoader or list of torch.nn.Tensor): Inputs and target class.
        optimizer (torch.optim.Optimizer): Optimizer for :obj:`classifier`.
        scheduler (torch.optim._LRScheduler, Optional): Learning rate scheduler. (default: :obj:`None`)
        train_transform (Callable, Optional): Transformation to use during training. (default: :obj:`None`)
        test_transform (Callable, Optional): Transformation to use during validation. Added for the purposes of
            normalization. (default: :obj:`None`)
        num_epochs (int, Optional): Number of training epochs. (default: :obj:`10`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)
        writer (torch.utils.tensorboard.SummaryWriter, Optional): Summary writer. (default: :obj:`None`)
        tag (String, Optional): Tag used in :obj:`writer`. (default: :obj:`""`)
        tqdm_progress (bool, Optional): If :obj:`True`, show training progress.
        smoothing_factor:

    Returns:
        MetricLogger: Accuracy.
    """
    encoder.eval()  # just in case this was not done before
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    acc = MetricLogger(moving_average=True, smoothing_factor=smoothing_factor)
    for epoch in tqdm(range(num_epochs), disable=not tqdm_progress):
        classifier.train()
        for x, label in train_loader:
            # load data
            x, label = x.to(device), label.to(device)
            label = torch.squeeze(label, 1).long()
            if train_transform is not None:
                x, label = train_transform(x, label=label)

            # forward
            optimizer.zero_grad()
            with torch.no_grad():
                representation = encoder(x).detach()
            pred_logits = classifier(representation)
            #pred_logits, label = downsample_excitatory(pred_logits, label)

            # loss and backward
            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # compute classification accuracy
        acc_value, acc_class_value, conf, f1 = compute_avg_accuracy(encoder, classifier, test_loader, transform=test_transform, device=device, num_classes=num_classes)

        # log
        acc.update(train=None, test=acc_value)
        if writer is not None:
            writer.add_scalar('eval_debug/test-%r' % tag, acc_value, epoch)
            #for i in range(2):
            #    writer.add_scalar('eval_debug/test-class-'+str(i), acc_class_value[i], epoch)
                
            writer.add_scalar('eval_debug/f1score', np.mean(f1[0]), epoch)
    return acc_value, conf, f1
