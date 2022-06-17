import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt

from MTNeuro.self_supervised.utils import MetricLogger


def construct_dist_matrix(X, Y, numclasses):
    dmat = np.zeros((numclasses, numclasses))
    for i in range(numclasses):
        g_i = X[Y == i]
        for j in range(i, numclasses):
            g_j = X[Y == j]
            d = CSM_fast(g_i, g_j)
            dmat[i,j] = d
            dmat[j,i] = d
    return dmat

def CSM_fast(g1,g2):
    m1 = np.mean(g1, axis=0)
    m2 = np.mean(g2, axis=0)
    m = np.mean(np.concatenate((g1,g2), axis=0), axis=0)
    S_b = np.sum((m1-m)**2 + (m2-m)**2)
    S_w = 0 
    for i in range(g1.shape[0]):
        S_w = S_w + np.sum((g1[i] - m1)**2)
    for i in range(g2.shape[0]):
        S_w = S_w + np.sum((g2[i] - m2)**2)
    
    return S_b/S_w

def compute_class_accuracy(preds, labels, cls):
    # p = torch.ones(preds.shape)*2
    # l = torch.ones(preds.shape)*2
    mask = torch.zeros(preds.shape).cuda()
    mask[labels == cls] = 1
    n_mask = mask.sum().item()
    cor = preds==labels
    cor = cor * mask
    corrects = cor.sum().item()
    # p[preds.cpu()==cls] = cls
    # l[labels.cpu()==cls] = cls
    # corrects = p.eq(l.view_as(p)).sum().item()
    return corrects, n_mask
    
def downsample_excitatory(pred_logits, label, p=0.35):
    mask0 = (label == 0)
    mask1 = (label != 0)
    label0 = label[mask0]
    label1 = label[mask1]
    pred_logits0 = pred_logits[mask0]
    pred_logits1 = pred_logits[mask1]
    pred_logits = torch.cat([pred_logits0, pred_logits1[:int(p*label1.size(0))]])
    label = torch.cat([label0, label1[:int(p*label1.size(0))]])
    return pred_logits, label


def compute_accuracy(classifier, loader, transform=None, device='cpu', num_classes = 2):
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
    classifier.eval()

    correct = 0
    total = 0
    correct_class = torch.zeros(num_classes)
    confusion = torch.zeros(num_classes,num_classes)
    n_class  = torch.zeros(num_classes)
    labels = []
    outputs = []
    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            label = torch.squeeze(label, 1).long().to(device)
            if transform is not None:
                x, label = transform(x, label = label)

            pred_logits = classifier(x.float())
            #pred_logits, label = downsample_excitatory(pred_logits, label)
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

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i],2), ha = 'center')


def generate_plots(acc, acc_class, plot_path, loader, device='cpu'):

    #Bar plot
    # labels = ['cortex', 'striatum', 'VP', 'ZI', 'full']
    labels = [' ', '0', '1']
    # positions = [0,1,2,3,4]
    positions = [0,1,2]
    values = np.append(acc_class.numpy(), acc)
    plt.ylim(0,1)
    plt.bar(positions, values)
    addlabels(positions, values)
    plt.xticks(positions, labels)
    plt.savefig(plot_path + 'bar.png')
    

def train_full_classifier_synapse(classifier, train_loader, test_loader, optimizer, num_epochs, scheduler=None,
                     train_transform=None, test_transform=None, device='cpu', writer=None, tag='',
                     tqdm_progress=False, smoothing_factor=0.5, plot_path= None, criterion = None, num_classes = 2):
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
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    acc = MetricLogger(moving_average=True, smoothing_factor=smoothing_factor)
    max_acc = 0.0
    for epoch in tqdm(range(num_epochs), disable=not tqdm_progress):
        classifier.train()
        for x, label in train_loader:
            # load data
            x, label = x.to(device), label.to(device)
            label = torch.squeeze(label, 1).long()
            if train_transform is not None:
                x, label = train_transform(x, label = label)

            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)
            # print(pred_logits.size())
            # loss and backward
            loss = criterion(pred_logits, label)
            # print(loss)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # compute classification accuracy
        acc_value, acc_class_value, conf, f1 = compute_accuracy(classifier, test_loader, transform=test_transform, device=device, num_classes = num_classes)
        if acc_value > max_acc:
            max_acc = acc_value
        # log
        acc.update(train=None, test=acc_value)
        if writer is not None:
            writer.add_scalar('superv_train_debug/test-%r' % tag, acc_value, epoch)
            writer.add_scalar('superv_train_debug/test-max_acc%r' % tag, max_acc, epoch)
            writer.add_scalar('superv_train_debug/f1score%r' % tag, np.mean(f1), epoch)
            for i in range(2):
                writer.add_scalar('superv_train_debug_classes/test-class-'+str(i), acc_class_value[i], epoch)
            #writer.add_scalar('superv_train_debug/f1score-c0', f1[0], epoch)
            #writer.add_scalar('superv_train_debug/f1score-c1', f1[1], epoch)
    if plot_path is not None:
        generate_plots(acc_value, acc_class_value, plot_path, test_loader, device=device)
        
        # labels = [' ', 'cortex',' ', 'striatum',' ', 'VP',' ', 'ZI']
        labels = [' ', '0', '1']
        fig, ax = plt.subplots()
        im = ax.imshow(conf/240, vmin = 0, vmax = 1, cmap=plt.get_cmap('Blues'))
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        fig.colorbar(im)
        fig.savefig(plot_path + 'conf.png')
        
    return acc_value, conf, f1
