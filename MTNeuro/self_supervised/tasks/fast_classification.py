import torch
from tqdm import tqdm

from MTNeuro.self_supervised.data import utils
from MTNeuro.self_supervised.utils import MetricLogger


def compute_representations(net, dataloader, device='cpu'):
    r"""Pre-computes the representation for the entire dataset.

    Args:
        net (torch.nn.Module): Frozen encoder.
        dataloader (torch.data.DataLoader): Dataloader.
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    for i, (x, label) in tqdm(enumerate(dataloader)):
        # load data
        x = x.to(device)#.squeeze()
        labels.append(label)

        # forward
        with torch.no_grad():
            representation = net(x)
            reps.append(representation.detach().cpu().squeeze())

        if i % 10 == 0:
            reps = [torch.cat(reps, dim=0)]
            labels = [torch.cat(labels, dim=0)]

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]


def compute_accuracy(classifier, data, batch_size, device='cpu'):
    r"""Evaluates the classification accuracy with representations pre-computed.

    Args:
        classifier (torch.nn.Module): Linear layer.
        data (list of torch.nn.Tensor): Inputs, target class and target angles.
        batch_size (int, Optional): Batch size used during evaluation. It has no impact on final accuracy.
            (default: :obj:`256`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        float: Accuracy.
    """
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, label in utils.batch_iter(*data, batch_size=batch_size):
            x, label = x.to(device), label.to(device)

            # feed to network and classifier
            pred_logits = classifier(x)
            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            correct += pred_class.eq(label.view_as(pred_class)).sum().item()
            total += len(label)
    return correct/total

def compute_top_k_accuracy(classifier, data, batch_size, topk=[1, 2], device='cpu'):
    classifier.eval()

    with torch.no_grad():
        x, label = data
        x, label = x.to(device), label.to(device)

        # feed to network and classifier
        pred_logits = classifier(x)

    pred_top = pred_logits.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = [(pred_top[:, :t] == label[..., None]).float().sum(1).mean().cpu().item() for t in topk]
    return acc

# num_epochs = 500
# batch_size = 1000
# optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
#     lr_start, lr_end = 1e-2, 1e-6
#     gamma = (lr_end / lr_start) ** (1 / num_epochs)
def train_linear_layer(classifier, train_data, test_data, optimizer, num_epochs, batch_size, scheduler=None,
                       device='cpu', writer=None, tag="", tqdm_progress=False, smoothing_factor=0.5):
    criterion = torch.nn.CrossEntropyLoss()

    acc = MetricLogger(moving_average=True, smoothing_factor=smoothing_factor)
    acc_top5 = MetricLogger(moving_average=True, smoothing_factor=smoothing_factor)
    for epoch in tqdm(range(num_epochs), disable=not tqdm_progress):
        classifier.train()

        correct = 0
        total = 0
        for x, label in utils.batch_iter(*train_data, batch_size=batch_size):
            # load data
            x, label = x.to(device), label.to(device)

            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)

            # loss and backprop
            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()

            _, pred_class = torch.max(pred_logits, 1)
            correct += pred_class.eq(label.view_as(pred_class)).sum().item()
            total += len(label)

        if scheduler is not None:
            scheduler.step()

        # compute classification accuracies
        train_acc = correct/total
        test_top_1, test_top_5 = compute_top_k_accuracy(classifier, test_data, batch_size=batch_size, device=device)

        # log
        acc.update(train=train_acc, test=test_top_1)
        acc_top5.update(train=test_top_5)
        if writer is not None:
            writer.add_scalar('eval_debug/train-%r' % tag, train_acc, epoch)
            writer.add_scalar('eval_debug/test-%r' % tag, test_top_1, epoch)
            writer.add_scalar('eval_debug/test-%r-top5' % tag, test_top_5, epoch)
    return acc, acc_top5
