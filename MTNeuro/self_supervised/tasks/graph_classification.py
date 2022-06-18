import numpy as np
import torch
from sklearn import metrics

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

    for data in dataloader:
        # forward
        data = data.to(device)
        with torch.no_grad():
            reps.append(net(data))
            labels.append(data.y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]


def train_linear_layer(num_classes, train_data, val_data, test_data, device='cpu', batch_size=512):
    def train(classifier, train_data, optimizer):
        classifier.train()

        for step in range(100):
            for x, label in utils.batch_iter(*train_data, batch_size=batch_size):
                x, label = x.to(device), label.to(device)

                # forward
                optimizer.zero_grad()
                pred_logits = classifier(x)

                # loss and backprop
                loss = criterion(pred_logits, label)
                loss.backward()
                optimizer.step()

        # compute train acc
        return 0.

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.cpu().numpy()

        # feed to network and classifier
        pred_logits = classifier(x.to(device))
        pred_class = (pred_logits > 0).float().cpu().numpy()

        return metrics.f1_score(label, pred_class, average='micro') if pred_class.sum() > 0 else 0

    num_feats = next(iter(train_data)).size(1)  # works for both dataloader and list
    criterion = torch.nn.BCEWithLogitsLoss()
    acc = MetricLogger(early_stopping=True, max=True)

    for weight_decay in 2.0 ** np.arange(-10, 11):
        classifier = torch.nn.Linear(num_feats, num_classes).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

        train_acc = train(classifier, train_data, optimizer)
        val_acc = test(classifier, val_data)
        test_acc = test(classifier, test_data)
        acc.update(train=train_acc, val=val_acc, test=test_acc)

    return acc.train_max, acc.val_max, acc.test_max
