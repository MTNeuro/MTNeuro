import torch
import numpy as np
from sklearn import metrics

from MTNeuro.self_supervised.data import utils
from MTNeuro.self_supervised.utils import MetricLogger


def compute_angle_representations(net, data, device='cpu'):
    net.eval()
    x, target_angle, target_cos_sin = data
    x = x.to(device)
    with torch.no_grad():
        repr = net(x)
    return repr, target_angle, target_cos_sin


def train_angle_classifier(train_data, val_data, test_data, device='cpu', batch_size=512):
    def train(classifier, train_data, optimizer):
        classifier.train()

        for step in range(100):
            for x, target_angle, target_cos_sin in utils.batch_iter(*train_data, batch_size=batch_size):
                x, target_cos_sin = x.to(device), target_cos_sin.to(device)

                # forward
                optimizer.zero_grad()
                pred_logits = classifier(x)

                # loss and backprop
                loss = criterion(pred_logits, target_cos_sin)
                loss.backward()
                optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, target_angle, target_cos_sin = data
        target_angle = target_angle.squeeze()
        # feed to network and classifier
        pred_logits = classifier(x.to(device)).detach().cpu()
        # compute acc
        pred_angles = torch.atan2(pred_logits[:, 1], pred_logits[:, 0])
        pred_angles[pred_angles < 0] = pred_angles[pred_angles < 0] + 2 * np.pi
        diff_angles = torch.abs(pred_angles - target_angle)
        diff_angles[diff_angles > np.pi] = torch.abs(diff_angles[diff_angles > np.pi] - 2 * np.pi)
        acc = (diff_angles < (np.pi / 8)).sum() / diff_angles.size(0)
        delta_acc = (diff_angles < (3 * np.pi / 16)).sum() / diff_angles.size(0)
        return acc, delta_acc

    num_feats = next(iter(train_data)).size(1)  # works for both dataloader and list
    criterion = torch.nn.MSELoss()
    acc = MetricLogger(early_stopping=True, max=True)
    delta_acc = MetricLogger(early_stopping=True, max=True)

    for weight_decay in 2.0 ** np.arange(-10, 10):
        classifier = torch.nn.Linear(num_feats, 2).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

        train(classifier, train_data, optimizer)
        train_acc, train_delta_acc = test(classifier, train_data)
        val_acc, val_delta_acc = test(classifier, val_data)
        test_acc, test_delta_acc = test(classifier, test_data)
        acc.update(train=train_acc, val=val_acc, test=test_acc, step=weight_decay)
        delta_acc.update(train=train_delta_acc, val=val_delta_acc, test=test_delta_acc, step=weight_decay)

    print(acc)
    print('Delta', delta_acc)

    return (acc.train_max, acc.val_max, acc.test_max), (delta_acc.hist(acc.step_minmax))


def compute_representations(net, data, device='cpu'):
    net.eval()
    x, y = data
    x = x.to(device)
    with torch.no_grad():
        repr = net(x)
    return repr, y


def train_classifier(train_data, val_data, test_data, device='cpu', batch_size=512):
    def train(classifier, train_data, optimizer):
        classifier.train()

        for step in range(100):
            for x, y in utils.batch_iter(*train_data, batch_size=batch_size):
                x, y = x.to(device), y.to(device)

                # forward
                optimizer.zero_grad()
                pred_logits = classifier(x)

                # loss and backprop
                loss = criterion(pred_logits, y)
                loss.backward()
                optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, y = data
        # feed to network and classifier
        pred_logits = classifier(x.to(device)).detach().cpu()
        # compute acc
        _, pred_class = torch.max(pred_logits, 1)
        acc = pred_class.eq(y.view_as(pred_class)).sum().item() / y.size(0)
        # compute f1-score
        f1 = metrics.f1_score(y.numpy(), pred_class.numpy(), average='weighted') if pred_class.sum() > 0 else 0
        return acc, f1

    num_feats = next(iter(train_data)).size(1)  # works for both dataloader and list
    criterion = torch.nn.CrossEntropyLoss()
    acc = MetricLogger(early_stopping=True, max=True)
    f1 = MetricLogger(early_stopping=True, max=True)

    for weight_decay in 2.0 ** np.arange(-10, 10):
        classifier = torch.nn.Linear(num_feats, 3).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

        train(classifier, train_data, optimizer)
        train_acc, train_f1 = test(classifier, train_data)
        val_acc, val_f1 = test(classifier, val_data)
        test_acc, test_f1 = test(classifier, test_data)
        acc.update(train=train_acc, val=val_acc, test=test_acc, step=weight_decay)
        f1.update(train=train_f1, val=val_f1, test=test_f1, step=weight_decay)

    return (acc.train_max, acc.val_max, acc.test_max), (f1.hist(acc.step_minmax))
