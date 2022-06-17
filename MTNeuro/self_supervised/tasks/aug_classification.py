import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Simple_Trans(Dataset):
    def __init__(self, data, transform=None):
        # [reps, labels]
        self.reps = data[0]
        self.labels = data[1]
        # print(self.reps.shape, self.labels.shape) # torch.Size([60000, 64]) torch.Size([60000])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.reps[idx, :], self.labels[idx]

def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # print(pred.shape, correct.shape)
    # torch.Size([5, 10000]) torch.Size([5, 10000])
    # print(pred[:, :5], correct[:, :5])

    res = []
    for k in topk:
        correct_k = torch.reshape(correct[:k], (-1,)).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).cpu().data.item())
    return res

class linear_clf(object):
    def __init__(self, net, classifier, optimizer, train_dataloader, test_dataloader, device = "cpu", batch_size=1024,
                 num_epochs = 10, disable_tqdm = False, tb_path = None):
        self.net = net
        self.classifier = classifier
        self.optimizer = optimizer

        self.disable_tqdm = disable_tqdm
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        if tb_path is not None:
            self.writer = SummaryWriter(tb_path)
        else:
            self.writer = False

        self.data_train = Simple_Trans(self.compute_representations(train_dataloader))
        self.data_test = Simple_Trans(self.compute_representations(test_dataloader))

        self.best_number = 0
        self.step = 0
        self.train_linear_layer()

        #self.train_acc = self.compute_accuracy(DataLoader(self.data_train, batch_size=batch_size))
        #self.test_acc = self.compute_accuracy(DataLoader(self.data_test, batch_size=batch_size))

    def compute_representations(self, dataloader):
        """ store the representations
        :param net: ResNet or smth
        :param dataloader: train_loader and test_loader
        """
        self.net.eval()
        reps, labels = [], []

        for i, (x, label) in enumerate(dataloader):
            # load data
            x = x.to(self.device)
            labels.append(label)

            # forward
            with torch.no_grad():
                representation = self.net(x)
                reps.append(representation.detach().cpu())

            if i % 100 == 0:
                reps = [torch.cat(reps, dim=0)]
                labels = [torch.cat(labels, dim=0)]

        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        return [reps, labels]

    def compute_accuracy(self, dataloader):
        self.classifier.eval()
        right = []
        total = []

        pred_ls = []
        label_ls = []

        for x, label in dataloader:
            x, label = x.to(self.device), label.to(self.device)
            # feed to network and classifier
            with torch.no_grad():
                pred_logits = self.classifier(x)
            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            right.append((pred_class == label).sum().item())
            total.append(label.size(0))

            pred_ls.append(pred_logits)
            label_ls.append(label)

        pred = torch.cat(pred_ls, dim=0)
        label = torch.cat(label_ls, dim=0)

        # print(pred_ls[0].shape, label_ls[0].shape, pred.shape, label.shape)
        # torch.Size([1024, 100]) torch.Size([1024]) torch.Size([10000, 100]) torch.Size([10000])

        res = accuracy(pred, label)
        self.classifier.train()
        return sum(right) / sum(total), res[0], res[1]

    def train_linear_layer(self):
        class_criterion = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(range(self.num_epochs), disable=self.disable_tqdm)
        for epoch in progress_bar:
            for x, label in DataLoader(self.data_train, batch_size=self.batch_size):
                self.step = self.step + 1
                self.classifier.train()
                x, label = x.to(self.device), label.to(self.device)
                pred_class = self.classifier(x)
                loss = class_criterion(pred_class, label)

                if self.writer:
                    self.writer.add_scalar('Train', loss, self.step)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            curr_number, acc1, acc5 = self.compute_accuracy(DataLoader(self.data_test, batch_size=self.batch_size))
            #print(acc5)
            if curr_number >= self.best_number:
                self.best_number = curr_number

            if self.writer:
                self.writer.add_scalar('Eval', curr_number, epoch)

            progress_bar.set_description('Linear_CLF Epoch: [{}/{}] Acc@1:{:.3f} BestAcc@1:{:.3f} K_Acc1:{:.3f} K_Acc5:{:.3f}'
                                         .format(epoch, self.num_epochs, curr_number, self.best_number, acc1, acc5))

class aug_linear_clf(object):
    def __init__(self, net, classifier, optimizer, train_dataloader, test_dataloader, train_transform, test_transform, device = "cpu", batch_size=1024,
                 num_epochs = 10, disable_tqdm = False):
        self.net = net
        self.classifier = classifier
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.disable_tqdm = disable_tqdm
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.best_number = 0
        self.train_linear_layer()

        self.train_acc = self.compute_accuracy(DataLoader(self.data_train, batch_size=batch_size))
        self.test_acc = self.compute_accuracy(DataLoader(self.data_test, batch_size=batch_size))

        print("train acc", self.train_acc, "val acc", self.test_acc, "best val acc", self.best_number)

    def compute_representations(self, dataloader, transform):
        """ store the representations
        :param net: ResNet or smth
        :param dataloader: train_loader and test_loader
        """
        self.net.eval()
        reps, labels = [], []

        for i, (x, label) in enumerate(dataloader):
            # load data
            x = transform(x).to(self.device)
            labels.append(label)

            # forward
            with torch.no_grad():
                representation = self.net(x)
                reps.append(representation.detach().cpu())

            if i % 100 == 0:
                reps = [torch.cat(reps, dim=0)]
                labels = [torch.cat(labels, dim=0)]

        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        return [reps, labels]

    def compute_accuracy(self, dataloader):
        self.classifier.eval()
        right = []
        total = []
        for x, label in dataloader:
            x, label = x.to(self.device), label.to(self.device)
            # feed to network and classifier
            with torch.no_grad():
                pred_logits = self.classifier(x)
            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            right.append((pred_class == label).sum().item())
            total.append(label.size(0))
        self.classifier.train()
        return sum(right) / sum(total)

    def train_linear_layer(self):
        class_criterion = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(range(self.num_epochs), disable=self.disable_tqdm)
        for epoch in progress_bar:
            self.data_train = Simple_Trans(self.compute_representations(self.train_dataloader, self.train_transform))
            self.data_test = Simple_Trans(self.compute_representations(self.test_dataloader, self.test_transform))
            for x, label in DataLoader(self.data_train, batch_size=self.batch_size):
                self.classifier.train()
                x, label = x.to(self.device), label.to(self.device)
                pred_class = self.classifier(x)
                loss = class_criterion(pred_class, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            curr_number = self.compute_accuracy(DataLoader(self.data_test, batch_size=self.batch_size))
            if curr_number >= self.best_number:
                self.best_number= curr_number

            progress_bar.set_description('Linear_CLF Epoch: [{}/{}] Acc@1:{:.3f}%  BestAcc@1:{:.3f}%'
                                         .format(epoch, self.num_epochs, curr_number, self.best_number))

