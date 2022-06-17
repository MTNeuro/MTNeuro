import os
import pickle

import numpy as np
from tqdm import tqdm
import torch

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
except:
    torch_geometric = None


class RodentDataset:
    FILENAMES = {
        'mouse': ('fr_bins_n_4full.npy', 'behav_states_n_4full.npy'),
        'rat': ('XYF06_1217_b5_neurons_qualityfr_bins_n_4full.npy', 'XYF06_1217_b5_neurons_qualitybehav_states_n_4full.npy'),
    }

    def __init__(self, root, rodent='mouse', split='train', train_split=0.8, val_split=0.1):
        if torch_geometric is None:
            raise ImportError('`RodentDataset` requires `torch_geometric`.')

        self.root = root
        # get path to data
        assert rodent in ['mouse', 'rat']
        self.rodent = rodent

        self.fr_filename, self.label_filename = self.FILENAMES[self.rodent]
        self.fr_path = os.path.join(self.root, self.fr_filename)
        self.label_path = os.path.join(self.root, self.label_filename)

        # train/val split
        assert split is None or split in ['train', 'val', 'test', 'trainval'], 'got {}'.format(split)
        self.split = split
        self.train_split = train_split
        self.val_split = val_split

        # load data
        data_train_test = self._load()
        data_train_test = self._convert_to_graph(data_train_test)

        # split data
        self.data = self._split_train_test(data_train_test, split=split)

    def _load(self):
        firing_rates = np.load(self.fr_path, allow_pickle=True).T
        labels = np.load(self.label_path, allow_pickle=True)
        return {'firing_rates': firing_rates, 'labels': labels}

    def _convert_to_graph(self, data):
        x = torch.Tensor(data['firing_rates'])
        y = torch.LongTensor(data['labels'])
        t = torch.arange(0, y.size(0), dtype=torch.long)

        # build index
        edge_index, _ = torch_geometric.utils.dense_to_sparse(torch.diag(torch.ones(x.size(0) - 1,), 1))
        # create graph
        graph = Data(x=x, y=y, t=t, edge_index=edge_index)
        return graph

    def _split_train_test(self, data, split):
        if split is None:
            return data

        num_samples = len(data.y)
        split_id = int(num_samples * (self.train_split + self.val_split))

        sub_data = Data()

        if split == 'test':
            for key, item in data:
                sub_data[key] = item[split_id:]
            sub_data.edge_index, _ = torch_geometric.utils.dense_to_sparse(torch.diag(torch.ones(sub_data.x.size(0) - 1,), 1))
            return sub_data
        else:
            for key, item in data:
                data[key] = item[:split_id]
            if split == 'trainval':
                data.edge_index, _ = torch_geometric.utils.dense_to_sparse(
                    torch.diag(torch.ones(data.x.size(0) - 1, ), 1))
                return data
            else:
                split_id = int(len(data.y) * self.train_split)
                if split == 'train':
                    for key, item in data:
                        sub_data[key] = item[:split_id]
                elif split == 'val':
                    for key, item in data:
                        sub_data[key] = item[split_id:]
                sub_data.edge_index, _ = torch_geometric.utils.dense_to_sparse(
                    torch.diag(torch.ones(sub_data.x.size(0) - 1, ), 1))
                return sub_data

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.data

    def get_mean_std(self, feature):
        feature = 'x' if feature == 'firing_rates' else feature
        x = self.data[feature]
        return x.mean(dim=0), x.std(dim=0)

    def get_class_data(self):
        firing_rates = self.data.x
        labels = self.data.y
        return firing_rates, labels

    def __repr__(self):
        return '{}(rodent={}, split={})'.format(self.__class__.__name__, self.rodent, self.split)
