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

from MTNeuro.self_supervised.data.io import loadmat



class ReachNeuralDataset:
    FILENAMES = {
        ('mihi', 1): 'full-mihi-03032014',
        ('mihi', 2): 'full-mihi-03062014',
        ('chewie', 1): 'full-chewie-10032013',
        ('chewie', 2): 'full-chewie-12192013',
    }

    def __init__(self, root, primate='mihi', day=1, split='train', binning_period=0.05, train_split=0.8, val_split=0.1):
        if torch_geometric is None:
            raise ImportError('`ReachNeuralDataset` requires `torch_geometric`.')

        self.root = root
        # get path to data
        assert primate in ['mihi', 'chewie']
        assert day in [1, 2]
        self.primate = primate
        self.day = day

        self.filename = self.FILENAMES[(self.primate, day)]
        self.raw_path = os.path.join(self.root, 'raw/%s.mat') % self.filename
        self.processed_path = os.path.join(self.root, 'processed/%s.pkl') % (self.filename + '-%.2f' % binning_period)

        # get binning parameters
        self.binning_period = binning_period

        # train/val split
        assert split is None or split in ['train', 'val', 'test', 'trainval'], 'got {}'.format(split)
        self.split = split
        self.train_split = train_split
        self.val_split = val_split

        ### Process data
        # load data
        if not os.path.exists(self.processed_path):
            data_train_test = self._process()
        else:
            data_train_test = self._load_processed_data()

        # split data
        self.data = self._split_train_test(data_train_test, split=split)

    def _process(self):
        print('Preparing dataset: Binning data.')
        # load data
        mat_dict = loadmat(self.raw_path)

        # bin data
        data = self._bin_data(mat_dict)

        # convert to graphs
        data = self._convert_to_graphs(data)

        self._save_processed_data(data)
        return data

    def _bin_data(self, mat_dict):
        # load matrix
        trialtable = mat_dict['trial_table']
        neurons = mat_dict['out_struct']['units']
        pos = np.array(mat_dict['out_struct']['pos'])
        vel = np.array(mat_dict['out_struct']['vel'])
        acc = np.array(mat_dict['out_struct']['acc'])
        force = np.array(mat_dict['out_struct']['force'])
        time = vel[:, 0]

        num_neurons = len(neurons)
        num_trials = trialtable.shape[0]

        data_list = {'firing_rates': [], 'position': [], 'velocity': [], 'acceleration': [],
                'force': [], 'labels': [], 'sequence': []}
        for trial_id in tqdm(range(num_trials)):
            min_T = trialtable[trial_id, 9]
            max_T = trialtable[trial_id, 12]

            # grids= minT:(delT-TO):(maxT-delT);
            grid = np.arange(min_T, max_T + self.binning_period, self.binning_period)
            grids = grid[:-1]
            gride = grid[1:]
            num_bins = len(grids)

            neurons_binned = np.zeros((num_bins, num_neurons))
            pos_binned = np.zeros((num_bins, 2))
            vel_binned = np.zeros((num_bins, 2))
            acc_binned = np.zeros((num_bins, 2))
            force_binned = np.zeros((num_bins, 2))
            targets_binned = np.zeros((num_bins,))
            id_binned = np.arange(num_bins)

            for k in range(num_bins):
                bin_mask = (time >= grids[k]) & (time <= gride[k])
                if len(pos) > 0:
                    pos_binned[k, :] = np.mean(pos[bin_mask, 1:], axis=0)
                vel_binned[k, :] = np.mean(vel[bin_mask, 1:], axis=0)
                if len(acc):
                    acc_binned[k, :] = np.mean(acc[bin_mask, 1:], axis=0)
                if len(force) > 0:
                    force_binned[k, :] = np.mean(force[bin_mask, 1:], axis=0)
                targets_binned[k] = trialtable[trial_id, 1]

            for i in range(num_neurons):
                for k in range(num_bins):
                    spike_times = neurons[i]['ts']
                    bin_mask = (spike_times >= grids[k]) & (spike_times <= gride[k])
                    neurons_binned[k, i] = np.sum(bin_mask) / self.binning_period

            data_list['firing_rates'].append(neurons_binned)
            data_list['position'].append(pos_binned)
            data_list['velocity'].append(vel_binned)
            data_list['acceleration'].append(acc_binned)
            data_list['force'].append(force_binned)
            data_list['labels'].append(targets_binned)
            data_list['sequence'].append(id_binned)
        return data_list

    def _convert_to_graphs(self, data_list):
        num_trials = len(data_list['firing_rates'])
        graph_list = []
        for trial_id in range(num_trials):
            x = torch.Tensor(data_list['firing_rates'][trial_id])
            pos = torch.Tensor(data_list['position'][trial_id])
            vel = torch.Tensor(data_list['velocity'][trial_id])
            acc = torch.Tensor(data_list['acceleration'][trial_id])
            force = torch.Tensor(data_list['force'][trial_id])
            y = torch.LongTensor(data_list['labels'][trial_id])
            t = torch.LongTensor(data_list['sequence'][trial_id])
            # build index
            edge_index, _ = torch_geometric.utils.dense_to_sparse(torch.diag(torch.ones(x.size(0) - 1,), 1))
            # create graph
            graph = Data(x=x, pos=pos, vel=vel, acc=acc, force=force, y=y, t=t, edge_index=edge_index)
            graph_list.append(graph)
        return graph_list

    def _save_processed_data(self, data):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        with open(self.processed_path, 'wb') as output:
            pickle.dump({'data': data}, output)

    def _load_processed_data(self):
        with open(self.processed_path, "rb") as fp:
            data = pickle.load(fp)['data']
        return data

    def _split_train_test(self, data, split):
        if split is None:
            return data
        num_trials = len(data)
        split_id = int(num_trials * (self.train_split + self.val_split))

        if split == 'test':
            return data[split_id:]
        else:
            data = data[:split_id]
            if split == 'trainval':
                return data
            else:
                split_id = int(len(data) * self.train_split)
                if split == 'train':
                    return data[:split_id]
                elif split == 'val':
                    return data[split_id:]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _unfold(self, tensor):
        """
        stride = round(self.binning_stride / self.ELEMENTARY_BIN_SIZE)
        tensor = tensor[:tensor.size(0) - tensor.size(0) % stride]
        tensor = tensor.float().view(1, 1, -1, stride)

        kernel_size = tensor.size(0) - round(self.binning_period / self.binning_stride) + 1
        return F.unfold(tensor, kernel_size=(kernel_size, 1), padding=(1, 0))
        """
        pass

    @property
    def full_data(self):
        return Batch.from_data_list(self.data)

    def get_mean_std(self, feature):
        feature = 'x' if feature == 'firing_rates' else feature
        x = self.full_data[feature]
        return x.mean(dim=0), x.std(dim=0)

    def get_class_data(self, velocity_threshold=-1.):
        data = self.full_data
        velocity_mask = torch.norm(data.vel, 2, dim=1) > velocity_threshold
        firing_rates = data.x[velocity_mask]
        labels = data.y[velocity_mask]
        return firing_rates, labels

    def get_angular_data(self, velocity_threshold=-1.):
        data = self.full_data
        velocity_mask = torch.norm(data.vel, 2, dim=1) > velocity_threshold
        firing_rates = data.x[velocity_mask]
        labels = data.y[velocity_mask]
        angles = (2 * np.pi / 8 * labels).unsqueeze(-1)
        cos_sin = torch.column_stack([torch.cos(angles), torch.sin(angles)])
        return firing_rates, angles

    def get_feature(self, feat, velocity_threshold=-1.):
        data = self.full_data
        velocity_mask = torch.norm(data.vel, 2, dim=1) > velocity_threshold
        feat_x = data[feat][velocity_mask]
        return feat_x

    def __repr__(self):
        return '{}(primate={}, day={}, split={})'.format(self.__class__.__name__, self.primate, self.day, self.split)
