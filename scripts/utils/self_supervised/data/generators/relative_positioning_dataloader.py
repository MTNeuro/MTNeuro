import copy

import numpy as np
import torch
from torch import Tensor
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import sort_edge_index, dense_to_sparse
except:
    torch_geometric = None


class RandomIndexSampler(torch.utils.data.Sampler):
    """Adapted from torch_geometric"""
    def __init__(self, num_nodes: int, batch_size: int, shuffle: bool = False, drop_last: bool = True):
        self.N = num_nodes
        self.batch_size = batch_size
        self.num_parts = int(np.ceil(self.N / self.batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_ids = self.get_node_indices()

    def get_node_indices(self):
        n_id = torch.arange(self.num_parts, dtype=torch.long)
        n_id = n_id.repeat_interleave(self.batch_size)[:self.N]  # last batch will have less than batch_size nodes.
        n_id = n_id[torch.randperm(n_id.size(0))]  # random permute
        n_ids = [(n_id == i).nonzero(as_tuple=False).view(-1)
                 for i in range(self.num_parts)]
        if self.drop_last and n_ids[-1].size(0) < self.batch_size:
            n_ids = n_ids[:-1]
        return n_ids

    def __iter__(self):
        if self.shuffle:
            self.n_ids = self.get_node_indices()
        return iter(self.n_ids)

    def __len__(self):
        return self.num_parts


class RelativePositioningSampler(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True,
                 pos_kmin=0, pos_kmax=0, neg_k=1, transform=None, precompute_neg=True, **kwargs):
        if torch_geometric is None:
            raise ImportError('`RelativePositioningSampler` requires `torch_geometric`.')

        self.pos_kmin, self.pos_kmax, self.neg_k = pos_kmin, pos_kmax, neg_k
        self.precompute_neg = precompute_neg

        data_list = [self._add_pos_cand_edge_indices(data) for data in dataset]
        self.data = Batch.from_data_list(data_list)
        self.data = copy.copy(self.data)

        print(self.data.ccand_edge_index.max(), self.data.ccand_edge_index.size())

        self.num_nodes = self.data.num_nodes
        self.num_examples = self.num_nodes

        # compute number of neighbors
        self.num_pos_neighs = self.data.pos_edge_index.new_zeros(self.data.num_nodes)
        self.num_pos_neighs.scatter_add_(0, self.data.pos_edge_index[0],
                                         self.data.pos_edge_index.new_ones(self.data.pos_edge_index.size(1)))

        # transform
        self.transform = transform

        super().__init__(
            self, batch_size=1,
            sampler=RandomIndexSampler(self.num_nodes, batch_size, shuffle, drop_last),
            collate_fn=self.__collate__, **kwargs)

    def _add_pos_cand_edge_indices(self, data):
        n = data.num_nodes
        # positive edge index: default self only
        data.pos_edge_index = self._fast_diag(n, self.pos_kmin, self.pos_kmax)
        # complementary negative edge index
        data.ccand_edge_index = self._fast_diag(n, -self.neg_k, self.neg_k)
        return data

    def _fill_diagonal(self, n, k_min=0, k_max=0):
        k_min = k_min if k_min is not None else -n
        k_max = k_max if k_max is not None else n
        dist_to_diag = np.add.outer(-np.arange(n), np.arange(n))

        adj = torch.BoolTensor(np.logical_and(dist_to_diag <= k_max, dist_to_diag >= k_min))
        edge_index, _ = dense_to_sparse(adj)
        return edge_index

    def _block_diag_indices(self, n, k_min=0, k_max=0):
        k_min = k_min if k_min is not None else -n
        k_max = k_max if k_max is not None else n
        all_pairs = set(zip(*map(np.ndarray.tolist, np.triu_indices(n, k_min))))
        rm_pairs = set(zip(*map(np.ndarray.tolist, np.triu_indices(n, 1 + k_max))))
        pairs = all_pairs - rm_pairs
        edge_index = torch.LongTensor(np.array(list(pairs))).T
        edge_index, _ = sort_edge_index(edge_index, num_nodes=n)
        return edge_index

    def _fast_diag(self, n, k_min=0, k_max=0):
        assert k_min <= k_max
        k_min = k_min if k_min is not None else -n
        k_max = k_max if k_max is not None else n

        s = torch.repeat_interleave(torch.arange(n, dtype=torch.long), k_max - k_min + 1)
        t = (torch.arange(k_min, k_max+1, dtype=torch.long) + torch.arange(n, dtype=torch.long).view((-1, 1))).view(-1)
        edge_index = torch.stack([s, t])
        mask = (edge_index[1] >= 0) & (edge_index[1] < n)
        edge_index = edge_index[:, mask]
        return edge_index

    def __getitem__(self, idx):
        return idx

    def sample_neighbor(self, edge_index, num_neighs):
        shuffled_edge_index = edge_index[:, torch.randperm(edge_index.size(1))]
        # sort
        sorted_edge_index = shuffled_edge_index[:, shuffled_edge_index[0].argsort()]
        # select first neighbor for each node
        slicer = torch.cat([num_neighs.new_zeros(1), num_neighs.cumsum(dim=0)[:-1]], dim=0)
        sampled_edge_index = sorted_edge_index[:, slicer]
        return sampled_edge_index

    def __collate__(self, node_idx):
        node_idx = node_idx[0]

        data = Data()

        n_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        n_mask[node_idx] = 1

        # sample positives
        pos_edge_index = self.data.pos_edge_index
        pos_e_mask = n_mask[pos_edge_index[0]]  # select sampled nodes
        pos_edge_index = pos_edge_index[:, pos_e_mask]
        pos_edge_index = self.sample_neighbor(pos_edge_index, self.num_pos_neighs[n_mask])

        # use positive views as mining candidates
        cn_mask = n_mask.clone()
        cn_mask[pos_edge_index[1]] = 1

        # print(cn_mask.sum(), n_mask.sum())

        # self-mining
        ccand_edge_index = self.data.ccand_edge_index
        ccand_e_mask = n_mask[ccand_edge_index[0]] & cn_mask[ccand_edge_index[1]]
        ccand_edge_index = ccand_edge_index[:, ccand_e_mask]

        # update edge_index, not useful right now
        edge_index = self.data.edge_index
        e_mask = cn_mask[edge_index[0]] & cn_mask[edge_index[1]]
        edge_index = edge_index[:, e_mask]

        # relabel nodes
        n_idx = torch.zeros(self.num_nodes, dtype=torch.long)
        n_idx[cn_mask] = torch.arange(cn_mask.sum())
        data.edge_index = n_idx[edge_index]
        data.pos_edge_index = n_idx[pos_edge_index]
        data.ccand_edge_index = n_idx[ccand_edge_index]

        # get feats for selected samples
        for key, item in self.data:
            if isinstance(item, Tensor) and item.size(0) == self.num_nodes:
                data[key] = item[cn_mask]

        # transform x
        data.x = self.transform(data.x, data.batch) if self.transform is not None else data.x
        return data

    @staticmethod
    def prepare_views(data):
        view_1_index = data.pos_edge_index[0]
        view_2_index = data.pos_edge_index[1]
        outputs = {'view_1': data.x[view_1_index], 'view_2': data.x[view_2_index],
                   'view_m': 'view_1', 'view_pool': data.x,
                   'c_cand_edge_index': data.ccand_edge_index,
                   'metadata_m': {'labels': data.y[view_1_index], 'trial': data.batch[view_1_index], 'time': data.t[view_1_index]},
                   'metadata_pool': {'labels': data.y, 'trial': data.batch, 'time': data.t}}
        return outputs

"""
    def compute_ccand_edge_index(self, n_mask, cn_mask):
        if self.precompute_neg:
            ccand_edge_index = self.data.ccand_edge_index
            ccand_e_mask = n_mask[ccand_edge_index[0]] & cn_mask[ccand_edge_index[1]]
            ccand_edge_index = ccand_edge_index[:, ccand_e_mask]
        else:
            # todo assert batch = 1 and data has t
            t_dist = torch.abs(self.data.t[n_mask] - self.data.t[cn_mask].view(-1, 1))
            ccand_adj = t_dist > self.neg_k
            ccand_edge_index = edge_index, _ = dense_to_sparse(ccand_adj)
        return ccand_edge_index
"""