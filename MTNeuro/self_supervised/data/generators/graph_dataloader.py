import logging

from scipy.sparse import coo_matrix
import numpy as np
import torch
from torch.utils.data import Dataset
from networkx.algorithms import centrality

try:
    import torch_geometric
except ImportError:
    torch_geometric = None

try:
    from torch_scatter import scatter_add
except:
    scatter_add = None

from MTNeuro.self_supervised.utils.console import console

log = logging.getLogger(__name__)


class MultiGraphGenerator(Dataset):
    def __init__(self, dataset, annulus_r, annulus_R, num_annulus_neighbors, method='annulus_w_padding',
                 weights='uniform', transform_1=None, transform_2=None, transform_m=None):
        super().__init__()

        self.dataset = dataset
        self.graph_generators = [GraphGenerator(data, annulus_r, annulus_R, num_annulus_neighbors, 1, method,
                 weights, transform_1, transform_2, transform_m) for data in dataset]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1
        return batch[0]

    def __getitem__(self, index):
        return self.graph_generators[index][0]

    @staticmethod
    def prepare_views(inputs):
        data, cand_edge_index = inputs
        # todo temporary, change to edge_index type
        outputs = {'view_1': data, 'view_2': 'view_1', 'view_m': 'view_1', 'view_pool': None,
                   'cand_edge_index': cand_edge_index, 'metadata_m': {'labels': 'data.y'},
                   'metadata_pool': {'labels': 'data.y'}}
        return outputs


class GraphGenerator(Dataset):
    # todo add transform here
    def __init__(self, data, annulus_r, annulus_R, num_annulus_neighbors, prefetch, method='annulus_w_padding',
                 weights='uniform', transform_1=None, transform_2=None, transform_m=None, use_oracle=False):
        super().__init__()
        if torch_geometric is None:
            raise ImportError('`GraphGenerator` requires `torch_geometric`.')
        if scatter_add is None:
            raise ImportError('`GraphGenerator` requires `torch_scatter`.')

        self.data = data
        self.r, self.R = annulus_r, annulus_R
        self.num_annulus_neighbors = num_annulus_neighbors
        self.method = method
        self.prefetch = prefetch
        self.use_oracle = use_oracle

        # todo add weight methods
        if weights == 'uniform':
            self.weights = torch.ones(self.data.num_nodes)  # default
        elif weights == 'degree_centrality':
            self.weights = self.degree_centrality
        elif weights == 'degree_centrality_inv':
            self.weights = 1 / self.degree_centrality
        else:
            raise ValueError

        if self.method == 'annulus_w_padding':
            self.precompute_annulus_stuff()
        elif self.method == 'sample_nodes':
            # todo generate random graph
            self.precompute_sample_nodes_stuff()
        else:
            raise ValueError

        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.transform_m = transform_m

        assert (self.transform_1 is not None) == (self.transform_2 is not None) == (self.transform_m is not None)

    def __len__(self):
        return self.prefetch

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1
        return batch[0]

    def precompute_sample_nodes_stuff(self):
        self.precomputed_source = torch.repeat_interleave(torch.arange(self.data.num_nodes), self.num_annulus_neighbors)

    def precompute_annulus_stuff(self):
        # get annulus graph
        annulus_edge_index = self.precompute_neigh_annulus(self.data, r=self.r, R=self.R)
        row, col = annulus_edge_index
        # compute the number of neighbors for each node
        num_neighs = row.new_zeros(self.data.num_nodes)
        num_neighs.scatter_add_(0, row, row.new_ones(row.size(0)))

        # well connected nodes - need to subsample their annulus neighborhood
        nodes_that_need_subsampling = num_neighs > self.num_annulus_neighbors
        self.edges_that_need_subsampling = annulus_edge_index[:, nodes_that_need_subsampling[row]]
        # precompute slicer, all that is left to do is shuffle and slice
        self.precomputed_annulus_slicer = self.precompute_slice(self.edges_that_need_subsampling[0],
                                                                self.num_annulus_neighbors, nodes_that_need_subsampling)

        # less connected nodes - keep the annulus neighborhood and pad with random edges to nodes sampled from graph
        nodes_that_need_padding = ~nodes_that_need_subsampling
        self.edges_to_keep = annulus_edge_index[:, nodes_that_need_padding[row]]
        self.padding = self.num_annulus_neighbors - num_neighs[nodes_that_need_padding]
        self.nodes_that_need_padding = torch.nonzero(nodes_that_need_padding).squeeze()

        self.max_pad = self.padding.max()
        self.num_pad_nodes = self.nodes_that_need_padding.size(0)
        self.precomputed_source = torch.repeat_interleave(self.nodes_that_need_padding, self.max_pad)

        self.priority_idx = torch.cat([self.edges_to_keep.new_zeros(self.edges_to_keep.size(1)),
                                       self.precomputed_source.new_ones(self.precomputed_source.size(0))])
        self.precomputed_padding_slicer = self.precompute_slice(torch.cat([self.edges_to_keep[0], self.precomputed_source]),
                                                                self.num_annulus_neighbors, nodes_that_need_padding)

    def precompute_slice(self, row, num_neighbors, mask):
        # row does not need to be sorted but will need to be when slicing
        # compute number of neighbors
        num_neighs = row.new_zeros(self.data.num_nodes)
        num_neighs.scatter_add_(0, row, row.new_ones(row.size(0)))
        cum_num_neighs = torch.cat([num_neighs.new_zeros(1), num_neighs.cumsum(dim=0)[:-1]], dim=0)

        cum_num_neighs = cum_num_neighs[mask]

        # slice neighbors
        slice = torch.arange(num_neighbors, device=row.device)
        slice = slice + cum_num_neighs.view(-1, 1)
        slice = slice.view(-1)
        return slice

    def __getitem__(self, index):
        if self.method == 'annulus_w_padding':
            # subsample neighborhood for well connected nodes
            # shuffle
            subsampled_edges = self.edges_that_need_subsampling[:, torch.randperm(self.edges_that_need_subsampling.size(1))]
            # sort
            subsampled_edges = subsampled_edges[:, subsampled_edges[0].argsort()]
            # slice
            subsampled_edges = subsampled_edges[:, self.precomputed_annulus_slicer]

            # subsample nodes from graph to pad less connected nodes
            # sample nodes from the graph
            selected_nodes = torch.multinomial(self.weights, self.max_pad, replacement=False)
            padding_edge_index = torch.stack([self.precomputed_source, selected_nodes.repeat(self.num_pad_nodes)])
            # shuffle
            padding_edge_index = padding_edge_index[:, torch.randperm(padding_edge_index.size(1))]
            # merge with edges to keep
            remaining_edges = torch.cat([self.edges_to_keep, padding_edge_index], dim=1)
            # sort giving priority to annulus edges
            perm_idx = remaining_edges[0] * 2 + self.priority_idx
            perm = perm_idx.argsort()
            remaining_edges = remaining_edges[:, perm]
            # slice
            remaining_edges = remaining_edges[:, self.precomputed_padding_slicer]

            # finally concat all edges
            mining_edges = torch.cat([subsampled_edges, remaining_edges], dim=1)  # not ordered!
            # todo drop self loops
        elif self.method == 'sample_nodes':
            selected_nodes = torch.multinomial(self.weights, self.num_annulus_neighbors, replacement=False)
            mining_edges = torch.stack([self.precomputed_source, selected_nodes.repeat(self.data.num_nodes)])
            # todo drop self loops
        else:
            raise ValueError

        # sort
        mining_edges, _ = torch_geometric.utils.sort_edge_index(mining_edges)  # sort

        # oracle
        if self.use_oracle:
            oracle_mask = self.data.y[mining_edges[0]] == self.data.y[mining_edges[1]]
            mining_edges = mining_edges[:, oracle_mask]

        # transform
        if self.transform_1 is not None:
            # transform on cpu
            view_1 = self.transform_1(self.data)
            view_2 = self.transform_2(self.data)
            view_m = self.transform_m(self.data)
            inputs = {'view_1': view_1, 'view_2': view_2, 'view_m': view_m, 'view_pool': None,
                      'cand_edge_index': mining_edges, 'metadata_m': {'labels': 'data.y'},
                      'metadata_pool': {'labels': 'data.y'}}
            return inputs
        else:
            return self.data, mining_edges

    @staticmethod
    def prepare_views(inputs):
        data, mining_edges = inputs
        outputs = {'view_1': data, 'view_2': 'view_1', 'view_m': 'view_1', 'view_pool': None,
                   'cand_edge_index': mining_edges, 'metadata_m': {'labels': 'data.y'},
                   'metadata_pool': {'labels': 'data.y'}}
        return outputs

    @property
    def degree_centrality(self):
        # todo only works for undirected graphs
        edge_index = self.data.edge_index
        degree_C = torch_geometric.utils.degree(edge_index[0], num_nodes=None)
        return degree_C

    @property
    def closeness_centrality(self):
        # todo add closeness_edges based on distance threshold
        # todo only works for undirected graphs
        G = torch_geometric.utils.to_networkx(self.data, to_undirected=True)
        closeness_C = centrality.closeness_centrality(G)
        # todo convert to vector
        return closeness_C

    @staticmethod
    def precompute_neigh_annulus(data, r, R, device='cpu'):
        # todo use TwoHop and torch sparse matrix multiplication
        edge_index = data.edge_index
        with console.status("[bold green]Building neighborhood annulus...") as status:
            log.debug('[bold orange]Neighbor annulus[/bold orange] Converting edge_index to sparse matrix.',
                      extra={"markup": True})
            adj_matrix = torch_geometric.utils.to_scipy_sparse_matrix(edge_index).astype(bool)

            # todo transfer to gpu
            k_hop_adj_matrix = adj_matrix ** 0
            less_than_r_hops_away = coo_matrix(k_hop_adj_matrix.shape)  # empty
            for k in range(r):
                less_than_r_hops_away += k_hop_adj_matrix
                k_hop_adj_matrix = k_hop_adj_matrix * adj_matrix
            log.debug('[bold orange]Neighbor annulus[/bold orange] inner r, number of neighbors: %d.'
                      % less_than_r_hops_away.getnnz(), extra={"markup": True})

            less_than_R_hops_away = coo_matrix(k_hop_adj_matrix.shape)  # empty
            for k in range(r, R):
                less_than_R_hops_away += k_hop_adj_matrix
                k_hop_adj_matrix = k_hop_adj_matrix * adj_matrix
            less_than_R_hops_away += k_hop_adj_matrix
            log.debug('[bold orange]Neighbor annulus[/bold orange] outer R, number of neighbors: %d.'
                      % less_than_R_hops_away.getnnz(), extra={"markup": True})

            log.debug('[bold orange]Neighbor annulus[/bold orange] Performing set difference.',
                      extra={"markup": True})
            # logical not will consume too much memory
            r_edge_index = np.column_stack(less_than_r_hops_away.nonzero())  # is C-Contiguous
            R_edge_index = np.column_stack(less_than_R_hops_away.nonzero())

            r_set = r_edge_index.view([('', r_edge_index.dtype)] * 2)
            R_set = R_edge_index.view([('', R_edge_index.dtype)] * 2)

            set_diff = np.setdiff1d(R_set, r_set).view(r_edge_index.dtype).reshape(-1, 2)
            row, col = set_diff.T
            # convert to torch sparse
            # this is fixed so copy it to gpu once and for all
            log.debug('[bold orange]Neighbor annulus[/bold orange] Converting to sparse coo tensor.',
                      extra={"markup": True})
            # todo add number of hops
        out = torch.LongTensor([row, col])
        out, _ = torch_geometric.utils.sort_edge_index(out)  # sort
        return out
