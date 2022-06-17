import torch
import numpy as np

try:
    import torch_geometric
    from torch_sparse import spspmm, coalesce
    from torch_geometric.utils import remove_self_loops
    from torch_sparse import SparseTensor
except:
    torch_geometric = None

from .graphs import get_graph_drop_transform


class TwoHop:
    r"""Adds the two hop edges to the edge indices."""
    def __init__(self, include_1_hop=True):
        if torch_geometric is None:
            raise ImportError('`TwoHop` requires `torch_geometric`.')
        self.include_1_hop = include_1_hop

    def __call__(self, data):
        edge_index = data.edge_index.cpu()
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        if self.include_1_hop:
            edge_index = torch.cat([edge_index, index], dim=1)
        else:
            edge_index = index
        data.edge_index = coalesce(edge_index, None, N, N)[0].to(data.x.device)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class HalfHop:
    def __init__(self):
        if torch_geometric is None:
            raise ImportError('`HalfHop` requires `torch_geometric`.')
        # todo add mutate
    def __call__(self, data):
        x, edge_index, y = data.x.cpu(), data.edge_index.cpu(), data.y.cpu()
        node_mask = torch.cat([torch.ones(x.size(0)), torch.zeros(edge_index.size(1))], dim=0).bool()
        new_x = torch.cat([x, x[edge_index[0]]], dim=0)
        new_y = torch.cat([y, y[edge_index[0]]], dim=0)
        edge_node_ids = torch.arange(edge_index.size(1)) + data.num_nodes
        new_edge_index = [torch.stack([edge_index[0], edge_node_ids]),
                          torch.stack([edge_node_ids, edge_index[1]]),
                          torch.stack([edge_index[1], edge_node_ids])]
        new_edge_index = torch.cat(new_edge_index, dim=1)
        data.x, data.edge_index, data.y, data.node_mask = new_x.to(data.x.device), new_edge_index.to(data.x.device), new_y.to(data.x.device), node_mask.to(data.x.device)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class DropRandomWalks:
    def __init__(self, num_random_walks, walk_length):
        if torch_geometric is None:
            raise ImportError('`DropRandomWalks` requires `torch_geometric`.')

        self.num_random_walks = num_random_walks
        self.walk_length = walk_length

    def __call__(self, data):
        N = data.num_nodes
        E = data.num_edges
        edge_index = data.edge_index.cpu()
        adj = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            value=torch.arange(E, device=edge_index.device),
            sparse_sizes=(N, N))
        # seed random walks
        start = torch.randint(0, N, (self.num_random_walks,), dtype=torch.long)
        # random walk
        node_idx = adj.random_walk(start.flatten(), self.walk_length)
        # get random walk edges
        node_idx = node_idx.view(self.walk_length + 1, -1)
        rw_edge_index = torch.stack([node_idx[:-1], node_idx[1:]], dim=-1).view(-1, 2).T
        # remove edges
        edge_index_np, rw_edge_index_np = np.ascontiguousarray(edge_index.numpy().T), np.ascontiguousarray(rw_edge_index.numpy().T)
        edge_index_set = edge_index_np.view([('', edge_index_np.dtype)] * 2)
        rw_edge_index_set = rw_edge_index_np.view([('', rw_edge_index_np.dtype)] * 2)

        set_diff = np.setdiff1d(edge_index_set, rw_edge_index_set).view(edge_index_np.dtype).reshape(-1, 2)
        row, col = set_diff.T
        new_edge_index = torch.LongTensor([row, col])
        new_edge_index, _ = torch_geometric.utils.sort_edge_index(new_edge_index)  # sort
        data.edge_index = new_edge_index.to(data.edge_index.device)
        return data


class CropTransformation:
    # todo for now assert one graph
    def __init__(self, type, drop_edge_p, drop_feat_p, num_random_walks=None, walk_length=None):
        if torch_geometric is None:
            raise ImportError('`CropTransformation` requires `torch_geometric`.')

        assert type in ['two_hop', 'half_hop', 'drop_random_walk']
        if type == 'two_hop':
            self.p_apply = 0.5
            self.geometric_transform = TwoHop()
        elif type == 'half_hop':
            self.p_apply = 0.5
            self.geometric_transform = HalfHop()
        elif type == 'drop_random_walk':
            self.p_apply = 1.0
            assert num_random_walks is not None and walk_length is not None
            self.geometric_transform = DropRandomWalks(num_random_walks, walk_length)

        self.drop_transform = get_graph_drop_transform(drop_edge_p, drop_feat_p)

    def __call__(self, data):
        data = self.drop_transform(data)
        if torch.rand(1) < self.p_apply:
            data = self.geometric_transform(data)
        return data
