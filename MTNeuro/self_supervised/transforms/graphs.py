import copy

import torch

try:
    import torch_geometric
    from torch_geometric.utils import dropout_adj
    from torch_geometric.transforms import Compose
    from torch_geometric.data import Batch
except:
    torch_geometric = None

from MTNeuro.self_supervised.utils import NotTestedError


class DropFeatures:
    def __init__(self, p, same_on_graph=True, same_on_batch=False):
        if torch_geometric is None:
            raise ImportError('`DropFeatures` requires `torch_geometric`.')

        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

        assert not(same_on_batch) or (same_on_batch and same_on_graph)
        self.same_on_graph = same_on_graph
        self.same_on_batch = same_on_batch

    def __call__(self, data):
        if 'batch' in data:  # todo isinstance(Batch)?
            data_list = data.to_data_list()
        else:
            data_list = [data]

        drop_mask = None
        for data in data_list:
            x = data.x
            if self.same_on_batch:
                # todo no need to batch
                if drop_mask is None:
                    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                x[:, drop_mask] = 0
            elif self.same_on_graph:
                drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                x[:, drop_mask] = 0
            else:
                drop_mask = torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                x[drop_mask] = 0
                raise NotTestedError
            data.x = x

        if 'batch' in data:
            return Batch.from_data_list(data_list)
        else:
            return data

    def __repr__(self):
        return '{}(p={}, same_on_graph={}, same_on_batch={})'.format(self.__class__.__name__, self.p,
                                                                     self.same_on_graph, self.same_on_batch)


class DropEdges:
    def __init__(self, p, force_undirected=False):
        if torch_geometric is None:
            raise ImportError('`DropEdges` requires `torch_geometric`.')

        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)


class CopyGraph:
    # todo probably a better way of doing this
    def __init__(self):
        pass

    def __call__(self, data):
        # clone should also work
        return copy.deepcopy(data)

    def __repr__(self):
        return self.__class__.__name__


def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    if torch_geometric is None:
        raise ImportError('`get_graph_drop_transform` requires `torch_geometric`.')

    transforms = [CopyGraph()]
    if drop_edge_p != 0.:
        transforms.append(DropEdges(drop_edge_p))
    if drop_feat_p != 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)


class GenerateViews:
    def __init__(self, *transforms):
        self.transforms = transforms

    @staticmethod
    def prepare_views(inputs):
        data_1, data_2 = inputs
        outputs = {'view_1': data_1, 'view_2': data_2}
        return outputs

    def __call__(self,data):
        return tuple([t(data) for t in self.transforms])
