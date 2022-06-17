import torch
import torch.nn as nn

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, BatchNorm
except:
    torch_geometric = None


from .sequential import Sequential


class GraphEncoder(torch.nn.Module):
    def __init__(self, layer_sizes, activation='prelu', batchnorm=True, batchnorm_mm=0.99):
        if torch_geometric is None:
            raise ImportError('`GraphEncoder` requires `torch_geometric`.')

        super().__init__()
        activation = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}[activation]

        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'),)
            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            if activation is not None:
                layers.append(activation)

        self.model = Sequential('x, edge_index', layers)

        # weight initialization
        # GCN already uses Glorot initialization

    def forward(self, data):
        # todo check GCN is using edge_index
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        # GCN uses Glorot initalization
        self.model.reset_parameters()
