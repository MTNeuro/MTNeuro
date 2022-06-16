import torch
import torch.nn as nn

try:
    import torch_geometric
    from torch_geometric.nn import SAGEConv, LayerNorm
except:
    torch_geometric = None


class SAGE_GCN(torch.nn.Module):
    # uses residual skip connections

    def __init__(self, input_size, hidden_size, embedding_size, activation='prelu', layernorm=True):
        if torch_geometric is None:
            raise ImportError('`SAGE_GCN` requires `torch_geometric`.')

        super().__init__()

        assert layernorm

        self.convs = torch.nn.ModuleList([
            SAGEConv(input_size, hidden_size),
            SAGEConv(hidden_size, hidden_size),
            SAGEConv(hidden_size, embedding_size)
        ])

        self.layer_norms = torch.nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size)
        ])

        self.activation = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}[activation]
        self.reset_parameters()  # Glorot initialization

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = x
        residual = 0  # x has a different shape from h
        for i in range(3):
            h = residual + self.convs[i](h, edge_index)
            h = self.layer_norms[i](h)
            h = self.activation(h)
            residual = h
        return h

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
