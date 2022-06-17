from torch import nn


class MLP3(nn.Module):
    r"""MLP class used for projector and predictor in :class:`BYOLBase`. The MLP has one hidden layer.

    .. note::
        The hidden layer should be larger than both input and output layers, according to the
        :class:`BYOL` paper.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features (projection or prediction).
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=4096, init="he_uniform", batchnorm_mm=0.1):
        super().__init__()
        self.init = init
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size, eps=1e-5, momentum=batchnorm_mm),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size, bias=False)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init == 'he_uniform':
                    m.reset_parameters()
                elif self.init == 'glorot_normal':  # todo change to glorot_uniform
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        # todo verify glorot uniform and init bias
                        nn.init.zeros_(m.bias)
                else:
                    raise NotImplementedError

    def forward(self, x):
        return self.net(x)
