from collections import namedtuple

import torch


class NamedTupleWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        data = self.model(input)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)
        elif isinstance(data, list):
            data = tuple(data)

        return data
