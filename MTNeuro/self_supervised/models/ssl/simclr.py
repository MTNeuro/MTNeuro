import torch


class SimCLR(torch.nn.Module):
    r"""
    Args:
        encoder (torch.nn.Module): Encoder network.
        projector (torch.nn.Module): Projector network.
    """
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    @property
    def trainable_module_list(self):
        return [self.encoder, self.projector]

    def forward(self, x, x_positive, accumulate=False):
        query_encoder = self.encoder
        key_encoder = self.encoder

        queries = self.encoder(x)
        queries = queries.reshape(queries.shape[0], -1)
        queries = self.projector(queries)

        with torch.no_grad():
            keys = self.encoder(x_positive).detach().clone()
            keys = keys.reshape(keys.shape[0], -1)
            keys = self.projector(keys)

        return queries, keys # .calculate_loss() if not accumulate else None
