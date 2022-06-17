import torch

try:
    import torch_geometric
    from torch_geometric.utils import dense_to_sparse
except:
    torch_geometric = None

from .relative_positioning_dataloader import RelativePositioningSampler


class RelativeSequenceSampler(RelativePositioningSampler):
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True,
                 pos_kmin=0, pos_kmax=0, transform=None, **kwargs):
        super().__init__(dataset, batch_size, drop_last=drop_last, shuffle=shuffle,
                       pos_kmin=pos_kmin, pos_kmax=pos_kmax, neg_k=None, transform=transform, **kwargs)

    def _add_pos_cand_edge_indices(self, data):
        n = data.num_nodes
        # positive edge index: default self only
        data.pos_edge_index = self._block_diag_indices(n, self.pos_kmin, self.pos_kmax)
        # complementary negative edge index
        data.ccand_edge_index, _ = dense_to_sparse(torch.ones((n, n)))
        return data
