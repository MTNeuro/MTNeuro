import logging

from scipy.sparse import coo_matrix
import numpy as np
import torch

try:
    import torch_geometric
except ImportError:
    torch_geometric = None

from MTNeuro.self_supervised.utils.console import console

log = logging.getLogger(__name__)


def neigh_annulus(edge_index, r, R, use_sparse=True, device='cpu'):
    if torch_geometric is None:
        raise ImportError('`neigh_annulus` requires `torch_geometric`.')

    with console.status("[bold green]Building neighborhood annulus...") as status:
        log.debug('[bold orange]Neighbor annulus[/bold orange] Converting edge_index to sparse matrix.',
                  extra={"markup": True})
        adj_matrix = torch_geometric.utils.to_scipy_sparse_matrix(edge_index).astype(bool)

        if use_sparse:
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
            annulus_mask = torch.sparse_coo_tensor(torch.LongTensor([row, col]), torch.ones(len(row), dtype=torch.bool), device=device)
            log.debug('[bold orange]Neighbor annulus[/bold orange] Sparse tensor\n%r' %annulus_mask,
                      extra={"markup": True})
        else:
            raise NotImplementedError
            adj_matrix = adj_matrix.toarray()

            log.info('Computing r.')
            k_hop_adj_matrix = np.eye(*adj_matrix.shape, dtype=bool)
            less_than_r_hops_away = np.zeros(adj_matrix.shape, dtype=bool)  # empty
            for k in range(r):
                less_than_r_hops_away += k_hop_adj_matrix
                k_hop_adj_matrix = k_hop_adj_matrix @ adj_matrix

            log.info('Computing R.')
            less_than_R_hops_away = np.zeros(k_hop_adj_matrix.shape, dtype=bool)  # empty
            for k in range(r, R):
                less_than_R_hops_away += k_hop_adj_matrix
                k_hop_adj_matrix = k_hop_adj_matrix @ adj_matrix
            less_than_R_hops_away += k_hop_adj_matrix

            annulus_mask = less_than_R_hops_away * ~less_than_r_hops_away
            # convert to torch sparse
            # this is fixed so copy it to gpu once and for all
            log.info('Converting to torch sparse.')
            annulus_mask = torch.Tensor(annulus_mask, dtype=torch.bool, device=device)

    return annulus_mask
