import logging

import torch
try:
    from torch_scatter import scatter_add, scatter_max, scatter_min
except:
    scatter_add = scatter_max = scatter_min = None

from MTNeuro.self_supervised.utils import you_only_log_once

log = logging.getLogger(__name__)
yolo = you_only_log_once()


def dense_topk(dist, k=1, descending=False):
    # compute k nearest neighbors
    _, topk_index = torch.topk(dist, k=k, largest=descending)
    random_select = torch.randint(k, size=(topk_index.size(0),))  # todo when there is less than k

    mined_view_id = topk_index[torch.arange(topk_index.size(0), dtype=torch.long, device=dist.device), random_select]
    mined_dist = dist[torch.arange(dist.size(0), dtype=torch.long, device=dist.device), mined_view_id]
    return mined_view_id, mined_dist


def sparse_topk(dist, source, target, k=2, descending=False):
    if scatter_add is None:
        raise ImportError('`sparse_topk` requires `torch_scatter`.')
    r"""Based on Matthias Fey's implementation of topk pooling.
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/topk_pool.html#TopKPooling
    """
    # sort source (and target accordingly)
    # index = torch.stack([source, target])
    # (source, target), _ = sort_edge_index(index) # todo does not need to be sorted
    fill_value = getattr(torch.finfo(dist.dtype), 'min' if descending else 'max')

    # build dense distance matrix that is still partially sparse
    num_neighs = scatter_add(source.new_ones(source.size(0)), source, dim=0)
    # todo what happens to nodes that have no neighbors assert num_nodes == source.max() + 1
    num_nodes, max_num_neighs = num_neighs.size(0), num_neighs.max().item()

    with yolo(log.isEnabledFor(logging.DEBUG)) as go_ahead:
        log.debug('Smallest neighborhood: %d' % num_neighs.min().item(), extra={"markup": True}) if go_ahead else None

    cum_num_neighs = torch.cat([num_neighs.new_zeros(1), num_neighs.cumsum(dim=0)[:-1]], dim=0)
    index = torch.arange(source.size(0), dtype=torch.long, device=dist.device)
    index = (index - cum_num_neighs[source]) + (source * max_num_neighs)

    dense_dist = dist.new_full((num_nodes * max_num_neighs,), fill_value=fill_value)  # new_full is for same device
                                                                                      # and same dtype
    dense_dist[index] = dist
    dense_dist = dense_dist.view(num_nodes, max_num_neighs)

    _, topk_index = dense_dist.sort(dim=-1, descending=descending)

    topk_index = topk_index + cum_num_neighs.view(-1, 1)
    topk_index = topk_index.view(-1)

    k = num_neighs.new_full((num_neighs.size(0),), k)
    k = torch.min(k, num_neighs)

    # random select
    mask = [torch.randint(k[i], (1,), dtype=torch.long, device=dist.device)
            + i * max_num_neighs for i in range(num_nodes)]
    mask = torch.cat(mask, dim=0)

    topk_index = topk_index[mask]
    mined_view_id = target[topk_index]
    mined_dist = dist[topk_index]
    return mined_view_id, mined_dist


def sparse_top1(dist, source, target, descending=False):
    if scatter_max is None:
        raise ImportError('`sparse_top1` requires `torch_scatter`.')
    if descending:
        mined_dist, argmax = scatter_max(dist, source, dim=-1)
    else:
        mined_dist, argmax = scatter_min(dist, source, dim=-1)
    mined_view_id = target[argmax]
    return mined_view_id, mined_dist

"""
Voting in knn
    y_knn = y[indices]
    dist = self._compute_distance(y_knn, y_knn)
    vote_vect = - dist.sum(dim=1)  # - gamma * values #todo figure this out
    selection_mask = torch.max(vote_vect, 1)[1]
"""


def sample(edge_index, num_neighbors=1):
    row, col = edge_index

    num_neighs = scatter_add(row.new_ones(row.size(0)), row, dim=0)
    cum_num_neighs = torch.cat([num_neighs.new_zeros(1), num_neighs.cumsum(dim=0)[:-1]], dim=0)
    torch.multinomial()
    rand = torch.rand((num_neighs.size(0), num_neighbors), device=col.device)
    rand.mul_(num_neighs.to(rand.dtype).view(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(cum_num_neighs.view(-1, 1))

    return edge_index[rand]