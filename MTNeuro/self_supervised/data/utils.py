import numpy as np
import torch


def diagu_indices(n, k_min=1, k_max=None):
    if k_max is None:
        return np.array(np.triu_indices(n, 1)).T
    else:
        all_pairs = set(zip(*map(np.ndarray.tolist, np.triu_indices(n, k_min))))
        rm_pairs = set(zip(*map(np.ndarray.tolist, np.triu_indices(n, 1 + k_max))))
        pairs = all_pairs - rm_pairs
        return np.array(list(pairs))


def onlywithin_indices(sequence_lengths, k_min=1, k_max=None):
    cum_n = 0
    pair_arrays = []
    for i, n in enumerate(sequence_lengths):
        pairs = diagu_indices(n, k_min, k_max) + cum_n
        pair_arrays.append(np.hstack([np.ones((pairs.shape[0], 1))*i, pairs]))
        cum_n += n
    return np.concatenate(pair_arrays).astype(int)


def batch_iter(X, *tensors, batch_size=256):
    # todo add drop_last
    # todo add shuffle flag
    r"""Creates iterator over tensors.

    Args:
        X (torch.tensor): Feature tensor (shape: num_instances x num_features).
        tensors (torch.tensor): Target tensors (shape: num_instances).
        batch_size (int, Optional): Batch size. (default: :obj:`256`)
    """
    idxs = torch.randperm(X.size(0))
    if X.is_cuda:
         idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        res = [X[batch_idxs]]
        for tensor in tensors:
            res.append(tensor[batch_idxs])
        yield res


def prepare_views(inputs):
    r"""Default input dict for most common image dataset, where a sample is (img, label) and where all images are the
    same size and thus it becomes possible to perform transformation on gpu in batch mode.

    Example: CIFAR, MNIST
    """
    x, labels = inputs
    outputs = {'view_1': x, 'view_2': 'view_1', 'metadata_m': {'labels': labels}}
    return outputs


def prepare_graph_views(data):
    r"""Default input dict for most common image dataset, where a sample is (img, label) and where all images are the
    same size and thus it becomes possible to perform transformation on gpu in batch mode.

    Example: CIFAR, MNIST
    """
    outputs = {'view_1': data, 'view_2': 'view_1'}
    return outputs


def prepare_graph_views_for_myow(inputs):
    data, restriction_mask = inputs
    outputs = {'view_1': data, 'view_2': 'view_1', 'view_m': 'view_1', 'view_pool': None,
               'cand_edge_index': restriction_mask, 'metadata_m': {'labels': 'data.y'},
               'metadata_pool': {'labels': 'data.y'}}
    return outputs
