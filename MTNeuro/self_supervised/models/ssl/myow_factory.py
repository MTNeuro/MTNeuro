import torch
import torch.nn.functional as F

from .ops.knn_ops import dense_topk, sparse_top1, sparse_topk
from .ops.ot_ops import dense_ot


def myow_factory(byol_class):
    r"""Factory function for adding mining feature to an architecture."""
    class MYOW(byol_class):
        r"""
        Class that adds ability to mine views to base class :obj:`byol_class`.
        Supports both knn and ot.

        Args:
            n_neighbors (int, optional): Number of neighbors used in knn. (default: :obj:`1`)
        """
        def __init__(self, *args, distance='cosine', view_miner='knn', n_neighbors=1, select_neigh='random',
                     gamma=1e0, niters=10, **kwargs):
            super().__init__(*args, **kwargs)
            assert distance in ['cosine', 'l2']
            self.__distance = distance
            assert view_miner in ['knn', 'ot']
            self.__method = view_miner

            # knn
            self.__k = n_neighbors
            assert select_neigh in ['random', 'vote']
            self.__select_neigh = select_neigh

            # ot
            self.__gamma = gamma
            self.__niters = niters

        def _compute_distance(self, x, y):
            if self.__distance == 'cosine':
                x = F.normalize(x, dim=-1, p=2)
                y = F.normalize(y, dim=-1, p=2)

                dist = 2 - 2 * torch.sum(x.view(x.shape[0], 1, x.shape[1]) *
                                         y.view(1, y.shape[0], y.shape[1]), -1)
            elif self.__distance == 'l2':
                dist = torch.sum(torch.pow(x.view(x.shape[0], 1, x.shape[1]) -
                                           y.view(1, y.shape[0], y.shape[1]), 2), -1)
            return dist

        def _compute_sparse_distance(self, x, y, edge_index):
            row, col = edge_index
            if self.__distance == 'cosine':
                dist = 2 - 2 * F.cosine_similarity(y[col], x[row], dim=-1)
            elif self.__distance == 'l2':
                dist = torch.norm(y[col] - x[row], p=2, dim=-1).view(-1, 1)
            return dist

        def mine_views(self, y, y_pool, cand_edge_index=None, ccand_edge_index=None):
            r"""Finds, for each element in batch :obj:`y`, its nearest neighbors in :obj:`y_pool`, randomly selects one
                of them and returns the corresponding index.

            Args:
                y (torch.Tensor): batch of representation vectors.
                y_pool (torch.Tensor): pool of candidate representation vectors.

            Returns:
                torch.Tensor: Indices of mined views in :obj:`y_pool`.
            """
            assert (cand_edge_index is None and ccand_edge_index is None) or \
                   (cand_edge_index is None) != (ccand_edge_index is None)
            sparse = cand_edge_index is not None

            # compute distance
            if not sparse:
                dist = self._compute_distance(y, y_pool)
                if ccand_edge_index is not None:
                    n_mask = torch.unique(ccand_edge_index[0])
                    n_idx = torch.zeros(y_pool.size(0), dtype=torch.long)
                    n_idx[n_mask] = torch.arange(n_mask.size(0))
                    dist[n_idx[ccand_edge_index[0]], ccand_edge_index[1]] = torch.finfo(dist.dtype).max  # todo ot
            else:
                # todo make sure it's undirected
                # todo assert not data.contains_isolated_nodes()
                # todo assert not data.contains_self_loops()
                dist = self._compute_sparse_distance(y, y_pool, cand_edge_index)

            # mine views
            if self.__method == 'knn':
                if not sparse:
                    mined_view_id, mined_dist = dense_topk(dist, k=self.__k)
                else:
                    source, target = cand_edge_index
                    if self.__k == 1:
                        mined_view_id, mined_dist = sparse_top1(dist, source, target)
                    else:
                        mined_view_id, mined_dist = sparse_topk(dist, source, target, self.__k)

            elif self.__method == 'ot':
                # todo update ot
                raise NotImplementedError
                mined_view_id, mined_dist = dense_ot(dist, gamma=self.__gamma, niters=self.__niters)
            return mined_view_id, mined_dist
    return MYOW
