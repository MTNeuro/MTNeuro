from functools import wraps

from sklearn import metrics
import torch
import numpy as np
from scipy import stats


def run_once_property(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            instance = getattr(self, '_' + fn.__name__)
            return instance
        except AttributeError:
            instance = fn(self, *args, **kwargs)
            setattr(self, '_' + fn.__name__, instance)
            return instance
    return property(wrapper)


class LatentSpace:
    r"""Python class for performing different computations over the latent space, guarantees each computation is
     done at most once.
     Args:
         x (torch.Tensor or np.array): Representation batch (batch_size, representation_size)
         labels (torch.Tensor or np.array): Labels (batch_size)
         num_neighbors (int, Optional): Num of neighbors used in different computations, including k-nearest neighbors.

    ..Example::
        ```
        >>> imgs, labels = next(iter(DataLoader(dataset, batch_size=1024, shuffle=True,
            worker_init_fn=lambda: np.random.seed(5)))) ##used to make sure it's the same samples everytime.
        >>> latent_space = LatentSpace.forward_imgs_and_build_latent_space(online_encoder, imgs, labels, num_neighbors=8)
        >>> lsl = LatentSpaceLogger(latent_space, step, writer)
        >>> lsl.log_silhouette_histogram()
        >>> lsl.log_knn_accuracy()
        ```
     """
    def __init__(self, x, labels, num_neighbors=8):
        self.x = self._convert_to_numpy(x)
        self.labels = self._convert_to_numpy(labels)
        self.num_neighbors = num_neighbors

    @run_once_property
    def distance_matrix(self):
        r"""Builds the distance matrix between each pair of samples."""
        dist = np.sum(np.power(self.x.reshape(self.x.shape[0], 1, self.x.shape[1]) -
                               self.x.reshape(1, self.x.shape[0], self.x.shape[1]), 2), -1)
        return dist

    @run_once_property
    def nearest_neighbors(self):
        r"""Returns for each sample, the indices of its nearest neighbors."""
        dist = self.distance_matrix + np.eye(self.distance_matrix.shape[0]) * self.distance_matrix.max()
        neigh_indices = np.argpartition(dist, self.num_neighbors)[:, :self.num_neighbors]
        return neigh_indices

    @run_once_property
    def pred_labels(self):
        r"""Returns predicted label based on the labels of the nearest neighbors."""
        neigh_labels = self.labels[self.nearest_neighbors]
        pred_labels = stats.mode(neigh_labels, 1)[0].squeeze()
        return pred_labels

    def _convert_to_numpy(self, x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    @staticmethod
    def forward_imgs_and_build_latent_space(nets, imgs, labels, num_neighbors=8):
        r"""Feeds :obj:`imgs` to :obj:`net` to get representations and builds the :class:`LatentSpace` object.

        Args:
            net (torch.nn.Module): Encoder
            imgs (torch.Tensor): Batch of images (batch_size, C, H, W)
            labels (torch.Tensor): Labels (batch_size)
            num_neighbors (int, Optional): Num of neighbors used in different computations,
                including k-nearest neighbors.
        """
        nets = [nets] if not isinstance(nets, list) else nets
        x = imgs
        for net in nets:
            x = net(x)
            x = x.view(x.shape[0], -1)
        x = x.detach()
        return LatentSpace(x, labels, num_neighbors)


class LatentSpaceLogger:
    r"""Python object for computing different metrics and logging plots to tensorboard.
    Args:
        latentspace (LatentSpace): Computation tracker.
        step (int): Current training step.
        writer (torch.tensorboard): Tensorboard writer.
        comment (string, Optional): suffix to add to tag in tensorboard.
    """
    def __init__(self, latentspace, step, writer, comment=''):
        self.ls = latentspace
        self.step = step
        self.writer = writer
        self.comment = comment

    def log_calinski_harabasz_score(self):
        chs = metrics.calinski_harabasz_score(self.ls.x, self.ls.labels)
        self.writer.add_scalar('space_metrics/calinski_harabasz_score' + self.comment, chs, self.step)

    def log_knn_accuracy(self):
        acc = np.sum(np.equal(self.ls.labels, self.ls.pred_labels)) * 1. / self.ls.labels.shape[0]
        self.writer.add_scalar('space_metrics/knn_acc'+ self.comment, acc, self.step)

    def log_adjusted_rand_score(self):
        ari = metrics.adjusted_rand_score(self.ls.labels, self.ls.pred_labels)
        self.writer.add_scalar('space_metrics/adjusted_rand_score'+ self.comment, ari, self.step)

    def log_silhouette_histogram(self):
        ss = metrics.silhouette_samples(self.ls.distance_matrix, self.ls.labels, metric="precomputed")
        self.writer.add_histogram('space_metrics/silhouette'+ self.comment, ss, self.step)

    def log_silhouette_score(self):
        ss = metrics.silhouette_score(self.ls.distance_matrix, self.ls.labels, metric="precomputed")
        self.writer.add_scalar('space_metrics/silhouette_score'+ self.comment, ss, self.step)

    def log_distance_hist(self):
        dist = self.ls.distance_matrix[np.triu_indices(self.ls.distance_matrix.shape[0], 1)]
        self.writer.add_histogram('space_metrics/distance_distribution'+ self.comment, dist, self.step)

    def log_variability_latent_factor(self, **latent_factors):
        for latent_factor_tag, latent_factor in latent_factors.items():
            std = (latent_factor[self.ls.nearest_neighbors] - latent_factor[:, np.newaxis]).std(axis=1)
            self.writer.add_histogram('std_latent_var_neighboring_points/'+latent_factor_tag+ self.comment, std, self.step)


# num_samples = self.lst.distance_matrix.shape[0]
# num_neighbors = self.lst.nearest_neighbors.shape[1]
# sub_distance = self.lst.distance_matrix[np.repeat(np.arange(num_samples)[:, np.newaxis], num_neighbors, axis=1),
#                                        self.lst.nearest_neighbors]
