import matplotlib.pyplot as plt
import numpy as np


def plot_2d_clusters(data, clusters, ax, **kwarg):
    r"""Project labelled data into 3D as colored clusters.
    Parameters
    ----------
    data : array-like, shape (n_points, n_features)
        Point coordinates, of arbitrary dimension. (n_features >= 2)
    clusters : array-like, shape (n_points)
        Point cluster labels.
    ax : matplotlib Axes3D object
        Axis to plot on.
    labels : list, shape (n_clusters)
        Names of the clusters in numerical order, for plot legend.

    keyword arguments :
        Any valid keyword arguments to ax.scatter().
    Raises
    ------
    ValueError : If labels are provided, but the number does not match the number of unique labels in clusters.
    """
    labels = kwarg.pop('labels', [''] * len(np.unique(clusters)))
    if len(labels) != len(np.unique(clusters)):
        raise ValueError('If providing labels, one for every cluster is required. (Provide the empty string \'\' for clusters meant to be unlabelled.)')
    for i, cluster in enumerate(np.unique(clusters)):
        ids = clusters == cluster
        ax.scatter(data[ids, 0], data[ids, 1], s=250, alpha=0.5, marker='.', label=labels[i], **kwarg)