import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


def circular_visualization(x_1, x_2, labels_1, labels_2):
    N = 128
    max_height = 4
    x = torch.cat([x_1, x_2])
    labels = torch.cat([labels_1, labels_2])

    x = x.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(x)
    x_theta = x_2d / np.linalg.norm(x_2d)
    x_circular = x_2d / np.linalg.norm(x_2d, axis=1, keepdims=True)
    theta = np.arctan2(x_circular[:, 1], x_circular[:, 0])

    class_hist = []
    for label in range(10):
        hist, _ = np.histogram(theta[labels==label], bins=N, range=[-np.pi, np.pi])
        class_hist.append(hist)

    theta_grid = np.linspace(-np.pi, np.pi, N, endpoint=False)
    width = (2*np.pi) / N
    cum_bottom = 40 * np.ones(N, dtype=np.float32)

    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    cmap = plt.get_cmap("tab10")
    for i in range(10):
        radii = max_height * class_hist[i]
        bars = ax.bar(theta_grid, radii, width=width, bottom=cum_bottom, color=cmap(i), alpha=0.8, zorder=10)
        cum_bottom += class_hist[i]

    t = np.vstack([theta[:x_1.shape[0]], theta[x_1.shape[0]:]]).T
    r = 40 * np.ones(t.shape)

    wrong_matches = (labels_1 != labels_2).detach().cpu().numpy()
    ax.plot(t[wrong_matches], r[wrong_matches], color='red', linewidth=1, alpha=0.2, zorder=3)
    ax.plot(t[~wrong_matches], r[~wrong_matches], color='dodgerblue', linewidth=1, alpha=0.2, zorder=3)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_rgrids([0, 40, 45, 50, 55, 60])
    ax.set_thetagrids([])
    return fig

    #ax.set_rgrids([0, 40])
    #ax.set_rticks([0, 40])
    #ax.grid(True)
    # Use custom colors and opacity
    # for r, bar in zip(radii, bars):
    #    bar.set_facecolor(plt.cm.jet(r / 10.))
    #    bar.set_alpha(0.8)

    # plt.show()
