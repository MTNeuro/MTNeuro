import matplotlib.pyplot as plt
import numpy as np


def show_tranport_plan(transport_plan, meta_1, meta_2):
    def sort_meta(meta):
        dtype = [('sample_id', int), ('reach', float), ('sequence', float), ('time', float)]
        m = np.hstack([np.arange(meta.shape[0])[:, np.newaxis], meta.numpy()])
        m = np.array(list(map(tuple, m.tolist())), dtype=dtype)
        sorted_m = np.sort(m, order=['reach', 'sequence', 'time'])
        return np.array([sorted_m[i][0] for i in range(len(sorted_m))])

    matrix = transport_plan.detach().numpy()
    matrix = matrix[sort_meta(meta_1)][:, sort_meta(meta_2)]
    plt.imshow(matrix, cmap='hot')
