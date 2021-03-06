__all__ = ['UMAPAnalysis']


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tdavis.data_processing import *
import numpy as np
import umap


class UMAPAnalysis:
    def __init__(self, data: np.array, n_neighbors=10, min_dist=0.5,
                 n_components=2, metric='correlation', filter_lowest=0.2,
                 window_size=5, method="avg", differenced=True, **kwargs):
        """
        :param data: a TxN matrix where T is the time dim and N is the variable
        dimension.
        """
        self._n_components = n_components
        self._data = data
        self._preprocessed_data = bin_time_series(
            pre_process_layer_states(self._data, filter_lowest, differenced),
            window_size, method)

        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                 n_components=n_components, metric=metric)
        self.reducer.fit(self._preprocessed_data)

    def visualize(self, filename=None, png=False, title='UMAP projection'):
        embedding = self.reducer.transform(self._preprocessed_data)

        fig = plt.figure()
        if self._n_components == 1:
            ax = fig.add_subplot(111)
            im = ax.scatter(embedding[:, 0], range(len(embedding)),
                       c=cm.viridis(
                           np.linspace(0, 1, self._preprocessed_data.shape[0])))
            ax.plot(embedding[:, 0], range(len(embedding)))
        if self._n_components == 2:
            ax = fig.add_subplot(111)
            im = ax.scatter(embedding[:, 0], embedding[:, 1],
                       c=cm.viridis(
                           np.linspace(0, 1, self._preprocessed_data.shape[0])))
            ax.plot(embedding[:, 0], embedding[:, 1], color='black',
                    ls='dotted', lw=0.5)
        if self._n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            im = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                       c=cm.viridis(
                           np.linspace(0, 1, self._preprocessed_data.shape[0])),
                       s=100)

        fig.colorbar(im, ax=ax, label='time')
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(title, fontsize=20)
        if filename is None:
            plt.show()
        else:
            if png:
                plt.savefig(filename, dpi=300)
            else:
                plt.savefig(filename)
        plt.close()
        plt.clf()
