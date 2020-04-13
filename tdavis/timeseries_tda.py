__all__ = ['calc_persistence_diagrams',
           'persistence_loss', 'mc_randomization', 'distribution_plot',
           'pvalue_plot', 'distance_plot', 'plot_umap']


from typing import Union, Dict, Sequence
from ripser import ripser
from persim import plot_diagrams
from persim import sliced_wasserstein
import seaborn
import umap
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import math


def calc_persistence_diagrams(x: np.ndarray, maxdim=2, n_perm=None,
                              metric="manhattan", full_out=False,
                              **kwargs) -> Union[Dict, np.ndarray]:
    """
    :param x: TxN
    :param maxdim: maximum number of dimensions of simplex (computes all lower
    or equal to value) so maxdim=1 computes H0 and H1.
    :param n_perm: number of permutations of data to use for subsampling. It
    cuts out points that are fully within the filtration.
    :param metric: the distance metric to use for the VR complex filtration
     on the data.
    :param full_out: whether to return only the diagrams or all of Ripser's
     output.
    :param kwargs: any other arguments taken by ripser.
    :return: list of persistence diagrams
    """
    if full_out:
        return ripser(x, maxdim=maxdim, n_perm=n_perm, metric=metric,
                      **kwargs)
    else:
        return ripser(x, maxdim=maxdim, n_perm=n_perm, metric=metric,
                      **kwargs)['dgms']


def persistence_loss(distance_matrix: np.ndarray, *groups: Sequence,
                     q=1, as_mean=False) -> float:
    """
    Calculates the p=q=pq loss. If p=q=2 this is the sum of the variances
    of the distances within groups.

    See: Hypothesis Testing for Topological Data Analysis equation (4)

    :param distance_matrix: a square matrix of distances between all diagrams.
    :param groups: a tuple of indices specifying which diagrams belong to
    which groups.
    :param q: should be set to the same value as the power for the distance.
    So for manhattan q=1, and for euclidean q=2.
    :param as_mean: whether to use the mean of the variances rather than the sum
    :return: the loss of the group configuration
    """
    statistic = np.sum
    if as_mean:
        statistic = np.mean

    loss = []
    for indices in groups:
        loss.append(statistic(np.power(distance_matrix[indices][:, indices], q),
                              axis=None)
                    / (2 * len(indices) * (len(indices) - 1)))
    return sum(loss)


def mc_randomization(observed_loss: float,
                     distance_matrix: np.ndarray,
                     num_permutations: int,
                     *group_sizes: int):
    """
    Uses monte-carlo approach to estimate the p-value of the hypothesis that
    the groups are the same.
    :param observed_loss: the loss observed in the correct grouping.
    :param distance_matrix: a square matrix of distances between all diagrams.
    :param num_permutations: the number of random permutations of the groupings
    to consider.
    :param group_sizes: the respective sizes of each grouping.
    :return:
    """
    group_sizes = [0] + list(group_sizes)
    indices = np.array(range(distance_matrix.shape[0]))
    z = 1
    null_losses = []
    for _ in range(num_permutations-1):
        np.random.shuffle(indices)
        groups = [indices[group_sizes[i-1]:group_sizes[i]]
                  for i in range(1, len(group_sizes))]
        loss = persistence_loss(distance_matrix, *groups)
        null_losses.append(loss)
        if loss <= observed_loss:
            z += 1
    z /= (num_permutations + 1)
    return z, null_losses


def distribution_plot(distances, *groups, prefix="test", as_png=False,
                      **hist_kwargs):
    hst_set = []
    for group in groups:
        hst_set.append(np.tril(distances[group][:, group]).flatten())
    plt.hist(hst_set, **hist_kwargs)
    if as_png:
        plt.savefig(prefix + ".png", dpi=300)
    else:
        plt.savefig(prefix + ".pdf")
    plt.close()
    plt.clf()


def pvalue_plot(obs_loss, p, loss_distribution,
                prefix="test", as_png=False, **hist_args):
    plt.hist(loss_distribution, **hist_args)
    plt.axvline(obs_loss)
    plt.title("p = " + str(p))
    if as_png:
        plt.savefig(prefix + ".png", dpi=300)
    else:
        plt.savefig(prefix + ".pdf")
    plt.close()
    plt.clf()


def distance_plot(distance_matrix, prefix="test", as_png=False):
    ax = seaborn.heatmap(distance_matrix, vmin=0.0, square=True,
                         cmap='inferno', xticklabels=True,
                         yticklabels=True, cbar=True)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.5, xlim[1] + 0.5)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] + 0.5, ylim[1] - 0.5)
    bottle_fig = ax.get_figure()
    if as_png:
        bottle_fig.savefig(prefix + ".png", dpi=300)
    else:
        bottle_fig.savefig(prefix + ".pdf")
    plt.close()
    bottle_fig.clf()


def plot_umap(data, *groups, prefix="test", n_neighbors=10, min_dist=0.5,
              metric='manhattan', as_png=True):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric
    )
    # TODO: cmap for distinct elements
    color_sequence = ["blue", "red"]
    colors = []
    for i in range(len(groups)):
        colors += [color_sequence[i] for _ in groups[i]]
    u = fit.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(u[:, 0], u[:, 1], c=colors)
    if as_png:
        plt.savefig(prefix + ".png", dpi=300)
    else:
        plt.savefig(prefix + ".pdf")
    plt.close()
    plt.clf()


# TODO: make full workthrough a class or function
# see mackey_2label_test, see if possible to support more than 2 groups.
# see F-statistic
