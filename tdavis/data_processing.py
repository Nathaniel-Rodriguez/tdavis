__all__ = ['flatten_layer', 'pre_process_layer_states', 'bin_time_series',
           'binarize_time_series', 'plot_state_distribution', 'pattern_plot',
           'max_binary', 'jensenshannon', 'individual_state_plots',
           'embed_time_series', 'plot_embedding', 'layer_preprocessing']


from typing import Tuple
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import rel_entr


def jensenshannon(p, q, base=None):
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=0)
    q = q / np.sum(q, axis=0)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    js = np.sum(left, axis=0) + np.sum(right, axis=0)
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)


def embed_time_series(x: np.ndarray,
                      tau: int,
                      dimensions: int) -> np.ndarray:
    """
    :param x: 1-D time-series (length N)
    :param tau: number of time-steps back for delay
    :param dimensions: dimensionality of the embedding
    :return: (T - dimensions*tau) x (dimensions)
    """
    embedding = np.zeros(shape=(len(x) - dimensions * tau, dimensions))
    for dim in range(embedding.shape[1]):
        embedding[:, dim] = x[dimensions * tau - dim * tau:len(x) - dim * tau]

    return embedding


def plot_embedding(embedding: np.ndarray,
                   tau: int,
                   dimensions: Tuple[int, int],
                   prefix: str,
                   as_png=False):
    """
    makes a plot of a time-series embedding
    :param embedding: a 2D array, first dim is time, second is dim
    :param tau: the delay size (for labelling)
    :param dimensions: tuple of first dimension vs second dimension
    :param prefix: filename prefix
    :param as_png: whether to write as a png file (default: true)
    """
    plt.scatter(embedding[:, dimensions[0]],
                embedding[:, dimensions[1]],
                s=1.0)
    plt.xlabel("x(t-" + str(dimensions[0] * tau) + ")")
    plt.ylabel("x(t-" + str(dimensions[1] * tau) + ")")
    if as_png:
        plt.savefig(prefix + ".png", dpi=300)
    else:
        plt.savefig(prefix + ".pdf")
    plt.clf()
    plt.close()


# TODO: remove neurons with inactive downsteam neighbors


def max_binary(layer_state):
    # difference time-series
    differenced_states = layer_state[1:, :] - layer_state[-1:, :]
    # calc variance of each neuron
    variances = np.var(differenced_states, axis=0)
    # remove no-variance neurons
    active_indices = [i for i, v in enumerate(variances) if v > 0.00001]
    filtered_layer_state = np.take(layer_state, active_indices, axis=1)

    binary_layer_state = np.zeros((filtered_layer_state.shape[0],
                                   filtered_layer_state.shape[1]), dtype=bool)
    for t in range(filtered_layer_state.shape[0]):
        max_index = np.argmax(filtered_layer_state[t])
        binary_layer_state[t, max_index] = True

    return binary_layer_state


def flatten_layer(layer_state):
    return layer_state.reshape(layer_state.shape[0], -1)


def _differenced_preprocessing(layer_state, filter_lowest=0.2):
    # difference time-series
    differenced_states = layer_state[1:, :] - layer_state[-1:, :]

    # calc variance of each neuron
    variances = np.var(differenced_states, axis=0)
    # remove no-variance neurons
    active_indices = [i for i, v in enumerate(variances) if v > 0.00001]
    print("removing...", len(variances) - len(active_indices),
          " neurons for inactivity")
    filtered_states = np.take(differenced_states, active_indices, axis=1)
    # remove low-variance neurons
    variances = np.var(filtered_states, axis=0)
    indices_l2g = np.argsort(variances)
    chosen = indices_l2g[int(filter_lowest * len(indices_l2g)):]
    print("removing...", len(variances) - len(chosen),
          " neurons for low activity")
    filtered_states = np.take(filtered_states, chosen, axis=1)
    # standardize whole set
    if filtered_states.shape[1] >= 1:
        scaler = sk.preprocessing.StandardScaler()
        scaled_states = scaler.fit_transform(filtered_states)
    else:
        scaled_states = filtered_states
    print("\tremaining states:", scaled_states.shape[1])
    return scaled_states


def _preprocessing(layer_state, filter_lowest=0.2):
    # difference time-series
    differenced_states = layer_state[1:, :] - layer_state[-1:, :]

    # calc variance of each neuron
    variances = np.var(differenced_states, axis=0)
    # remove no-variance neurons
    active_indices = [i for i, v in enumerate(variances) if v > 0.00001]
    print("removing...", len(variances) - len(active_indices),
          " neurons for inactivity")
    filtered_states = np.take(differenced_states, active_indices, axis=1)
    filtered_layer_state = np.take(layer_state, active_indices, axis=1)
    # remove low-variance neurons
    variances = np.var(filtered_states, axis=0)
    indices_l2g = np.argsort(variances)
    chosen = indices_l2g[int(filter_lowest * len(indices_l2g)):]
    print("removing...", len(variances) - len(chosen),
          " neurons for low activity")
    filtered_states = np.take(filtered_states, chosen, axis=1)
    filtered_layer_state = np.take(filtered_layer_state, chosen, axis=1)
    # standardize whole set
    if filtered_states.shape[1] >= 1:
        scaler = sk.preprocessing.StandardScaler()
        scaled_states = scaler.fit_transform(filtered_layer_state)
    else:
        scaled_states = filtered_layer_state
    print("\tremaining states:", scaled_states.shape[1])
    return scaled_states


def pre_process_layer_states(layer_state, filter_lowest=0.2, differenced=True):
    """
    Differences the time-series, removes the neurons with low
    variance and then rescales the rest.
    :param layer_state: a TxN matrix where N is the number of neurons.
    :param filter_lowest: remove lowest X percent AFTER neurons with
    no variance are removed.
    :return: a TxA matrix where A is the remaining number of neurons
    """
    if differenced:
        return _differenced_preprocessing(layer_state, filter_lowest)
    else:
        return _preprocessing(layer_state, filter_lowest)


def layer_preprocessing(layer_state):
    # difference time-series
    differenced_states = layer_state[1:, :] - layer_state[-1:, :]

    # calc variance of each neuron
    variances = np.var(differenced_states, axis=0)
    # remove no-variance neurons
    active_indices = [i for i, v in enumerate(variances) if v > 0.00001]
    filtered_layer_state = np.take(layer_state, active_indices, axis=1)
    # if everything filtered out, then leave as is
    if filtered_layer_state.shape[1] == 0:
        filtered_layer_state = layer_state
    scaler = sk.preprocessing.StandardScaler()
    scaled_states = scaler.fit_transform(filtered_layer_state)
    return scaled_states


def bin_time_series(layer_state, window_size=5, method="avg"):
    """
    bins the activity in a non-rolling window of a given size by a chosen method
    :param layer_state: TxN matrix where N is the number of neurons.
    :param window_size: size in time-steps to bin
    :param method: default "avg" ("max" for largest value)
    :return: txN matrix where t is the size of the time-series post binning
    """
    num_bins = int(layer_state.shape[0] / window_size)
    binned_state = np.zeros((num_bins, layer_state.shape[1]))
    if method == "avg":
        for i in range(num_bins):
            binned_state[i, :] = np.mean(layer_state[i * window_size:(i+1)*window_size, :],
                                         axis=0)

    elif method == "max":
        for i in range(num_bins):
            binned_state[i, :] = np.max(layer_state[i * window_size:(i+1)*window_size, :],
                                        axis=0)
    else:
        raise NotImplementedError
    print("new time end:", binned_state.shape[0])
    return binned_state


def binarize_time_series(layer_state, threshold, absolute=False):
    """
    Given a threshold return a binary matrix of 0's or 1's based on threshold
    :param layer_state: TxN matrix where N is the number of neurons.
    :param threshold: threshold above which the value is considered 1.
    :param absolute: whether to threshold by absolute value (default: False)
    :return: binarized matrix of shape TxN
    """
    if absolute:
        return sk.preprocessing.binarize(np.abs(layer_state), threshold)
    else:
        return sk.preprocessing.binarize(layer_state, threshold)


def plot_state_distribution(layer, prefix="test", plot_min_max=True,
                            png=False):
    """
    Makes a time series plot of the max/min/median/90%/10% quartile ranges
    :param layer: a TxN matrix, where N is the number of neurons
    :param prefix: prefix for output file name
    :param plot_min_max: if True plots min/max values (default: True)
    :return: None
    """
    plt.clf()

    mins = []
    maxes = []
    tops = []
    bots = []
    centers = []
    for state in layer:

        # Max, min
        mins.append(min(state))
        maxes.append(max(state))

        # Percentile 90, 10, 50
        tops.append(np.percentile(state, 90))
        bots.append(np.percentile(state, 10))
        centers.append(np.percentile(state, 50))

    plt.plot(centers, color="black", marker="None", lw=1, label='median')
    plt.fill_between(range(len(tops)), tops, bots, color="red", alpha=0.3,
                     label='10%/90%')
    if plot_min_max:
        plt.plot(mins, ls="dotted", marker="None", color="black", lw=1,
                 label='min/max')
        plt.plot(maxes, ls="dotted", marker="None", color="black", lw=1)
    plt.xlim(0, len(tops))
    plt.legend()
    if png:
        plt.savefig(prefix + "_state_distribution.png", dpi=300)
    else:
        plt.savefig(prefix + "_state_distribution.pdf")
    plt.clf()
    plt.close()


def pattern_plot(layer, prefix="test", window=None, xlabel="time",
                 ylabel="activation", png=False):
    """
    :param layer: TxN activation matrix
    :param prefix: for file name
    :param xlabel: name for T dimension (default: time)
    :param ylabel: name for N dimension (default: activation)
    :return: None
    """
    plt.clf()
    fig, ax = plt.subplots()
    cmap = mpl.colors.ListedColormap(['w', 'k'])
    bounds = [0, 0.5, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    if window is None:
        ax.imshow(layer.T, interpolation='none', cmap=cmap, norm=norm)
    else:
        ax.imshow(layer.T[:, window[0]:window[1]],
                  interpolation='none', cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if png:
        plt.savefig(prefix + "_pattern.png", dpi=300)
    else:
        plt.savefig(prefix + "_pattern.pdf")
    plt.clf()
    plt.close()


def individual_state_plots(history, layer, prefix="test"):
    flat_layer = flatten_layer(history[layer])
    processed_layer = pre_process_layer_states(flat_layer, 0, False)
    for i in range(processed_layer.shape[1]):
        plt.plot(processed_layer[:, i])
        plt.savefig(str(i) + "_" + prefix + "_state_ts.png", dpi=300)
        plt.close()
        plt.clf()
