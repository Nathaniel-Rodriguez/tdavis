__all__ = ['flatten_layer', 'pre_process_layer_states', 'bin_time_series',
           'binarize_time_series', 'plot_state_distribution', 'pattern_plot']


import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib as mpl


# TODO: remove neurons with inactive downsteam neighbors

def flatten_layer(layer_state):
    return layer_state.reshape(layer_state.shape[0], -1)


def pre_process_layer_states(layer_state, filter_lowest=0.2):
    """
    Differences the time-series, removes the neurons with low
    variance and then rescales the rest.
    :param layer_state: a TxN matrix where N is the number of neurons.
    :param filter_lowest: remove lowest X percent AFTER neurons with
    no variance are removed.
    :return: a TxA matrix where A is the remaining number of neurons
    """
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
    scaler = sk.preprocessing.StandardScaler()
    scaled_states = scaler.fit_transform(filtered_states)
    print("\tremaining states:", scaled_states.shape[1])
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


def plot_state_distribution(layer, prefix="test", plot_min_max=True):
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
    plt.savefig(prefix + "_state_distribution.pdf")
    plt.clf()
    plt.close()


def pattern_plot(layer, prefix="test", window=None, xlabel="time",
                 ylabel="activation"):
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
    plt.savefig(prefix + "_pattern.pdf")
    plt.clf()
    plt.close()

