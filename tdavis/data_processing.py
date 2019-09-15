__all__ = ['flatten_layer', 'pre_process_layer_states', 'bin_time_series',
           'binarize_time_series']


import numpy as np
import sklearn as sk


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
    scaler = sk.preprocessing.RobustScaler()
    scaled_states = scaler.fit_transform(filtered_states)
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
            binned_state[i, :] = np.mean(layer_state[i * num_bins:(i+1)*num_bins, :],
                                         axis=0)

    elif method == "max":
        for i in range(num_bins):
            binned_state[i, :] = np.max(layer_state[i * num_bins:(i+1)*num_bins, :],
                                        axis=0)
    else:
        raise NotImplementedError

    return binned_state


def binarize_time_series(layer_state, threshold):
    """
    Given a threshold return a binary matrix of 0's or 1's based on threshold
    :param layer_state: TxN matrix where N is the number of neurons.
    :param threshold: threshold above which the value is considered 1.
    :return: binarized matrix of shape TxN
    """
    return sk.preprocessing.binarize(layer_state, threshold)
